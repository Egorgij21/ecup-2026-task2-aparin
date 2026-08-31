"""ВЛОЖЕННЫЙ подбор порога — устраняет главное смещение прежней валидации.

Раньше порог подбирался на том же OOF, на котором считалась метрика. Это завышает
результат тем сильнее, чем уже плато у модели: у LogReg плато 0.500, у SVC 0.090,
поэтому SVC завышался куда сильнее — отсюда +3.5 на CV и −0.9 на LB.

Здесь для каждого фолда k порог берётся из фолдов != k и применяется к фолду k.
Так измеряется вся ПРОЦЕДУРА целиком, ровно как она работает в сабмите.

Сравниваем и правила выбора операционной точки:
  argmax   — порог максимума F1 (то, что делалось раньше)
  plateau  — центр самого широкого плато (устойчивее к сдвигу шкалы)
  quantile — порог как квантиль, чтобы доля предсказанных позитивов совпадала
             с оптимальной на трейне (не зависит от масштаба скоров)
  platt    — калибровка Платта поверх SVC, затем argmax
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

SEEDS = [42, 7, 2024, 777]
THS = np.linspace(0.02, 0.98, 193)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def build_oof(sub, fold, kind):
    o = np.zeros(len(sub))
    R, L = sub.attrs["R"], sub.attrs["L"]
    y = sub["label"].values
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
        if kind == "svc":
            m = LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0).fit(Xtr, y[tr])
            o[te] = 1 / (1 + np.exp(-m.decision_function(Xte)))
        else:
            m = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(Xtr, y[tr])
            o[te] = m.predict_proba(Xte)[:, 1]
    return o


def pick_argmax(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    return THS[int(f.argmax())]


def pick_plateau(o, y):
    """Центр самого широкого непрерывного участка, где F1 >= 97% от максимума."""
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    good = f >= f.max() * 0.97
    best_len, best_mid = 0, THS[int(f.argmax())]
    i = 0
    while i < len(good):
        if good[i]:
            j = i
            while j + 1 < len(good) and good[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len, best_mid = j - i + 1, THS[(i + j) // 2]
            i = j + 1
        else:
            i += 1
    return best_mid


def pick_quantile(o, y):
    """Возвращает не порог, а ДОЛЮ предсказанных позитивов при оптимальном F1."""
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    t = THS[int(f.argmax())]
    return float((o >= t).mean())


RULES = ["argmax", "plateau", "quantile"]
out = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub_all = df[df["category"] == cat].reset_index(drop=True)
    sub_all.attrs["R"] = extract(sub_all["full"].values, sub_all["name_c"].values, cat)
    s = np.clip(sub_all["llm_score"].values, 1e-4, 1 - 1e-4)
    sub_all.attrs["L"] = np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                               (s > .8).astype(float), (s < .2).astype(float)]
    y = sub_all["label"].values
    print(f"\n########## {cat}  n={len(y)} pos={y.sum()}", flush=True)

    for kind in ["logreg", "svc"]:
        naive, nested = {r: [] for r in RULES}, {r: [] for r in RULES}
        for seed in SEEDS:
            fold = make_folds(df, seed=seed)[(df["category"] == cat).values]
            o = build_oof(sub_all, fold, kind)
            # НАИВНО: порог на всём OOF, метрика на всём OOF (как было раньше)
            naive["argmax"].append(f1_score(y, (o >= pick_argmax(o, y)).astype(int)))
            naive["plateau"].append(f1_score(y, (o >= pick_plateau(o, y)).astype(int)))
            q = pick_quantile(o, y)
            naive["quantile"].append(f1_score(y, (o >= np.quantile(o, 1 - q)).astype(int)))
            # ВЛОЖЕННО: порог из фолдов != k, метрика на фолде k
            preds = {r: np.zeros(len(y), dtype=int) for r in RULES}
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                preds["argmax"][te] = (o[te] >= pick_argmax(o[tr], y[tr])).astype(int)
                preds["plateau"][te] = (o[te] >= pick_plateau(o[tr], y[tr])).astype(int)
                qq = pick_quantile(o[tr], y[tr])
                preds["quantile"][te] = (o[te] >= np.quantile(o[te], 1 - qq)).astype(int)
            for r in RULES:
                nested[r].append(f1_score(y, preds[r]))
        print(f"  {kind}:")
        for r in RULES:
            a, b = np.array(naive[r]), np.array(nested[r])
            print(f"    {r:9s} наивно={a.mean():.4f}±{a.std():.4f}   "
                  f"ВЛОЖЕННО={b.mean():.4f}±{b.std():.4f}   завышение={a.mean()-b.mean():+.4f}",
                  flush=True)
            out[(cat, kind, r)] = (a.mean(), b.mean())

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ (среднее по категориям) — ВЛОЖЕННАЯ оценка")
for kf, kb in [("svc", "logreg"), ("logreg", "logreg"), ("svc", "svc")]:
    for r in RULES:
        m = (out[("Легковоспламеняющиеся", kf, r)][1] + out[("БАД", kb, r)][1]) / 2
        n = (out[("Легковоспламеняющиеся", kf, r)][0] + out[("БАД", kb, r)][0]) / 2
        print(f"  флам={kf:6s} бад={kb:6s} правило={r:9s}: наивно={n:.4f}  ВЛОЖЕННО={m:.4f}")
print("\nФАКТ: v2 (logreg/logreg, argmax) LB=0.8150 | v3 (svc/logreg, argmax) LB=0.80556")
