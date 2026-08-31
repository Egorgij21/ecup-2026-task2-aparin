"""В каком ПРОСТРАНСТВЕ смешивать текст и LoRA?

В сабмите стоит 0.5*p_текст + 0.5*p_LoRA по сырым вероятностям. Но шкалы у них
разные по построению: логрег обучен с class_weight="balanced" (смещён к 0.5),
LoRA — софтмакс по двум токенам (переуверенный, массы у 0 и 1). Среднее таких
вероятностей арифметически произвольно.

Проверяются альтернативы: логит-среднее, ранговое среднее (свободно от шкалы
вообще), геометрическое, а также предварительная калибровка каждого компонента
изотонической регрессией.

ГЛАВНОЕ ПРО ЧЕСТНОСТЬ. Выбирается не только порог, но и способ смешивания с весом —
значит вложенным должен быть ВЕСЬ выбор. Для фолда k правило (метод, вес, порог)
подбирается на фолдах != k и применяется к k. Иначе перебор 40+ вариантов на том же
OOF даст оптимистичный сдвиг больше самого эффекта — ровно так мы потеряли сабмит v3.
Для сравнения печатается и наивная оценка: разрыв между ними и есть цена перебора.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THS = np.linspace(0.005, 0.995, 199)
WS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CACHE = ROOT + "exp/txt_oof.parquet"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt_in"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_txt_oof(sub, cat):
    """OOF-скоры текстового компонента из сабмита (TF-IDF + правила + zero-shot)."""
    y, fold = sub["label"].values, sub["fold"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    o = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt_in"].values[tr]), csr_matrix(R[tr]),
                      csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt_in"].values[te]), csr_matrix(R[te]),
                      csr_matrix(L[te])]).tocsr()
        o[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return o


def make_combos(a, b, y, fold):
    """Все способы смешать два скора. Калибровка фитится на train-фолде каждого k."""
    out = {}
    for w in WS:
        out[f"prob w={w}"] = (1 - w) * a + w * b
        out[f"logit w={w}"] = sigm((1 - w) * logit(a) + w * logit(b))
        out[f"geom w={w}"] = np.exp((1 - w) * np.log(np.clip(a, 1e-6, 1)) +
                                    w * np.log(np.clip(b, 1e-6, 1)))
        ra = np.zeros(len(a))
        rb = np.zeros(len(b))
        for k in range(5):
            te = np.where(fold == k)[0]          # ранги внутри фолда: шкала фолда своя
            ra[te] = rankdata(a[te]) / len(te)
            rb[te] = rankdata(b[te]) / len(te)
        out[f"rank w={w}"] = (1 - w) * ra + w * rb
        ca, cb = np.zeros(len(a)), np.zeros(len(b))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            ca[te] = IsotonicRegression(out_of_bounds="clip").fit(a[tr], y[tr]).predict(a[te])
            cb[te] = IsotonicRegression(out_of_bounds="clip").fit(b[tr], y[tr]).predict(b[te])
        out[f"isoton w={w}"] = (1 - w) * ca + w * cb
    out["только текст"] = a.copy()
    out["только LoRA"] = b.copy()
    return out


def best_thr(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    i = int(f.argmax())
    return THS[i], f[i]


def nested_fixed(o, y, fold):
    """Правило зафиксировано заранее; вложенный только порог."""
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= best_thr(o[tr], y[tr])[0]).astype(int)
    return f1_score(y, pred)


def nested_select(combos, y, fold):
    """Вложенный выбор И правила, И порога. Честная цена перебора."""
    pred = np.zeros(len(y), dtype=int)
    picked = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        best = None
        for tag, o in combos.items():
            t, f = best_thr(o[tr], y[tr])
            if best is None or f > best[1]:
                best = (tag, f, t)
        tag, _, t = best
        picked.append(tag)
        pred[te] = (combos[tag][te] >= t).astype(int)
    return f1_score(y, pred), picked


summary = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    key = f"txt_{cat}"
    if os.path.exists(CACHE) and key in pd.read_parquet(CACHE).columns:
        txt = pd.read_parquet(CACHE)[key].dropna().values[:len(y)]
    else:
        txt = build_txt_oof(sub, cat)
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)
    print(f"  текст один: {nested_fixed(txt, y, fold):.4f} | "
          f"LoRA одна: {nested_fixed(lo, y, fold):.4f} | "
          f"corr={np.corrcoef(txt, lo)[0, 1]:.3f}")

    combos = make_combos(txt, lo, y, fold)
    rows = []
    for tag, o in combos.items():
        rows.append((tag, nested_fixed(o, y, fold), roc_auc_score(y, o),
                     average_precision_score(y, o)))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'правило':22s} {'F1(вложенный порог)':>20s} {'AUC':>8s} {'PR':>8s}")
    for tag, f, a, p in rows[:12]:
        print(f"  {tag:22s} {f:20.4f} {a:8.4f} {p:8.4f}")
    print(f"  ... худшее: {rows[-1][0]} {rows[-1][1]:.4f}")

    naive = rows[0][1]
    honest, picked = nested_select(combos, y, fold)
    summary[cat] = (naive, honest, rows[0][0], picked)
    print(f"  НАИВНО (лучшее из {len(combos)}): {naive:.4f} [{rows[0][0]}]")
    print(f"  ЧЕСТНО (правило тоже вложенно): {honest:.4f}")
    print(f"  цена перебора: {naive - honest:+.4f} | выбор по фолдам: {picked}", flush=True)
    print(f"  для справки, сабмит (prob w=0.5): {nested_fixed(combos['prob w=0.5'], y, fold):.4f}")

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ")
cats = ["Легковоспламеняющиеся", "БАД"]
print(f"  наивно (нельзя верить): {np.mean([summary[c][0] for c in cats]):.4f}")
print(f"  ЧЕСТНО:                 {np.mean([summary[c][1] for c in cats]):.4f}")
print("\n  цель: 0.915 | текущий сабмит: вложенно 0.9006 -> LB 0.87820")
