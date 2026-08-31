"""Разбор: почему SVC выиграл на CV, но проиграл на public LB.

Гипотезы:
  H1. Порог у SVC хрупкий — пик F1 узкий, малый сдвиг шкалы решающей функции
      сильно роняет метрику. У LogReg вероятности калиброваны, пик широкий.
  H2. Выигрыш SVC на CV — это в основном выигрыш «в точке оптимального порога»,
      а не в качестве ранжирования (AUC/PR почти не растут).
  H3. Масштаб decision_function у SVC зависит от выборки, поэтому порог,
      подобранный на трейне, не переносится на другое распределение.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

CAT = "Легковоспламеняющиеся"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]
df["fold"] = make_folds(df)

sub = df[df["category"] == CAT].reset_index(drop=True)
y = sub["label"].values
R = extract(sub["full"].values, sub["name_c"].values, CAT)
s = np.clip(sub["llm_score"].values, 1e-4, 1 - 1e-4)
L = np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float), (s > .8).astype(float), (s < .2).astype(float)]


def oof_for(kind):
    o = np.zeros(len(y))
    raw = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(sub["fold"] != k)[0], np.where(sub["fold"] == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
        if kind == "svc":
            m = LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0).fit(Xtr, y[tr])
            d = m.decision_function(Xte)
            raw[te] = d
            o[te] = 1 / (1 + np.exp(-d))
        else:
            m = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(Xtr, y[tr])
            o[te] = m.predict_proba(Xte)[:, 1]
            raw[te] = np.log(o[te] / (1 - o[te] + 1e-12) + 1e-12)
    return o, raw


res = {}
for kind in ["logreg", "svc"]:
    o, raw = oof_for(kind)
    res[kind] = (o, raw)
    print(f"\n### {kind}: AUC={roc_auc_score(y, o):.4f}  PR-AUC={average_precision_score(y, o):.4f}")

print("\n" + "=" * 88)
print("H2. КАЧЕСТВО РАНЖИРОВАНИЯ vs ВЫИГРЫШ В ТОЧКЕ ПОРОГА")
for kind in ["logreg", "svc"]:
    o, _ = res[kind]
    ths = np.linspace(0.02, 0.98, 193)
    f1s = np.array([f1_score(y, (o >= t).astype(int)) for t in ths])
    print(f"  {kind:7s} AUC={roc_auc_score(y,o):.4f} PR={average_precision_score(y,o):.4f} "
          f"maxF1={f1s.max():.4f}")

print("\n" + "=" * 88)
print("H1. ХРУПКОСТЬ ПОРОГА: насколько падает F1 при сдвиге порога от оптимума")
for kind in ["logreg", "svc"]:
    o, _ = res[kind]
    ths = np.linspace(0.02, 0.98, 193)
    f1s = np.array([f1_score(y, (o >= t).astype(int)) for t in ths])
    best_i = int(f1s.argmax())
    best_t, best_f = ths[best_i], f1s[best_i]
    # ширина плато: доля порогов, где F1 в пределах 2% от максимума
    wide = ths[f1s >= best_f * 0.98]
    print(f"\n  {kind}: оптимум t={best_t:.3f} F1={best_f:.4f}")
    print(f"     плато (F1 >= 98% от макс): t от {wide.min():.3f} до {wide.max():.3f} "
          f"— ширина {wide.max()-wide.min():.3f}")
    for dt in [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]:
        t = np.clip(best_t + dt, 0.01, 0.99)
        f = f1_score(y, (o >= t).astype(int))
        print(f"     сдвиг {dt:+.2f} -> t={t:.3f} F1={f:.4f}  ({(f-best_f)/best_f*100:+.1f}%)")

print("\n" + "=" * 88)
print("H3. КАЛИБРОВКА: сколько позитивов предсказано при выбранном пороге")
print("   (истинная доля позитивов = %.4f)" % y.mean())
for kind, thr in [("logreg", 0.365), ("svc", 0.415)]:
    o, _ = res[kind]
    p = (o >= thr).astype(int)
    print(f"  {kind:7s} t={thr}: предсказано позитивов {p.sum():4d} ({p.mean():.4f}) "
          f"prec={((p == 1) & (y == 1)).sum()/max(p.sum(),1):.3f} "
          f"rec={((p == 1) & (y == 1)).sum()/y.sum():.3f} F1={f1_score(y,p):.4f}")

print("\n" + "=" * 88)
print("H3b. РАСПРЕДЕЛЕНИЕ СКОРОВ: насколько плотно они жмутся к порогу")
for kind in ["logreg", "svc"]:
    o, raw = res[kind]
    print(f"  {kind:7s} decision: p1={np.percentile(raw,1):+.2f} p50={np.percentile(raw,50):+.2f} "
          f"p99={np.percentile(raw,99):+.2f} std={raw.std():.2f}")
    print(f"          доля скоров в окне ±0.1 вокруг порога: "
          f"{np.mean(np.abs(o - (0.415 if kind=='svc' else 0.365)) < 0.1):.4f}")

print("\n" + "=" * 88)
print("ВЫВОД ПО СТАБИЛЬНОСТИ: F1 при фиксированной ДОЛЕ предсказанных позитивов")
print("(если порог задавать квантилем, а не абсолютным значением — переносится ли лучше?)")
for kind in ["logreg", "svc"]:
    o, _ = res[kind]
    for q in [0.020, 0.025, 0.030, 0.035, 0.040]:
        t = np.quantile(o, 1 - q)
        p = (o >= t).astype(int)
        print(f"  {kind:7s} доля={q:.3f} -> F1={f1_score(y,p):.4f}")
