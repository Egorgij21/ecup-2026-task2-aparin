"""Проверка: пик бленда w=0.5 на флам — настоящий или шум?

Подозрительные признаки: кривая по весу рваная (0.74 -> 0.86 -> 0.81), а AUC при
этом ПАДАЕТ. Выигрыш в F1 при ухудшении ранжирования — это почти всегда попадание
порога в удачную точку, а не улучшение модели. Ровно так выглядел LinearSVC,
который дал +3.5 на CV и -0.9 на LB.

Смотрим по 4 независимым нарезкам семей. Порог всегда вложенный.
ВАЖНО: LoRA-скоры фиксированы (адаптеры обучены на нарезке seed=42), поэтому для
других сидов они частично «подсматривают» в свои обучающие фолды. Это завышает
абсолютные числа для сидов != 42 одинаково для всех вариантов, так что сравнение
МЕЖДУ вариантами остаётся честным.
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

SEEDS = [42, 7, 2024, 777]
THS = np.linspace(0.01, 0.99, 197)
CAT = "Легковоспламеняющиеся"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold*.parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]

sub = df[df["category"] == CAT].reset_index(drop=True)
y = sub["label"].values
lo = sub["lora_score"].values
R = extract(sub["full"].values, sub["name_c"].values, CAT)
s = np.clip(sub["llm_score"].values, 1e-4, 1 - 1e-4)
L = np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
          (s > .8).astype(float), (s < .2).astype(float)]


def pick(o, yy):
    f = np.array([f1_score(yy, (o >= t).astype(int)) for t in THS])
    return THS[int(f.argmax())]


def nested_f1(o, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= pick(o[tr], y[tr])).astype(int)
    return f1_score(y, pred)


WS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
res = {w: [] for w in WS}
aucs = {w: [] for w in WS}
res_fixed = []

for seed in SEEDS:
    fold = make_folds(df, seed=seed)[(df["category"] == CAT).values]
    text = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]),
                      csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]),
                      csr_matrix(L[te])]).tocsr()
        text[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    for w in WS:
        bl = (1 - w) * text + w * lo
        res[w].append(nested_f1(bl, fold))
        aucs[w].append(roc_auc_score(y, bl))
    res_fixed.append(f1_score(y, (lo >= 0.5).astype(int)))

print(f"флам, n={len(y)} pos={y.sum()}, сиды {SEEDS}\n")
print(f"{'w_lora':>7s} {'F1 по сидам':>36s} {'среднее':>9s} {'разброс':>8s} {'AUC':>8s}")
for w in WS:
    a = np.array(res[w])
    print(f"{w:7.2f} {str(np.round(a, 4)):>36s} {a.mean():9.4f} {a.std():8.4f} "
          f"{np.mean(aucs[w]):8.4f}")
print(f"\nLoRA одна, ФИКСИРОВАННЫЙ t=0.5: {np.round(res_fixed, 4)} "
      f"среднее={np.mean(res_fixed):.4f}")

best_w = max(WS, key=lambda w: float(np.mean(res[w])))
print(f"\nлучший вес в среднем по сидам: w={best_w:.2f} ({np.mean(res[best_w]):.4f})")
print(f"на сиде 42 в одиночку он давал: {res[best_w][0]:.4f}")
print("\nЕсли пик w=0.5 держится на всех сидах и AUC не проседает — сигнал настоящий.")
print("Если скачет — это шум операционной точки, и берём вариант с лучшим ранжированием.")
