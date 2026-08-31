"""Бленд TF-IDF-ансамбля с разными LoRA — то, что реально стоит в сабмите.

Прошлый замер сравнивал адаптеры МЕЖДУ СОБОЙ (корреляция 0.95, ансамбль бесполезен).
Но в сабмите работает другое: 0.5*TF-IDF + 0.5*LoRA, и держится оно на корреляции
0.732 между TF-IDF и LoRA. Поэтому gemma может быть бесполезна рядом с Qwen,
но полезна рядом с TF-IDF — это и проверяем.

Всё на fold 0, которого не видел ни один адаптер. ВАЖНО: 40 позитивов, поэтому
разница в пару пунктов F1 — шум (тот же конфиг с другим сидом дал разброс 4 пункта).
Смотрим в первую очередь на PR-AUC и корреляции, они устойчивее.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THS = np.linspace(0.01, 0.99, 197)
ADAPTERS = {
    "Qwen s0": "exp/lora_oof_fold0.parquet",
    "Qwen s777": "exp/lora_oof_fold0_s777.parquet",
    "gemma": "exp/lora_oof_fold0_gemma.parquet",
}

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
for tag, path in ADAPTERS.items():
    if os.path.exists(ROOT + path):
        d = pd.read_parquet(ROOT + path)[["id", "lora_score"]].rename(
            columns={"lora_score": tag})
        df = df.merge(d, on="id", how="left")
have = [t for t in ADAPTERS if t in df.columns]
print("адаптеры:", have)

df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def best_f1(y, s):
    return max(f1_score(y, (s >= t).astype(int)) for t in THS)


for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y_all, fold = sub["label"].values, sub["fold"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    s = np.clip(sub["llm_score"].values, 1e-4, 1 - 1e-4)
    L = np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
              (s > .8).astype(float), (s < .2).astype(float)]

    tr = np.where(fold != 0)[0]
    te = np.where(fold == 0)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]),
                  csr_matrix(L[tr])]).tocsr()
    Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]),
                  csr_matrix(L[te])]).tocsr()
    tfidf = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(Xtr, y_all[tr]).predict_proba(Xte)[:, 1]

    y = y_all[te]
    comps = {"TF-IDF ансамбль": tfidf}
    for t in have:
        col = sub[t].values[te]
        if not np.isnan(col).any():
            comps[t] = col
    print(f"\n########## {cat}: fold0 n={len(y)} pos={y.sum()}")
    print("  корреляция TF-IDF с адаптерами:")
    for t in have:
        if t in comps:
            print(f"     TF-IDF x {t}: {np.corrcoef(tfidf, comps[t])[0, 1]:.3f}")

    print(f"\n  {'вариант':46s} {'F1':>7s} {'AUC':>7s} {'PR':>7s}")
    for t, o in comps.items():
        print(f"  {t:46s} {best_f1(y, o):7.4f} {roc_auc_score(y, o):7.4f} "
              f"{average_precision_score(y, o):7.4f}")

    print("  --- бленды 50/50 с TF-IDF (то, что в сабмите) ---")
    for t in have:
        if t not in comps:
            continue
        bl = 0.5 * tfidf + 0.5 * comps[t]
        print(f"  {'TF-IDF + ' + t:46s} {best_f1(y, bl):7.4f} "
              f"{roc_auc_score(y, bl):7.4f} {average_precision_score(y, bl):7.4f}")

    print("  --- TF-IDF + несколько адаптеров (адаптеры усредняются) ---")
    ad = [t for t in have if t in comps]
    for r in range(2, len(ad) + 1):
        for combo in itertools.combinations(ad, r):
            mean_ad = np.mean([comps[t] for t in combo], axis=0)
            bl = 0.5 * tfidf + 0.5 * mean_ad
            name = "TF-IDF + (" + " + ".join(combo) + ")"
            print(f"  {name:46s} {best_f1(y, bl):7.4f} {roc_auc_score(y, bl):7.4f} "
                  f"{average_precision_score(y, bl):7.4f}")

    print("  --- равные веса на всё ---")
    for r in range(1, len(ad) + 1):
        for combo in itertools.combinations(ad, r):
            allc = [tfidf] + [comps[t] for t in combo]
            bl = np.mean(allc, axis=0)
            name = "равно: TF-IDF + " + " + ".join(combo)
            print(f"  {name:46s} {best_f1(y, bl):7.4f} {roc_auc_score(y, bl):7.4f} "
                  f"{average_precision_score(y, bl):7.4f}")
