"""Нужен ли ещё TF-IDF, и помогает ли OCR обучаемой модели.

Бленд держится не силой TF-IDF (он даёт 0.6716 на флам), а его декоррелированностью
с LoRA (корреляция предсказаний 0.732). Поэтому смотрим не только на силу компонента,
но и на корреляцию: более сильный, но более похожий на LoRA компонент может дать
в бленде МЕНЬШЕ.

Всё вложенно. Запуск: python exp/eval_encoder_blend.py
"""
import glob
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

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")

ENC_FILES = {
    "e5-base без OCR": "exp/enc_no_ocr.parquet",
    "e5-small без OCR": "exp/enc_small_no_ocr.parquet",
    "e5-small с OCR": "exp/enc_small_with_ocr.parquet",
}
available = {}
for tag, path in ENC_FILES.items():
    full = ROOT + path
    if os.path.exists(full):
        e = pd.read_parquet(full)[["id", "enc_score"]].rename(columns={"enc_score": tag})
        df = df.merge(e, on="id", how="left")
        available[tag] = tag
print("энкодеры в наличии:", list(available) or "нет ни одного")

df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def pick(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    return THS[int(f.argmax())]


def nested(o, y, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= pick(o[tr], y[tr])).astype(int)
    return f1_score(y, pred)


summary = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)

    # текстовый ансамбль v2 (то, что сейчас в бленде)
    tfidf = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]),
                      csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]),
                      csr_matrix(L[te])]).tocsr()
        tfidf[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

    comps = {"TF-IDF ансамбль (v2)": tfidf}
    for tag in available:
        if tag in sub.columns and sub[tag].notna().all():
            comps[tag] = sub[tag].values

    print(f"  {'компонент':26s} {'F1':>7s} {'AUC':>7s} {'PR':>7s} {'corr с LoRA':>12s}")
    for tag, o in comps.items():
        print(f"  {tag:26s} {nested(o, y, fold):7.4f} {roc_auc_score(y, o):7.4f} "
              f"{average_precision_score(y, o):7.4f} {np.corrcoef(o, lo)[0, 1]:12.3f}")
    print(f"  {'LoRA одна':26s} {nested(lo, y, fold):7.4f} {roc_auc_score(y, lo):7.4f} "
          f"{average_precision_score(y, lo):7.4f}")

    print("\n  --- блендим с LoRA 50/50 ---")
    for tag, o in comps.items():
        bl = 0.5 * o + 0.5 * lo
        f = nested(bl, y, fold)
        summary[(cat, tag)] = f
        print(f"  {tag:26s} F1={f:.4f}  AUC={roc_auc_score(y, bl):.4f}")
    # тройной бленд: TF-IDF + лучший энкодер + LoRA
    best_enc = max((t for t in comps if t != "TF-IDF ансамбль (v2)"),
                   key=lambda t: summary.get((cat, t), 0), default=None)
    if best_enc:
        tri = (tfidf + comps[best_enc] + lo) / 3
        f = nested(tri, y, fold)
        summary[(cat, "тройной бленд")] = f
        print(f"  {'TF-IDF + ' + best_enc + ' + LoRA':26s} F1={f:.4f} "
              f"AUC={roc_auc_score(y, tri):.4f}")

print("\n" + "=" * 92)
print("МЕТРИКА СОРЕВНОВАНИЯ (вложенно)")
tags = sorted({t for (_, t) in summary})
for tag in tags:
    vals = [summary.get((c, tag)) for c in ["Легковоспламеняющиеся", "БАД"]]
    if all(v is not None for v in vals):
        print(f"  {tag:34s} {np.mean(vals):.4f}")
print("\n  текущий сабмит blend v6: вложенно 0.9006 -> LB 0.87820")
