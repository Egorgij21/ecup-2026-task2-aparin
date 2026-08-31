"""Есть ли в визуальных эмбеддингах НЕЗАВИСИМЫЙ сигнал?

Критерий задан заранее, до просмотра результата:
  1. корреляция с LoRA заметно ниже 0.73 (столько у TF-IDF, который в бленде работает);
  2. бленд с LoRA лучше, чем LoRA одна;
  3. тройной бленд (TF-IDF + картинки + LoRA) лучше текущего парного.

Пункт 1 — главный. Замер компонентов показал, что сильный, но коррелированный
компонент даёт в бленде ХУЖЕ слабого независимого: сила 0.6716 < 0.6910 < 0.7297
дала бленды 0.8615 > 0.8439 > 0.8098, порядок строго обратный.

Всё вложенно, полный OOF.
"""
import glob
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

E = np.load(ROOT + "exp/img_emb.npy")
idx = pd.read_parquet(ROOT + "exp/img_emb_index.parquet")
print(f"визуальные эмбеддинги: {E.shape}, нулевых (нет картинок): {(np.abs(E).sum(1) == 0).sum()}")

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
assert (df["id"].values == idx["id"].values).all(), "порядок строк не совпал"
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
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
    m = (df["category"] == cat).values
    sub = df[m].reset_index(drop=True)
    Ecat = E[m]
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)

    # 1) только картинки
    img = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        img[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")\
            .fit(Ecat[tr], y[tr]).predict_proba(Ecat[te])[:, 1]

    # 2) текстовый компонент из сабмита
    txt = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]),
                      csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]),
                      csr_matrix(L[te])]).tocsr()
        txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

    print(f"  {'источник':22s} {'F1':>7s} {'AUC':>7s} {'PR':>7s} {'corr с LoRA':>12s}")
    for tag, o in [("картинки", img), ("текст (в сабмите)", txt), ("LoRA", lo)]:
        c = np.corrcoef(o, lo)[0, 1] if tag != "LoRA" else 1.0
        print(f"  {tag:22s} {nested(o, y, fold):7.4f} {roc_auc_score(y, o):7.4f} "
              f"{average_precision_score(y, o):7.4f} {c:12.3f}")
    print(f"  корреляция картинки x текст: {np.corrcoef(img, txt)[0, 1]:.3f}")

    print("  --- бленды ---")
    variants = {
        "текст + LoRA (сабмит)": 0.5 * txt + 0.5 * lo,
        "картинки + LoRA": 0.5 * img + 0.5 * lo,
        "текст + картинки + LoRA": (txt + img + lo) / 3,
        "0.4 текст + 0.2 картинки + 0.4 LoRA": 0.4 * txt + 0.2 * img + 0.4 * lo,
        "0.35 текст + 0.15 картинки + 0.5 LoRA": 0.35 * txt + 0.15 * img + 0.5 * lo,
    }
    for tag, o in variants.items():
        f = nested(o, y, fold)
        summary[(cat, tag)] = f
        print(f"  {tag:38s} F1={f:.4f} AUC={roc_auc_score(y, o):.4f} "
              f"PR={average_precision_score(y, o):.4f}", flush=True)

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ (вложенно)")
tags = sorted({t for (_, t) in summary})
for tag in tags:
    vals = [summary[(c, tag)] for c in ["Легковоспламеняющиеся", "БАД"]]
    print(f"  {tag:40s} {np.mean(vals):.4f}   (флам {vals[0]:.4f}, БАД {vals[1]:.4f})")
print("\n  цель: 0.92 | текущий сабмит: вложенно ~0.900 -> LB 0.87820")
