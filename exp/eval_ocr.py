"""Дешёвая проверка: есть ли в OCR-тексте сигнал, которого нет в карточке?

Логика: прежде чем тратить ~13 часов на переобучение LoRA с OCR в промпте, смотрим
на CPU, добавляет ли OCR что-нибудь существующему текстовому ансамблю. Если не
добавляет даже там, где текст обрабатывается напрямую — в LoRA он тем более не поможет.

Всё вложенно: порог из фолдов != k. Наивные числа не приводим.
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

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
ocr = pd.read_parquet(ROOT + "exp/ocr.parquet")[["id", "ocr"]]
df = df.merge(ocr, on="id", how="left")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")

df["ocr"] = df["ocr"].fillna("").map(clean)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]
df["txt_ocr"] = df["txt"] + " [упаковка] " + df["ocr"]
df["fold"] = make_folds(df)

print(f"OCR непустой у {(df['ocr'].str.len() > 0).sum()}/{len(df)} товаров, "
      f"средняя длина {df.loc[df['ocr'].str.len() > 0, 'ocr'].str.len().mean():.0f} символов\n")


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
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    lo = sub["lora_score"].values
    print(f"########## {cat}: n={len(y)} pos={y.sum()}", flush=True)

    # отдельно: сколько сигнала В САМОМ OCR, без карточки
    variants = {
        "только OCR": ("ocr", []),
        "карточка (v2)": ("txt", [R, L]),
        "карточка + OCR": ("txt_ocr", [R, L]),
    }
    oofs = {}
    for tag, (col, blocks) in variants.items():
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                max_features=200_000)
            ptr = [v.fit_transform(sub[col].values[tr])]
            pte = [v.transform(sub[col].values[te])]
            for b in blocks:
                ptr.append(csr_matrix(b[tr]))
                pte.append(csr_matrix(b[te]))
            m = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")
            m.fit(hstack(ptr).tocsr(), y[tr])
            o[te] = m.predict_proba(hstack(pte).tocsr())[:, 1]
        oofs[tag] = o
        print(f"  {tag:22s} F1={nested(o, y, fold):.4f}  AUC={roc_auc_score(y, o):.4f} "
              f"PR={average_precision_score(y, o):.4f}", flush=True)

    # и главное: помогает ли OCR ПОВЕРХ LoRA (то, что реально в сабмите)
    print(f"  {'LoRA одна':22s} F1={nested(lo, y, fold):.4f}  AUC={roc_auc_score(y, lo):.4f}")
    for tag in ["карточка (v2)", "карточка + OCR"]:
        bl = 0.5 * oofs[tag] + 0.5 * lo
        f = nested(bl, y, fold)
        summary[(cat, tag)] = f
        print(f"  {'бленд 0.5*[' + tag + '] + 0.5*LoRA':44s} F1={f:.4f} "
              f"AUC={roc_auc_score(y, bl):.4f}", flush=True)
    print()

print("=" * 92)
print("МЕТРИКА СОРЕВНОВАНИЯ (бленд с LoRA, вложенно)")
for tag in ["карточка (v2)", "карточка + OCR"]:
    m = np.mean([summary[(c, tag)] for c in ["Легковоспламеняющиеся", "БАД"]])
    print(f"  {tag:22s} {m:.4f}")
print("\n  текущий сабмит blend v6: вложенно 0.9006 -> LB 0.87820")
