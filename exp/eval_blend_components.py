"""Какой текстовый компонент лучше блендить с LoRA?

Упущение: в сабмите стоит компонент «TF-IDF + правила + zero-shot» (нестед 0.6716
на флам), но есть вариант сильнее — с атомарными вопросами вместо zero-shot (0.6910).
Как самостоятельные модели они мерились, а в БЛЕНДЕ с LoRA — нет.

Важно: сильнее сам по себе не значит лучше в бленде. Проверка энкодеров показала,
что решает сочетание силы и декорреляции, поэтому смотрим и на корреляцию с LoRA.

Всё вложенно, полный OOF. CPU.
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
qq = pd.read_parquet(ROOT + "exp/llm_questions.parquet")
qcols = [c for c in qq.columns if c.startswith("q_")]
df = df.merge(qq[["id"] + qcols], on="id", how="left")
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
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    cols = [c for c in qcols if sub[c].notna().any()]
    Q = np.hstack([pf(sub[c].fillna(0.5).values) for c in cols])

    COMPONENTS = {
        "rules+zero-shot (в сабмите)": [R, L],
        "rules+вопросы": [R, Q],
        "rules+zero-shot+вопросы": [R, L, Q],
    }
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)
    print(f"  LoRA одна: F1={nested(lo, y, fold):.4f} AUC={roc_auc_score(y, lo):.4f}")

    oofs = {}
    for tag, blocks in COMPONENTS.items():
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                max_features=200_000)
            ptr = [v.fit_transform(sub["txt"].values[tr])]
            pte = [v.transform(sub["txt"].values[te])]
            for b in blocks:
                ptr.append(csr_matrix(b[tr]))
                pte.append(csr_matrix(b[te]))
            m = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")
            m.fit(hstack(ptr).tocsr(), y[tr])
            o[te] = m.predict_proba(hstack(pte).tocsr())[:, 1]
        oofs[tag] = o
        print(f"  компонент {tag:28s} F1={nested(o, y, fold):.4f} "
              f"AUC={roc_auc_score(o * 0 + o, y) if False else roc_auc_score(y, o):.4f} "
              f"corr с LoRA={np.corrcoef(o, lo)[0, 1]:.3f}", flush=True)

    print("  --- бленды с LoRA ---")
    for tag, o in oofs.items():
        best = None
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            bl = (1 - w) * o + w * lo
            f = nested(bl, y, fold)
            if best is None or f > best[0]:
                best = (f, w, bl)
        f, w, bl = best
        summary[(cat, tag)] = f
        print(f"  0.5/0.5 с [{tag:28s}] F1={nested(0.5 * o + 0.5 * lo, y, fold):.4f} | "
              f"лучший вес w_lora={w:.1f} -> F1={f:.4f} PR={average_precision_score(y, bl):.4f}",
              flush=True)

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ (лучший вес на категорию, вложенно)")
for tag in COMPONENTS:
    vals = [summary[(c, tag)] for c in ["Легковоспламеняющиеся", "БАД"]]
    print(f"  {tag:30s} {np.mean(vals):.4f}   (флам {vals[0]:.4f}, БАД {vals[1]:.4f})")
print("\n  текущий сабмит: вложенно 0.9006 -> LB 0.87820")
