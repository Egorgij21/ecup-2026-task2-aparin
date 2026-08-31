"""Декорреляция по МОДАЛЬНОСТИ на БАД: текст-LoRA vs картинки-LoRA. Их ошибки могут
расходиться сильнее, чем две картиночные базы (corr 0.915) — вход-то разный (текст vs
пиксели упаковки). Проверяем ВСЕ имеющиеся БАД-LoRA сигналы (OOF уже есть, замер бесплатный).

Сигналы:
  _mmtp   — gemma картинки + badimg (= v20, эталон)
  _mm1    — gemma картинки, обычный промпт (v16)
  _mmqwen2— Qwen картинки + badimg
  _gemma  — gemma ТЕКСТ (обучен на обеих категориях, OOF по БАД есть)
  (пусто) — Qwen ТЕКСТ

Конфиг сабмита: 0.5*текст-ансамбль + 0.5*(среднее выбранных LoRA), фикс. порог 0.47.
Критерий (ДО замера): бленд бьёт v20, если F1@0.47 > 0.9487 и парный bootstrap P(>v20)>0.9.
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, family_labels, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

BAD = "БАД"
THR = 0.47
B = 3000
RNG = np.random.RandomState(0)
TAGS = ["_mmtp", "_mm1", "_mmqwen2", "_gemma", ""]
LABEL = {"_mmtp": "gemma-img-badimg(v20)", "_mm1": "gemma-img-orig",
         "_mmqwen2": "qwen-img", "_gemma": "gemma-TEXT", "": "qwen-TEXT"}


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    paths = [ROOT + f"exp/lora_oof_fold{k}{tag}.parquet" for k in range(5)]
    if any(not os.path.exists(p) for p in paths):
        return None
    return pd.concat([pd.read_parquet(p)[["id", "lora_score"]] for p in paths],
                     ignore_index=True).drop_duplicates("id")


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]

sub = df[df["category"] == BAD].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
R = extract(sub["full"].values, sub["nc"].values, BAD)
L = pf(sub["llm_score"].values)
txt = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
    Bm = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, y[tr]).predict_proba(Bm)[:, 1]

lo = {}
for t in TAGS:
    g = load_tag(t)
    if g is not None:
        lo[t] = sub.merge(g.rename(columns={"lora_score": "s"}), on="id", how="left")["s"].values
present = [t for t in TAGS if t in lo]

fam = family_labels(sub["name"].fillna("").values)
fams = np.unique(fam)
fam_idx = {f: np.where(fam == f)[0] for f in fams}


def boot(pred):
    out = np.empty(B)
    for b in range(B):
        pick = fams[RNG.randint(0, len(fams), len(fams))]
        idx = np.concatenate([fam_idx[f] for f in pick])
        out[b] = f1_score(y[idx], pred[idx])
    return out


def f1_blend(comps):
    loa = np.mean([lo[c] for c in comps], axis=0)
    return f1_score(y, (0.5 * txt + 0.5 * loa >= THR)), (0.5 * txt + 0.5 * loa >= THR).astype(int)


print(f"БАД n={len(y)} поз={y.sum()} порог {THR}\n")
print("=== КОРРЕЛЯЦИИ LoRA-сигналов (spearman) ===")
print(f"{'':22s}" + "".join(f"{LABEL[t][:10]:>12s}" for t in present))
for a in present:
    row = "".join(f"{st.spearmanr(lo[a], lo[b]).correlation:12.3f}" for b in present)
    print(f"{LABEL[a]:22s}{row}")

print("\n=== одиночные F1@0.47 ===")
for t in present:
    print(f"  {LABEL[t]:26s} {f1_blend([t])[0]:.4f}")

# v20 эталон
base_f, base_pred = f1_blend(["_mmtp"])
base_boot = boot(base_pred)

print(f"\n=== БЛЕНДЫ (пары и тройки с _mmtp), парный bootstrap против v20={base_f:.4f} ===")
cands = []
for r in (2, 3):
    for combo in combinations(present, r):
        if "_mmtp" not in combo:
            continue
        f, pred = f1_blend(list(combo))
        cands.append((f, combo, pred))
for f, combo, pred in sorted(cands, key=lambda x: -x[0])[:10]:
    d = boot(pred) - base_boot
    p = np.mean(d > 0)
    names = "+".join(LABEL[c].split("(")[0][:8] for c in combo)
    print(f"  {names:34s} F1={f:.4f} ΔF1={f-base_f:+.4f} P(>v20)={p:.2f}")
