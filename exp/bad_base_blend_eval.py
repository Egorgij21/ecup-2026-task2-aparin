"""Декорреляция ПО БАЗАМ на БАД: бленд gemma-картинок (_mmtp=v20) и Qwen-картинок
(_mmqwen2), оба с промптом badimg — отличие РОВНО в базе. Единственный подтверждённый
рычаг (флам две базы: corr 0.5 -> +0.009). БАД измерим (CI ±0.009, совпадает с пабликом).

Решающее число — КОРРЕЛЯЦИЯ _mmtp × _mmqwen2. Если низкая (как флам ~0.5) — бленд
должен дать прирост; если высокая (0.9+) — базы тут не декоррелируют.

Критерий (ДО замера): бленд бьёт v20, если F1(БАД)@0.47 > 0.9487 и парный bootstrap
по семьям P(>v20)>0.9.

Запуск: python exp/bad_base_blend_eval.py
"""
import os
import sys

import numpy as np
import pandas as pd
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


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    paths = [ROOT + f"exp/lora_oof_fold{k}{tag}.parquet" for k in range(5)]
    miss = [k for k in range(5) if not os.path.exists(paths[k])]
    if miss:
        return None, miss
    return pd.concat([pd.read_parquet(p)[["id", "lora_score"]] for p in paths],
                     ignore_index=True).drop_duplicates("id"), []


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


lo = {}
for t in ["_mmtp", "_mmqwen2", "_mm1"]:
    g, miss = load_tag(t)
    if g is None:
        print(f"[{t}] нет фолдов {miss} — пропускаю")
        continue
    lo[t] = sub.merge(g.rename(columns={"lora_score": "s"}), on="id", how="left")["s"].values

print(f"\nБАД n={len(y)} поз={y.sum()} порог {THR}")
if "_mmtp" in lo and "_mmqwen2" in lo:
    import scipy.stats as st
    c = st.spearmanr(lo["_mmtp"], lo["_mmqwen2"]).correlation
    print(f"\n*** КОРРЕЛЯЦИЯ gemma(_mmtp) × Qwen(_mmqwen2) = {c:.3f} ***")
    print("    (флам две базы 0.5 -> +0.009; промпты 0.97 -> ноль)")

scored = {}
combos = [("v20 gemma(_mmtp)", ["_mmtp"]), ("Qwen(_mmqwen2)", ["_mmqwen2"]),
          ("gemma+Qwen картинки", ["_mmtp", "_mmqwen2"])]
print()
for name, comps in combos:
    if any(c not in lo for c in comps):
        continue
    loa = np.mean([lo[c] for c in comps], axis=0)
    pred = (0.5 * txt + 0.5 * loa >= THR).astype(int)
    scored[name] = (f1_score(y, pred), boot(pred))
    print(f"  {name:22s} F1={scored[name][0]:.4f}")

if "v20 gemma(_mmtp)" in scored:
    base = scored["v20 gemma(_mmtp)"][1]
    print("\n=== парная разница против v20 (bootstrap семей) ===")
    for name, (f, v) in scored.items():
        if name == "v20 gemma(_mmtp)":
            continue
        d = v - base
        dl, dh = np.percentile(d, [2.5, 97.5])
        print(f"  {name:22s} ΔF1={f-scored['v20 gemma(_mmtp)'][0]:+.4f} "
              f"CI[{dl:+.4f},{dh:+.4f}] P(>v20)={np.mean(d>0):.2f}")
