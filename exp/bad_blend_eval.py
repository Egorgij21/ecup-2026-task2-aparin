"""Замер БАД-адаптеров и их БЛЕНДОВ ПРОМПТОВ при ФИКС. пороге 0.47 на полном OOF.

Идея бленда промптов (декорреляция как v10 по базам): v20 (_mmtp), noexcl, sportbad —
три адаптера, обученные с РАЗНЫМИ промптами на одной базе+картинках. Если их ошибки
декоррелированы, среднее скоров может побить одиночный v20. БАД измеримо (CI ±0.009).

Критерий (задан ДО замера): вариант/бленд бьёт v20, если F1(БАД)@0.47 на полном OOF
выше 0.9487 и парный bootstrap по семьям даёт P(>v20)>0.9.

Запуск: python exp/bad_blend_eval.py
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
TAGS = ["_mmtp", "_mmnoexcl", "_mmsport"]   # v20, noexcl, sportbad


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    paths = [ROOT + f"exp/lora_oof_fold{k}{tag}.parquet" for k in range(5)]
    if any(not os.path.exists(p) for p in paths):
        have = [k for k in range(5) if os.path.exists(paths[k])]
        return None, have
    parts = [pd.read_parquet(p)[["id", "lora_score"]] for p in paths]
    return pd.concat(parts, ignore_index=True).drop_duplicates("id"), [0, 1, 2, 3, 4]


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]

sub0 = df[df["category"] == BAD].reset_index(drop=True)
y, fold = sub0["label"].values, sub0["fold"].values
R = extract(sub0["full"].values, sub0["nc"].values, BAD)
L = pf(sub0["llm_score"].values)
txt = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub0["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
    Bm = hstack([v.transform(sub0["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, y[tr]).predict_proba(Bm)[:, 1]

fam = family_labels(sub0["name"].fillna("").values)
fams = np.unique(fam)
fam_idx = {f: np.where(fam == f)[0] for f in fams}


def boot_pred(pred):
    vals = np.empty(B)
    for b in range(B):
        pick = fams[RNG.randint(0, len(fams), len(fams))]
        idx = np.concatenate([fam_idx[f] for f in pick])
        vals[b] = f1_score(y[idx], pred[idx])
    return vals


lo = {}
for t in TAGS:
    g, have = load_tag(t)
    if g is None:
        print(f"[{t}] готовы фолды {have} из 5 — пропускаю (нужны все)")
        continue
    lo[t] = sub0.merge(g.rename(columns={"lora_score": "s"}), on="id", how="left")["s"].values

print(f"\nБАД n={len(y)} поз={y.sum()} порог {THR}\n")
results = {}
scored = {}
for name, comps in [("v20(_mmtp)", ["_mmtp"]), ("noexcl", ["_mmnoexcl"]), ("sportbad", ["_mmsport"]),
                    ("v20+noexcl", ["_mmtp", "_mmnoexcl"]), ("v20+sport", ["_mmtp", "_mmsport"]),
                    ("noexcl+sport", ["_mmnoexcl", "_mmsport"]),
                    ("v20+noexcl+sport", ["_mmtp", "_mmnoexcl", "_mmsport"])]:
    if any(c not in lo for c in comps):
        continue
    loa = np.mean([lo[c] for c in comps], axis=0)
    score = 0.5 * txt + 0.5 * loa
    pred = (score >= THR).astype(int)
    f = f1_score(y, pred)
    results[name] = f
    scored[name] = boot_pred(pred)
    print(f"  {name:20s} F1={f:.4f}")

if "v20(_mmtp)" in scored:
    base = scored["v20(_mmtp)"]
    print(f"\n=== парная разница против v20 (bootstrap семей) ===")
    for name in results:
        if name == "v20(_mmtp)":
            continue
        d = scored[name] - base
        dl, dh = np.percentile(d, [2.5, 97.5])
        print(f"  {name:20s} ΔF1={results[name]-results['v20(_mmtp)']:+.4f} "
              f"CI[{dl:+.4f},{dh:+.4f}] P(>v20)={np.mean(d>0):.2f}")
