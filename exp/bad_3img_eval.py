"""3 картинки на БАД (_mm3tp2) vs v20 (_mmtp, 1 картинка) — НОВЫЙ сигнал (оборот упаковки).
Фикс. порог 0.47. Отличие ровно одно: IMAGES 1->3. Плюс проверяем бленд (вдруг 3-карт.
декоррелирован с 1-карт.). Критерий: бьёт v20, если F1@0.47 > 0.9487 и bootstrap P(>v20)>0.9.
"""
import os
import sys

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

BAD, THR, B = "БАД", 0.47, 3000
RNG = np.random.RandomState(0)


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
for t in ["_mmtp", "_mm3tp2"]:
    g = load_tag(t)
    if g is None:
        print(f"[{t}] не все 5 фолдов — стоп"); sys.exit(0)
    lo[t] = sub.merge(g.rename(columns={"lora_score": "s"}), on="id", how="left")["s"].values

fam = family_labels(sub["name"].fillna("").values)
fams = np.unique(fam); fam_idx = {f: np.where(fam == f)[0] for f in fams}
def boot(pred):
    out = np.empty(B)
    for b in range(B):
        pick = fams[RNG.randint(0, len(fams), len(fams))]
        idx = np.concatenate([fam_idx[f] for f in pick]); out[b] = f1_score(y[idx], pred[idx])
    return out
def blend(comps):
    loa = np.mean([lo[c] for c in comps], axis=0); pred = (0.5 * txt + 0.5 * loa >= THR).astype(int)
    return f1_score(y, pred), pred

print(f"\nБАД n={len(y)} поз={y.sum()} порог {THR}")
print(f"корр 1-карт(_mmtp) × 3-карт(_mm3tp2) = {st.spearmanr(lo['_mmtp'], lo['_mm3tp2']).correlation:.3f}\n")
base_f, base_pred = blend(["_mmtp"]); base_b = boot(base_pred)
for name, comps in [("v20 1 картинка", ["_mmtp"]), ("3 картинки", ["_mm3tp2"]),
                    ("бленд 1+3", ["_mmtp", "_mm3tp2"])]:
    f, pred = blend(comps)
    d = boot(pred) - base_b
    dl, dh = np.percentile(d, [2.5, 97.5])
    tag = "" if name == "v20 1 картинка" else f"  ΔF1={f-base_f:+.4f} CI[{dl:+.4f},{dh:+.4f}] P(>v20)={np.mean(d>0):.2f}"
    print(f"  {name:16s} F1={f:.4f}{tag}")
