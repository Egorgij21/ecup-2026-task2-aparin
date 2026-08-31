"""Оценка текстовых эмбеддингов на той же group-CV + ансамбль с TF-IDF."""
import re
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

ROOT = "/workspace/counter/"
EMB = sys.argv[1]
_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def clean(s):
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip().lower()


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["txt"] = [((clean(n) + " ") * 3 + clean(d)).strip()
             for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
df["gid"] = df.groupby(df["txt"]).ngroup()
E = np.load(EMB)
print(f"эмбеддинги {EMB}: {E.shape}\n")


def report(tag, y, oof):
    ths = np.linspace(0.05, 0.95, 91)
    fb = max((f1_score(y, (oof >= t).astype(int)), t) for t in ths)
    fm = max((f1_score(y, (oof >= t).astype(int), average="macro"), t) for t in ths)
    print(f"    {tag:16s} AUC={roc_auc_score(y, oof):.4f} PR={average_precision_score(y, oof):.4f} "
          f"F1bin={fb[0]:.4f}@{fb[1]:.2f}  F1mac={fm[0]:.4f}@{fm[1]:.2f}")
    return fb[0], fm[0]


agg = {}
for cat in ["БАД", "Легковоспламеняющиеся"]:
    m = (df["category"] == cat).values
    sub = df[m].reset_index(drop=True)
    X, y, g = sub["txt"].values, sub["label"].values, sub["gid"].values
    Xe = E[m]
    oof_t = np.zeros(len(y))
    oof_e = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        c = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")
        c.fit(v.fit_transform(X[tr]), y[tr])
        oof_t[te] = c.predict_proba(v.transform(X[te]))[:, 1]

        ce = LogisticRegression(max_iter=5000, C=10.0, class_weight="balanced")
        ce.fit(Xe[tr], y[tr])
        oof_e[te] = ce.predict_proba(Xe[te])[:, 1]

    print(f"### {cat}: n={len(y)} pos={y.sum()}")
    rt = report("TF-IDF", y, oof_t)
    re_ = report("эмбеддинги", y, oof_e)
    best = None
    for w in np.linspace(0, 1, 21):
        blend = (1 - w) * oof_t + w * oof_e
        ths = np.linspace(0.05, 0.95, 91)
        fb = max(f1_score(y, (blend >= t).astype(int)) for t in ths)
        fm = max(f1_score(y, (blend >= t).astype(int), average="macro") for t in ths)
        if best is None or fb + fm > best[0]:
            best = (fb + fm, w, fb, fm)
    print(f"    ЛУЧШИЙ БЛЕНД w_emb={best[1]:.2f}: F1bin={best[2]:.4f} F1mac={best[3]:.4f}")
    agg[cat] = dict(tfidf=rt, emb=re_, blend=(best[2], best[3]), w=best[1])
    print()

print("=" * 80)
for k, i in [("TF-IDF", "tfidf"), ("эмбеддинги", "emb"), ("бленд", "blend")]:
    mb = np.mean([agg[c][i][0] for c in agg])
    mm = np.mean([agg[c][i][1] for c in agg])
    print(f"{k:12s} mean_bin={mb:.4f} mean_mac={mm:.4f}")
