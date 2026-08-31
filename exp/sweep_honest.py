"""Перебор текстовых конфигов на ЧЕСТНОЙ (семейной) CV.

Прошлый перебор (exp/tune_tfidf.py) шёл на завышенной CV по точному тексту,
поэтому его выводы (например «name x3 + C=10») требуют перепроверки.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["fold"] = make_folds(df)
CACHE = {}


def llm_feats(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def blocks(sub, cfg, tr, te):
    """Собирает матрицы признаков для одного сплита."""
    txt = ((sub["name_c"] + " ") * cfg["name_rep"] + sub["desc_c"]).values
    parts_tr, parts_te = [], []
    v = TfidfVectorizer(ngram_range=cfg["word_ng"], min_df=cfg["min_df"],
                        sublinear_tf=True, max_features=200_000)
    parts_tr.append(v.fit_transform(txt[tr]))
    parts_te.append(v.transform(txt[te]))
    if cfg.get("char"):
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg["char"], min_df=3,
                             sublinear_tf=True, max_features=300_000)
        parts_tr.append(vc.fit_transform(txt[tr]))
        parts_te.append(vc.transform(txt[te]))
    if cfg.get("name_block"):
        vn = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        nm = sub["name_c"].values
        parts_tr.append(vn.fit_transform(nm[tr]))
        parts_te.append(vn.transform(nm[te]))
    if cfg.get("rules"):
        R = CACHE["R"]
        parts_tr.append(csr_matrix(R[tr]))
        parts_te.append(csr_matrix(R[te]))
    if cfg.get("llm"):
        L = CACHE["L"]
        parts_tr.append(csr_matrix(L[tr]))
        parts_te.append(csr_matrix(L[te]))
    return hstack(parts_tr).tocsr(), hstack(parts_te).tocsr()


def run(sub, cfg):
    y = sub["label"].values
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(sub["fold"] != k)[0], np.where(sub["fold"] == k)[0]
        Xtr, Xte = blocks(sub, cfg, tr, te)
        if cfg.get("svm"):
            m = LinearSVC(C=cfg["C"], class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
            d = m.decision_function(Xte)
            oof[te] = 1 / (1 + np.exp(-d))
        else:
            m = LogisticRegression(max_iter=4000, C=cfg["C"], class_weight="balanced").fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1]
    ths = np.linspace(0.02, 0.98, 193)
    fb = max((f1_score(y, (oof >= t).astype(int)), t) for t in ths)
    fm = max((f1_score(y, (oof >= t).astype(int), average="macro"), t) for t in ths)
    return dict(bin=fb[0], tb=fb[1], mac=fm[0], auc=roc_auc_score(y, oof),
                pr=average_precision_score(y, oof), oof=oof)


BASE = dict(name_rep=3, word_ng=(1, 2), min_df=2, C=10.0, rules=True, llm=True)
GRID = [
    ("текущий v2 (rep3,C10,rules,llm)", {}),
    ("name_rep=1", dict(name_rep=1)),
    ("name_rep=2", dict(name_rep=2)),
    ("name_rep=5", dict(name_rep=5)),
    ("C=3", dict(C=3.0)),
    ("C=30", dict(C=30.0)),
    ("+ отдельный блок по name", dict(name_block=True)),
    ("+ char_wb(3,5)", dict(char=(3, 5))),
    ("+ char_wb(2,4)", dict(char=(2, 4))),
    ("+ char + name_block", dict(char=(3, 5), name_block=True)),
    ("min_df=1", dict(min_df=1)),
    ("word_ng=(1,3)", dict(word_ng=(1, 3))),
    ("LinearSVC C=1", dict(svm=True, C=1.0)),
    ("LinearSVC C=0.3", dict(svm=True, C=0.3)),
]

results = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    CACHE["R"] = extract(sub["full"].values, sub["name_c"].values, cat)
    CACHE["L"] = llm_feats(sub["llm_score"].values)
    print(f"\n########## {cat}  n={len(sub)} pos={sub['label'].sum()}")
    rows = []
    for tag, upd in GRID:
        r = run(sub, {**BASE, **upd})
        rows.append((tag, r["bin"], r["mac"], r["auc"], r["pr"], r["tb"]))
        print(f"  {tag:34s} F1bin={r['bin']:.4f}@{r['tb']:.2f} F1mac={r['mac']:.4f} "
              f"AUC={r['auc']:.4f} PR={r['pr']:.4f}", flush=True)
    results[cat] = sorted(rows, key=lambda x: -x[1])

print("\n" + "=" * 100)
print("ЛУЧШИЕ ПО КАТЕГОРИЯМ (F1 binary, честная CV)")
best = {}
for cat, rows in results.items():
    print(f"\n{cat}:")
    for r in rows[:5]:
        print(f"   {r[1]:.4f}  {r[0]}")
    best[cat] = rows[0]
print(f"\nЕсли взять лучший конфиг в каждой категории: "
      f"mean F1bin = {np.mean([best[c][1] for c in best]):.4f}")
print("  (текущий v2 = 0.8308, цель = 0.86)")
