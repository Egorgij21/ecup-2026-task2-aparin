"""Разбор LoRA на fold 0: устойчивость порога и стыковка с ансамблем.

Fold 0 честный — адаптер обучался на фолдах 1-4 и этих семей не видел.
Но порог, подобранный на fold 0, сам по себе оптимистичен, поэтому смотрим
в первую очередь на ШИРИНУ ПЛАТО (устойчивость), а не на пиковый F1:
именно узкое плато погубило SVC.
"""
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

lora = pd.read_parquet(ROOT + "exp/lora_oof_fold0.parquet")[["id", "lora_score"]]
df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
qq = pd.read_parquet(ROOT + "exp/llm_questions.parquet")
qcols = [c for c in qq.columns if c.startswith("q_")]
df = df.merge(qq[["id"] + qcols], on="id", how="left")
df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def prob_feats(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def plateau(o, y, frac=0.97):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    good = THS[f >= f.max() * frac]
    return f.max(), THS[int(f.argmax())], good.min(), good.max()


for cat in ["Легковоспламеняющиеся", "БАД"]:
    m = (df["category"] == cat).values
    sub = df[m].reset_index(drop=True)
    y_all = sub["label"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = prob_feats(sub["llm_score"].values)
    cols = [c for c in qcols if sub[c].notna().any()]
    Q = np.hstack([prob_feats(sub[c].fillna(0.5).values) for c in cols])
    fold = sub["fold"].values
    tr, te = np.where(fold != 0)[0], np.where(fold == 0)[0]

    # ансамбль «tfidf+rules+вопросы» (лучший из вложенной оценки), обучен на фолдах 1-4
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]), csr_matrix(Q[tr])]).tocsr()
    Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]), csr_matrix(Q[te])]).tocsr()
    ens = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(Xtr, y_all[tr]).predict_proba(Xte)[:, 1]

    ids_te = sub["id"].values[te]
    lo = pd.DataFrame({"id": ids_te}).merge(lora, on="id", how="left")["lora_score"].values
    y = y_all[te]
    print(f"\n########## {cat}: fold0 n={len(y)} pos={y.sum()}")
    if np.isnan(lo).any():
        print("  ВНИМАНИЕ: у части строк нет LoRA-скора:", int(np.isnan(lo).sum()))
        lo = np.nan_to_num(lo, nan=0.5)

    variants = {"ансамбль (tfidf+rules+вопросы)": ens, "LoRA": lo,
                "0.5*ансамбль+0.5*LoRA": 0.5 * ens + 0.5 * lo,
                "0.3*ансамбль+0.7*LoRA": 0.3 * ens + 0.7 * lo}
    for tag, o in variants.items():
        f, t, lo_t, hi_t = plateau(o, y)
        print(f"  {tag:32s} AUC={roc_auc_score(y,o):.4f} PR={average_precision_score(y,o):.4f} "
              f"F1={f:.4f}@{t:.2f}  плато[{lo_t:.2f}..{hi_t:.2f}] ширина={hi_t-lo_t:.2f}")

    print("  --- устойчивость LoRA к сдвигу порога ---")
    f_lo = np.array([f1_score(y, (lo >= t).astype(int)) for t in THS])
    best_t = THS[int(f_lo.argmax())]
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        print(f"     t={t:.2f}: F1={f1_score(y,(lo>=t).astype(int)):.4f} "
              f"предсказано поз.={int((lo>=t).sum())}")
    print(f"     оптимум t={best_t:.2f}, истинно позитивов {y.sum()}")
    print(f"  --- распределение скоров LoRA: p50={np.median(lo):.4f} "
          f"p90={np.percentile(lo,90):.4f} p99={np.percentile(lo,99):.4f}")
