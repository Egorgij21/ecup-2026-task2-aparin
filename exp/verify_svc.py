"""Проверка устойчивости находки «LinearSVC вместо LogReg» по нескольким сидам нарезки.

На флам всего 198 позитивов, а конфиг выбран как лучший из 14 на той же CV —
это классическая ловушка selection bias. Гоняем на 4 независимых нарезках семей.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

SEEDS = [42, 7, 2024, 777]

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def llm_feats(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def make_est(kind, C):
    if kind == "logreg":
        return LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
    if kind == "svc":
        return LinearSVC(C=C, class_weight="balanced", max_iter=8000)
    if kind == "svc_cal":
        return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", max_iter=8000),
                                      cv=3, method="sigmoid")
    raise ValueError(kind)


def score_of(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-est.decision_function(X)))


CONFIGS = [
    ("logreg C=10           ", "logreg", 10.0, False),
    ("svc    C=1            ", "svc", 1.0, False),
    ("svc    C=0.5          ", "svc", 0.5, False),
    ("svc    C=2            ", "svc", 2.0, False),
    ("svc    C=1 + char_wb  ", "svc", 1.0, True),
    ("svc_cal C=1           ", "svc_cal", 1.0, False),
    ("logreg C=10 + char_wb ", "logreg", 10.0, True),
]

for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub_all = df[df["category"] == cat].reset_index(drop=True)
    R = extract(sub_all["full"].values, sub_all["name_c"].values, cat)
    L = llm_feats(sub_all["llm_score"].values)
    y = sub_all["label"].values
    txt = sub_all["txt"].values
    print(f"\n########## {cat}  n={len(y)} pos={y.sum()}")
    table = {}
    for tag, kind, C, use_char in CONFIGS:
        per_seed = []
        for seed in SEEDS:
            fold = make_folds(df, seed=seed)[(df["category"] == cat).values]
            oof = np.zeros(len(y))
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    max_features=200_000)
                ptr, pte = [v.fit_transform(txt[tr])], [v.transform(txt[te])]
                if use_char:
                    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                         sublinear_tf=True, max_features=300_000)
                    ptr.append(vc.fit_transform(txt[tr]))
                    pte.append(vc.transform(txt[te]))
                ptr += [csr_matrix(R[tr]), csr_matrix(L[tr])]
                pte += [csr_matrix(R[te]), csr_matrix(L[te])]
                Xtr, Xte = hstack(ptr).tocsr(), hstack(pte).tocsr()
                est = make_est(kind, C).fit(Xtr, y[tr])
                oof[te] = score_of(est, Xte)
            ths = np.linspace(0.02, 0.98, 193)
            per_seed.append(max(f1_score(y, (oof >= t).astype(int)) for t in ths))
        a = np.array(per_seed)
        table[tag] = a
        print(f"  {tag} F1bin по сидам: {np.round(a,4)}  среднее={a.mean():.4f} ±{a.std():.4f}",
              flush=True)
    best = max(table, key=lambda k: table[k].mean())
    print(f"  -> устойчиво лучший: {best.strip()} ({table[best].mean():.4f})")
