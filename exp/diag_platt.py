"""Чинит ли калибровка Платта перенос порога у SVC?

Ровно та поломка, что убила v3: порог снят с модели на 80% данных, применён к модели
на 100%. У SVC это стоило −0.045 F1, у LogReg −0.021.

Платт монотонен (ранжирование SVC сохраняется), но растягивает шкалу в вероятности.
Калибратор обучается ТОЛЬКО на out-of-fold решениях, иначе утечка.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

CAT = "Легковоспламеняющиеся"
THS = np.linspace(0.02, 0.98, 193)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]
df["fold"] = make_folds(df)

sub = df[df["category"] == CAT].reset_index(drop=True)
fold, y = sub["fold"].values, sub["label"].values
R = extract(sub["full"].values, sub["name_c"].values, CAT)
s = np.clip(sub["llm_score"].values, 1e-4, 1 - 1e-4)
L = np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float), (s > .8).astype(float), (s < .2).astype(float)]

HOLD = fold == 4
rest, hold = np.where(~HOLD)[0], np.where(HOLD)[0]


def features(train_idx, apply_idx):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([v.fit_transform(sub["txt"].values[train_idx]),
                  csr_matrix(R[train_idx]), csr_matrix(L[train_idx])]).tocsr()
    Xap = hstack([v.transform(sub["txt"].values[apply_idx]),
                  csr_matrix(R[apply_idx]), csr_matrix(L[apply_idx])]).tocsr()
    return Xtr, Xap


def svc_platt(train_idx, apply_idx):
    """SVC + калибратор Платта, обученный на OOF-решениях внутри train_idx."""
    Xtr, Xap = features(train_idx, apply_idx)
    ytr = y[train_idx]
    # OOF-решения внутри обучающей части — на них учим калибратор
    oof_dec = np.zeros(len(ytr))
    for tr2, te2 in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
        m2 = LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0)
        m2.fit(Xtr[tr2], ytr[tr2])
        oof_dec[te2] = m2.decision_function(Xtr[te2])
    platt = LogisticRegression(max_iter=2000).fit(oof_dec.reshape(-1, 1), ytr)
    m = LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0).fit(Xtr, ytr)
    return platt.predict_proba(m.decision_function(Xap).reshape(-1, 1))[:, 1]


def svc_raw(train_idx, apply_idx):
    Xtr, Xap = features(train_idx, apply_idx)
    m = LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0).fit(Xtr, y[train_idx])
    return 1 / (1 + np.exp(-m.decision_function(Xap)))


def logreg(train_idx, apply_idx):
    Xtr, Xap = features(train_idx, apply_idx)
    m = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(Xtr, y[train_idx])
    return m.predict_proba(Xap)[:, 1]


rng = np.random.RandomState(0)
idx75 = rng.choice(rest, int(len(rest) * 0.75), replace=False)

print(f"отложено {len(hold)} строк ({y[hold].sum()} позитивов)\n")
print(f"{'модель':12s} {'AUC':>7s} {'плато':>7s} {'F1@свой':>9s} {'F1@чужой':>9s} {'ПОТЕРЯ':>8s}")
print("-" * 60)
for name, fn in [("logreg", logreg), ("svc_raw", svc_raw), ("svc_platt", svc_platt)]:
    sc75 = fn(idx75, hold)
    sc100 = fn(rest, hold)
    f75 = np.array([f1_score(y[hold], (sc75 >= t).astype(int)) for t in THS])
    f100 = np.array([f1_score(y[hold], (sc100 >= t).astype(int)) for t in THS])
    t75 = THS[int(f75.argmax())]
    own, foreign = f100.max(), f1_score(y[hold], (sc100 >= t75).astype(int))
    good = THS[f100 >= f100.max() * 0.98]
    print(f"{name:12s} {roc_auc_score(y[hold], sc100):7.4f} {good.max()-good.min():7.3f} "
          f"{own:9.4f} {foreign:9.4f} {foreign-own:+8.4f}")
    print(f"{'':12s} скоры p50={np.median(sc100):.3f} p90={np.percentile(sc100,90):.3f} "
          f"p99={np.percentile(sc100,99):.3f}")
