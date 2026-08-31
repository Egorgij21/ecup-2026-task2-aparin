"""Гипотеза: у SVC шкала decision_function зависит от размера обучающей выборки.

Порог снимается с OOF-моделей (обучены на 80%), а в сабмит идёт модель на 100%.
Если шкала полной модели смещена, фиксированный порог даёт другую долю позитивов —
именно так CV растёт, а LB падает. У LogReg вероятности калиброваны и такого нет.

Проверяем на ОТЛОЖЕННЫХ данных: учим на 60% и на 80% одной и той же выборки,
сравниваем скоры на одном и том же отложенном куске.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

CAT = "Легковоспламеняющиеся"
df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]
df["fold"] = make_folds(df)

sub = df[df["category"] == CAT].reset_index(drop=True)
fold = sub["fold"].values
y = sub["label"].values
R = extract(sub["full"].values, sub["name_c"].values, CAT)
s = np.clip(sub["llm_score"].values, 1e-4, 1 - 1e-4)
L = np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float), (s > .8).astype(float), (s < .2).astype(float)]

HOLD = fold == 4                      # отложенный кусок, его не видит никто
rest = np.where(~HOLD)[0]
hold = np.where(HOLD)[0]
print(f"отложено: {len(hold)} строк, позитивов {y[hold].sum()}")
print(f"остальное: {len(rest)} строк, позитивов {y[rest].sum()}\n")


def fit_score(train_idx, kind):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([v.fit_transform(sub["txt"].values[train_idx]),
                  csr_matrix(R[train_idx]), csr_matrix(L[train_idx])]).tocsr()
    Xho = hstack([v.transform(sub["txt"].values[hold]),
                  csr_matrix(R[hold]), csr_matrix(L[hold])]).tocsr()
    if kind == "svc":
        m = LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0).fit(Xtr, y[train_idx])
        return 1 / (1 + np.exp(-m.decision_function(Xho)))
    m = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(Xtr, y[train_idx])
    return m.predict_proba(Xho)[:, 1]


rng = np.random.RandomState(0)
FRACS = [0.5, 0.75, 1.0]
THS = np.linspace(0.02, 0.98, 193)

for kind in ["logreg", "svc"]:
    print("=" * 88)
    print(f"### {kind}")
    scores = {}
    for f in FRACS:
        idx = rest if f == 1.0 else rng.choice(rest, int(len(rest) * f), replace=False)
        scores[f] = fit_score(idx, kind)
        sc = scores[f]
        print(f"  обучено на {int(f*100):3d}% ({len(idx):4d} строк): "
              f"скоры p50={np.median(sc):.4f} p90={np.percentile(sc,90):.4f} "
              f"p99={np.percentile(sc,99):.4f} std={sc.std():.4f}")

    # порог, оптимальный для модели на 75%, применяем к модели на 100%
    f75 = np.array([f1_score(y[hold], (scores[0.75] >= t).astype(int)) for t in THS])
    t75 = THS[int(f75.argmax())]
    f100 = np.array([f1_score(y[hold], (scores[1.0] >= t).astype(int)) for t in THS])
    t100 = THS[int(f100.argmax())]
    print(f"\n  оптимальный порог для модели на 75%:  {t75:.3f} (F1={f75.max():.4f})")
    print(f"  оптимальный порог для модели на 100%: {t100:.3f} (F1={f100.max():.4f})")
    print(f"  СДВИГ ОПТИМУМА: {t100-t75:+.3f}")
    applied = f1_score(y[hold], (scores[1.0] >= t75).astype(int))
    print(f"  F1 модели на 100% при пороге от модели на 75%: {applied:.4f} "
          f"(потеря {applied-f100.max():+.4f})")
    n75 = (scores[1.0] >= t75).sum()
    n100 = (scores[1.0] >= t100).sum()
    print(f"  предсказано позитивов: при чужом пороге {n75}, при своём {n100} "
          f"(истинно {y[hold].sum()})")
