"""Вклад атомарных вопросов — сразу с ВЛОЖЕННЫМ подбором порога.

Базой берём конфиг v2 (LogReg везде), потому что SVC откачен: он выиграл на наивной
CV и проиграл на LB. Никаких выводов из наивных чисел больше не делаем.
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

SEEDS = [42, 7, 2024]
THS = np.linspace(0.02, 0.98, 193)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
q = pd.read_parquet(ROOT + "exp/llm_questions.parquet")
qcols = [c for c in q.columns if c.startswith("q_")]
df = df.merge(q[["id"] + qcols], on="id", how="left")
print(f"вопросов подтянуто: {qcols}")

df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def prob_feats(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def pick_argmax(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    return THS[int(f.argmax())]


VARIANTS = ["tfidf+rules", "+llm (=v2)", "+вопросы", "+llm+вопросы"]

summary = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y = sub["label"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = prob_feats(sub["llm_score"].values)
    cols = [c for c in qcols if sub[c].notna().any()]
    Q = np.hstack([prob_feats(sub[c].fillna(0.5).values) for c in cols])
    print(f"\n########## {cat}  n={len(y)} pos={y.sum()}  вопросов={len(cols)} "
          f"(признаков {Q.shape[1]})", flush=True)

    for tag in VARIANTS:
        nested, naive, aucs, prs = [], [], [], []
        for seed in SEEDS:
            fold = make_folds(df, seed=seed)[(df["category"] == cat).values]
            o = np.zeros(len(y))
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    max_features=200_000)
                ptr = [v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr])]
                pte = [v.transform(sub["txt"].values[te]), csr_matrix(R[te])]
                if "llm" in tag:
                    ptr.append(csr_matrix(L[tr]))
                    pte.append(csr_matrix(L[te]))
                if "вопросы" in tag:
                    ptr.append(csr_matrix(Q[tr]))
                    pte.append(csr_matrix(Q[te]))
                m = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")
                m.fit(hstack(ptr).tocsr(), y[tr])
                o[te] = m.predict_proba(hstack(pte).tocsr())[:, 1]
            aucs.append(roc_auc_score(y, o))
            prs.append(average_precision_score(y, o))
            naive.append(f1_score(y, (o >= pick_argmax(o, y)).astype(int)))
            pred = np.zeros(len(y), dtype=int)
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                pred[te] = (o[te] >= pick_argmax(o[tr], y[tr])).astype(int)
            nested.append(f1_score(y, pred))
        a, b = np.array(naive), np.array(nested)
        print(f"  {tag:14s} AUC={np.mean(aucs):.4f} PR={np.mean(prs):.4f}  "
              f"наивно={a.mean():.4f}  ВЛОЖЕННО={b.mean():.4f}±{b.std():.4f}", flush=True)
        summary[(cat, tag)] = b.mean()

print("\n" + "=" * 92)
print("МЕТРИКА СОРЕВНОВАНИЯ — ВЛОЖЕННАЯ оценка (среднее по категориям)")
for tag in VARIANTS:
    m = np.mean([summary[(c, tag)] for c in ["Легковоспламеняющиеся", "БАД"]])
    print(f"  {tag:14s} {m:.4f}")
print("\n  напоминание: v2 (=«+llm») вложенно 0.8001, факт на LB 0.8150")
