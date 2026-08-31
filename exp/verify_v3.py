"""Проверка: ручной скорер submit3 воспроизводит sklearn-пайплайн (обе ветки)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "submit3")
sys.path.insert(0, ROOT + "exp")
from pathlib import Path  # noqa: E402

from folds import clean  # noqa: E402
from src.model import build_text, llm_features, load_models  # noqa: E402
from src.rules import extract  # noqa: E402

CFG = dict(word_ng=(1, 2), min_df=2, max_feat=200_000, C=10.0, name_rep=3)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]

models, name_rep, meta = load_models(Path(ROOT + "submit3/artifacts"))
ok = True
for cat in ["БАД", "Легковоспламеняющиеся"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y = sub["label"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = llm_features(sub["llm_score"].values)

    v = TfidfVectorizer(ngram_range=CFG["word_ng"], min_df=CFG["min_df"],
                        sublinear_tf=True, max_features=CFG["max_feat"])
    X = v.fit_transform(sub["txt"].values)
    A = hstack([X, csr_matrix(R)]).tocsr()
    B = hstack([A, csr_matrix(L)]).tocsr()
    def make_est():
        if cat == "Легковоспламеняющиеся":
            return LinearSVC(C=2.0, class_weight="balanced", max_iter=50000, random_state=0)
        return LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")

    def proba(est, X):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1]
        return 1.0 / (1.0 + np.exp(-est.decision_function(X)))

    ref_llm = proba(make_est().fit(B, y), B)
    ref_nl = proba(make_est().fit(A, y), A)

    m = models[cat]
    texts = [build_text(n, d, name_rep) for n, d in zip(sub["name"].fillna(""),
                                                        sub["description"].fillna(""))]
    mine_llm = np.array([m.score(t, R[i], L[i], True) for i, t in enumerate(texts)])
    mine_nl = np.array([m.score(t, R[i], None, False) for i, t in enumerate(texts)])

    for tag, ref, mine, thr in [("с LLM", ref_llm, mine_llm, m.threshold_llm),
                                ("без LLM", ref_nl, mine_nl, m.threshold_nollm)]:
        d = np.abs(ref - mine)
        same = ((ref >= thr) == (mine >= thr)).all()
        print(f"{cat:22s} [{tag:7s}] max|Δ|={d.max():.3e} предсказания идентичны: {same}")
        if d.max() > 1e-6 or not same:
            ok = False

print("\nРЕЗУЛЬТАТ:", "OK" if ok else "РАСХОЖДЕНИЕ!")
sys.exit(0 if ok else 1)
