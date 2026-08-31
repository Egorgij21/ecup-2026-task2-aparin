"""Проверка: ручной TfidfLogReg воспроизводит sklearn-пайплайн бит-в-бит."""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/counter/submit")
from src.vectorizer import TfidfLogReg, build_text  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = "/workspace/counter/"
ART = ROOT + "submit/artifacts/"
CFG = dict(word_ng=(1, 2), min_df=2, max_feat=200_000, C=10.0, name_rep=3)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["txt"] = [build_text(n, d) for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
meta = json.load(open(ART + "meta.json", encoding="utf-8"))

ok = True
for cat, m in meta["categories"].items():
    sub = df[df["category"] == cat].reset_index(drop=True)
    X, y = sub["txt"].values, sub["label"].values

    # эталон: sklearn, обученный ровно как в train_export.py
    vec = TfidfVectorizer(ngram_range=CFG["word_ng"], min_df=CFG["min_df"],
                          sublinear_tf=True, max_features=CFG["max_feat"])
    Xall = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced")
    clf.fit(Xall, y)
    ref = clf.predict_proba(Xall)[:, 1]

    # наша реализация из экспортированных артефактов
    mine_model = TfidfLogReg.load(ART + f"vocab_{m['slug']}.json", ART + f"model_{m['slug']}.npz",
                                  m["intercept"], m["threshold"])
    mine = mine_model.predict_proba(list(X))

    d = np.abs(ref - mine)
    same_pred = ((ref >= m["threshold"]) == (mine >= m["threshold"])).all()
    print(f"{cat}: n={len(X)} max|Δprob|={d.max():.3e} mean={d.mean():.3e} "
          f"одинаковые предсказания: {same_pred}")
    if d.max() > 1e-6 or not same_pred:
        ok = False
        bad = np.argsort(-d)[:3]
        for i in bad:
            print(f"   ! i={i} ref={ref[i]:.6f} mine={mine[i]:.6f} txt={X[i][:120]!r}")

print("\nРЕЗУЛЬТАТ:", "OK — реализации совпадают" if ok else "РАСХОЖДЕНИЕ!")
sys.exit(0 if ok else 1)
