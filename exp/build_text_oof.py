"""Кэш OOF-скоров текстового ансамбля + разметка ошибок. Нужен и проверке картинок,
и визуализатору, поэтому считается один раз и кладётся в exp/text_oof.parquet."""
import sys
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
R = "/workspace/counter/"
sys.path.insert(0, R + "exp")
from folds import clean, make_folds
from rule_features import extract
THS = np.linspace(.005, .995, 199)
df = pd.read_csv(R + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(R + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean); df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]; df["ti"] = (df["nc"] + " ") * 3 + df["dc"]
def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]
out = []
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    Rr = extract(sub["full"].values, sub["nc"].values, cat); L = pf(sub["llm_score"].values)
    sc = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(Rr[tr]), csr_matrix(L[tr])]).tocsr()
        B = hstack([v.transform(sub["ti"].values[te]), csr_matrix(Rr[te]), csr_matrix(L[te])]).tocsr()
        sc[te] = LogisticRegression(max_iter=4000, C=10., class_weight="balanced").fit(A, y[tr]).predict_proba(B)[:, 1]
    pred = np.zeros(len(y), int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        f = np.array([f1_score(y[tr], (sc[tr] >= t).astype(int)) for t in THS])
        pred[te] = (sc[te] >= THS[int(f.argmax())]).astype(int)
    sub["text_score"] = sc; sub["text_pred"] = pred
    sub["err"] = np.where((pred == 1) & (y == 0), "ложное", np.where((pred == 0) & (y == 1), "пропуск", "верно"))
    print(f"{cat}: F1={f1_score(y, pred):.4f} | ложных {int((sub.err=='ложное').sum())} "
          f"| пропусков {int((sub.err=='пропуск').sum())}", flush=True)
    out.append(sub[["id", "category", "label", "text_score", "text_pred", "err", "fold"]])
pd.concat(out).to_parquet(R + "exp/text_oof.parquet")
print("-> exp/text_oof.parquet")
