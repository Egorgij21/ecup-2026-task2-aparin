"""Ансамбль TF-IDF + правила + zero-shot LLM на ЧЕСТНОЙ (семейной) CV."""
import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, "/workspace/counter/exp")
from rule_features import extract  # noqa: E402

ROOT = "/workspace/counter/"
_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")
_NUM, _PUNCT = re.compile(r"\d+[.,]?\d*"), re.compile(r"[^\w\s]")
_UNITS = re.compile(r"\b(шт|уп|упак\w*|мл|л|гр|г|кг|мг|мкг|см|мм|м|табл\w*|капс\w*|"
                    r"порц\w*|блок\w*|коробк\w*|набор\w*|компл\w*|пач\w*|штук\w*)\b")


def clean(s):
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip().lower()


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
llm = pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]]
df = df.merge(llm, on="id", how="left")
print("LLM-скоры подтянуты:", df["llm_score"].notna().sum(), "из", len(df))

df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]
df["name_n"] = df["name_c"].map(
    lambda s: _WS.sub(" ", _UNITS.sub(" ", _PUNCT.sub(" ", _NUM.sub(" ", s)))).strip())


def families(sub, thr=0.75):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform(sub["name_n"].values)
    S = (X @ X.T).tocsr()
    S.data = (S.data >= thr).astype(np.int8)
    S.eliminate_zeros()
    return connected_components(S, directed=False)[1]


def rep(y, s, tag):
    ths = np.linspace(0.02, 0.98, 193)
    fb = max((f1_score(y, (s >= t).astype(int)), t) for t in ths)
    fm = max((f1_score(y, (s >= t).astype(int), average="macro"), t) for t in ths)
    print(f"    {tag:24s} AUC={roc_auc_score(y,s):.4f} PR={average_precision_score(y,s):.4f}"
          f"  F1bin={fb[0]:.4f}@{fb[1]:.2f}  F1mac={fm[0]:.4f}@{fm[1]:.2f}")
    return dict(bin=fb[0], tbin=fb[1], mac=fm[0], tmac=fm[1])


res = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y = sub["label"].values
    fam = families(sub)
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = sub["llm_score"].fillna(0.5).values.astype(np.float32)
    print(f"\n### {cat}: n={len(y)} pos={y.sum()} семей={len(set(fam))}")

    oof = {k: np.zeros(len(y)) for k in ["tfidf", "tfidf+rules", "tfidf+rules+llm"]}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(sub["txt"].values, y, fam):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr, Xte = v.fit_transform(sub["txt"].values[tr]), v.transform(sub["txt"].values[te])

        oof["tfidf"][te] = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

        A_tr = hstack([Xtr, csr_matrix(R[tr])]).tocsr()
        A_te = hstack([Xte, csr_matrix(R[te])]).tocsr()
        oof["tfidf+rules"][te] = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")\
            .fit(A_tr, y[tr]).predict_proba(A_te)[:, 1]

        # LLM как признак: сам скор + логит + индикаторы уверенности
        def llm_feats(idx):
            s = np.clip(L[idx], 1e-4, 1 - 1e-4)
            return np.c_[s, np.log(s / (1 - s)), (s > 0.5).astype(np.float32),
                         (s > 0.8).astype(np.float32), (s < 0.2).astype(np.float32)]
        B_tr = hstack([A_tr, csr_matrix(llm_feats(tr))]).tocsr()
        B_te = hstack([A_te, csr_matrix(llm_feats(te))]).tocsr()
        oof["tfidf+rules+llm"][te] = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")\
            .fit(B_tr, y[tr]).predict_proba(B_te)[:, 1]

    r = {}
    r["llm zero-shot"] = rep(y, L, "llm zero-shot")
    for k in ["tfidf", "tfidf+rules", "tfidf+rules+llm"]:
        r[k] = rep(y, oof[k], k)

    # ранговый бленд stacked-модели с сырым LLM-скором
    from scipy.stats import rankdata
    best = None
    ra, rb = rankdata(oof["tfidf+rules"]) / len(y), rankdata(L) / len(y)
    for w in np.linspace(0, 1, 41):
        bl = (1 - w) * ra + w * rb
        fb = max(f1_score(y, (bl >= t).astype(int)) for t in np.linspace(.5, .999, 120))
        if best is None or fb > best[0]:
            best = (fb, w)
    print(f"    ранговый бленд (tfidf+rules) x llm: F1bin={best[0]:.4f} @ w_llm={best[1]:.2f}")
    r["rank blend"] = dict(bin=best[0], mac=np.nan)
    res[cat] = r

print("\n" + "=" * 95)
print("МЕТРИКА СОРЕВНОВАНИЯ (среднее по категориям), честная семейная CV")
for k in ["llm zero-shot", "tfidf", "tfidf+rules", "tfidf+rules+llm", "rank blend"]:
    mb = np.mean([res[c][k]["bin"] for c in res])
    mm = np.mean([res[c][k].get("mac", np.nan) for c in res])
    print(f"  {k:20s} mean F1bin={mb:.4f}   mean F1mac={mm:.4f}")
print("\n  текущий сабмит на public LB = 0.73867 (семейная CV предсказывала 0.7830)")
