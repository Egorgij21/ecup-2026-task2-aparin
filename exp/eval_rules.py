"""Оценка brand-agnostic правил на ЧЕСТНОЙ (семейной) CV."""
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
from rule_features import extract, feature_names  # noqa: E402

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
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]
df["name_n"] = df["name_c"].map(lambda s: _WS.sub(" ", _UNITS.sub(" ", _PUNCT.sub(" ", _NUM.sub(" ", s)))).strip())


def families(sub, thr=0.75):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform(sub["name_n"].values)
    S = (X @ X.T).tocsr()
    S.data = (S.data >= thr).astype(np.int8)
    S.eliminate_zeros()
    return connected_components(S, directed=False)[1]


def score(y, oof, tag):
    ths = np.linspace(0.05, 0.95, 91)
    fb = max((f1_score(y, (oof >= t).astype(int)), t) for t in ths)
    fm = max((f1_score(y, (oof >= t).astype(int), average="macro"), t) for t in ths)
    print(f"    {tag:26s} AUC={roc_auc_score(y, oof):.4f} PR={average_precision_score(y, oof):.4f}"
          f"  F1bin={fb[0]:.4f}@{fb[1]:.2f}  F1mac={fm[0]:.4f}@{fm[1]:.2f}")
    return fb[0], fm[0], fb[1]


res = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y = sub["label"].values
    fam = families(sub)
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    print(f"\n### {cat}: n={len(y)} pos={y.sum()} семей={len(set(fam))}")

    variants = {}
    oofs = {k: np.zeros(len(y)) for k in ["tfidf", "rules", "tfidf+rules", "tfidf_lowC"]}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(sub["txt"].values, y, fam):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = v.fit_transform(sub["txt"].values[tr])
        Xte = v.transform(sub["txt"].values[te])

        c = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(Xtr, y[tr])
        oofs["tfidf"][te] = c.predict_proba(Xte)[:, 1]

        c2 = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, y[tr])
        oofs["tfidf_lowC"][te] = c2.predict_proba(Xte)[:, 1]

        cr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(R[tr], y[tr])
        oofs["rules"][te] = cr.predict_proba(R[te])[:, 1]

        Xtr2 = hstack([Xtr, csr_matrix(R[tr])]).tocsr()
        Xte2 = hstack([Xte, csr_matrix(R[te])]).tocsr()
        cb = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(Xtr2, y[tr])
        oofs["tfidf+rules"][te] = cb.predict_proba(Xte2)[:, 1]

    for k in ["tfidf", "tfidf_lowC", "rules", "tfidf+rules"]:
        variants[k] = score(y, oofs[k], k)
    # бленд tfidf + rules по вероятностям
    best = None
    for w in np.linspace(0, 1, 21):
        bl = (1 - w) * oofs["tfidf"] + w * oofs["rules"]
        fb = max(f1_score(y, (bl >= t).astype(int)) for t in np.linspace(.05, .95, 91))
        if best is None or fb > best[0]:
            best = (fb, w)
    print(f"    бленд tfidf+rules: лучший F1bin={best[0]:.4f} при w_rules={best[1]:.2f}")
    variants["blend"] = (best[0], None, None)
    res[cat] = variants

    if cat == "Легковоспламеняющиеся":
        cr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(R, y)
        names = feature_names(cat)
        allnames = [f"{n}|full" for n in names] + [f"{n}|name" for n in names]
        order = np.argsort(cr.coef_[0])
        print("    правила ЗА позитив:", ", ".join(f"{allnames[i]}({cr.coef_[0][i]:+.2f})" for i in order[-8:][::-1]))
        print("    правила ЗА негатив:", ", ".join(f"{allnames[i]}({cr.coef_[0][i]:+.2f})" for i in order[:8]))

print("\n" + "=" * 95)
for k in ["tfidf", "tfidf_lowC", "rules", "tfidf+rules", "blend"]:
    mb = np.mean([res[c][k][0] for c in res])
    print(f"  {k:14s} mean F1bin = {mb:.4f}")
print("  (public LB текущего сабмита = 0.73867)")
