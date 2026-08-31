"""Честная валидация: группировка по СЕМЬЯМ товаров (near-duplicates), а не по точному тексту.

Гипотеза: CV 0.886 завышена, потому что почти-дубли (одна и та же позиция в разных
фасовках/цветах/артикулах) расползались по фолдам. Проверяем, воспроизводит ли
семейная группировка публичный скор 0.7387.
"""
import re
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

ROOT = "/workspace/counter/"
_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def clean(s):
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip().lower()


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]

# --- нормализованное имя: убираем числа, единицы, фасовки, артикулы -------
_NUM = re.compile(r"\d+[.,]?\d*")
_UNITS = re.compile(r"\b(шт|уп|упак\w*|мл|л|гр|г|кг|мг|мкг|см|мм|м|табл\w*|капс\w*|"
                    r"порц\w*|блок\w*|коробк\w*|набор\w*|компл\w*|пач\w*|штук\w*)\b")
_PUNCT = re.compile(r"[^\w\s]")


def norm_name(s):
    s = _NUM.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _UNITS.sub(" ", s)
    return _WS.sub(" ", s).strip()


df["name_n"] = df["name_c"].map(norm_name)


def family_groups(sub, thr=0.75):
    """Связные компоненты по косинусной близости char-нграмм нормализованного имени."""
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform(sub["name_n"].values)
    S = (X @ X.T).tocsr()
    S.data = (S.data >= thr).astype(np.int8)
    S.eliminate_zeros()
    n, lab = connected_components(S, directed=False)
    return lab, n


def run_cv(sub, groups, seed=42):
    X, y = sub["txt"].values, sub["label"].values
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        c = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")
        c.fit(v.fit_transform(X[tr]), y[tr])
        oof[te] = c.predict_proba(v.transform(X[te]))[:, 1]
    ths = np.linspace(0.05, 0.95, 91)
    fb = max((f1_score(y, (oof >= t).astype(int)), t) for t in ths)
    fm = max((f1_score(y, (oof >= t).astype(int), average="macro"), t) for t in ths)
    # метрика при ФИКСИРОВАННОМ пороге из сабмита
    return fb, fm, oof


SUBMIT_THR = {"БАД": 0.38, "Легковоспламеняющиеся": 0.42}
summary = {}
for cat in ["БАД", "Легковоспламеняющиеся"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y = sub["label"].values
    exact = sub.groupby(sub["txt"]).ngroup().values
    fam, nfam = family_groups(sub)
    print(f"\n### {cat}: n={len(sub)} pos={y.sum()}")
    print(f"  групп: точный текст={len(set(exact))}  семьи={nfam}")
    sizes = pd.Series(fam).value_counts()
    print(f"  крупнейшие семьи: {sizes.head(5).tolist()}  семей размера 1: {(sizes==1).sum()}")
    pos_fam = pd.Series(fam)[y == 1].nunique()
    print(f"  позитивы лежат в {pos_fam} семьях (позитивов {y.sum()})")

    row = {}
    for tag, g in [("точный текст", exact), ("СЕМЬИ", fam)]:
        fb, fm, oof = run_cv(sub, g)
        t = SUBMIT_THR[cat]
        fb_fix = f1_score(y, (oof >= t).astype(int))
        fm_fix = f1_score(y, (oof >= t).astype(int), average="macro")
        print(f"  [{tag:12s}] best F1bin={fb[0]:.4f}@{fb[1]:.2f}  best F1mac={fm[0]:.4f}@{fm[1]:.2f}"
              f"   | при пороге сабмита {t}: bin={fb_fix:.4f} mac={fm_fix:.4f}")
        row[tag] = (fb_fix, fm_fix, fb[0], fm[0])
    summary[cat] = row

print("\n" + "=" * 95)
print("ИТОГ (метрика соревнования = среднее по двум категориям)")
for tag in ["точный текст", "СЕМЬИ"]:
    mb = np.mean([summary[c][tag][0] for c in summary])
    mm = np.mean([summary[c][tag][1] for c in summary])
    mbb = np.mean([summary[c][tag][2] for c in summary])
    mmb = np.mean([summary[c][tag][3] for c in summary])
    print(f"  группировка «{tag:12s}»: при порогах сабмита bin={mb:.4f} mac={mm:.4f}"
          f"   | при лучших порогах bin={mbb:.4f} mac={mmb:.4f}")
print("\n  ФАКТ на public LB: 0.73867")
