"""Единая ЧЕСТНАЯ нарезка фолдов — по семьям товаров, а не по точному тексту.

Почему это критично: сабмит с CV 0.886 (группировка по точному совпадению текста)
дал на public LB 0.7387. Семейная группировка даёт 0.7830 — почти попадание.
Разница в том, что «Розжиг набор 22/35/44 коробков» или «Хлопушка ТР 102/106/108» —
это один товар в вариантах: текст разный, но по фолдам их разносить нельзя.
"""
import re

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")
_NUM, _PUNCT = re.compile(r"\d+[.,]?\d*"), re.compile(r"[^\w\s]")
_UNITS = re.compile(r"\b(шт|уп|упак\w*|мл|л|гр|г|кг|мг|мкг|см|мм|м|табл\w*|капс\w*|"
                    r"порц\w*|блок\w*|коробк\w*|набор\w*|компл\w*|пач\w*|штук\w*)\b")


def clean(s) -> str:
    s = str(s) if s is not None else ""
    if s == "nan":
        s = ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip().lower()


def norm_name(s: str) -> str:
    """Убираем числа, пунктуацию и единицы фасовки — остаётся «скелет» товара."""
    return _WS.sub(" ", _UNITS.sub(" ", _PUNCT.sub(" ", _NUM.sub(" ", s)))).strip()


def family_labels(names, thr: float = 0.75) -> np.ndarray:
    """Связные компоненты по косинусной близости char-нграмм нормализованного названия."""
    normed = [norm_name(clean(n)) for n in names]
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform(normed)
    S = (X @ X.T).tocsr()
    S.data = (S.data >= thr).astype(np.int8)
    S.eliminate_zeros()
    return connected_components(S, directed=False)[1]


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Номер фолда для каждой строки. Стратификация по label ВНУТРИ каждой категории,
    группировка по семьям — чтобы варианты одного товара не пересекали границу фолда."""
    fold = np.full(len(df), -1, dtype=int)
    for cat in df["category"].unique():
        m = (df["category"] == cat).values
        sub = df[m]
        fam = family_labels(sub["name"].fillna("").values)
        y = sub["label"].values
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        idx = np.where(m)[0]
        for k, (_, te) in enumerate(splitter.split(np.zeros(len(y)), y, fam)):
            fold[idx[te]] = k
    assert (fold >= 0).all()
    return fold


def fold_report(df: pd.DataFrame, fold: np.ndarray) -> pd.DataFrame:
    t = df.assign(fold=fold)
    return t.pivot_table(index="fold", columns="category", values="label",
                         aggfunc=["size", "sum"])
