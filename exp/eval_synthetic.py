"""Даёт ли синтетика хоть что-то? Быстрая проверка на дешёвой модели.

Дообучение LoRA стоит ~5 часов GPU. Прежде чем платить, смотрим на логреге: если
323 синтетические карточки не помогают даже TF-IDF, ставка на них сомнительна.
Обратное неверно — помощь логрегу не гарантирует помощи LoRA, — поэтому это
отсеивающая проверка, а не подтверждающая.

Сравнение честное: блок llm-признаков выключен У ОБЕИХ веток, потому что для
синтетики zero-shot не считался. Иначе базовая ветка получила бы фору.

Утечки нет по построению: синтетика порождена только из опубликованных правил
площадки, ни одна строка данных в генерацию не подавалась (см. gen_synthetic.py).
Синтетика идёт ТОЛЬКО в train-часть каждого фолда, в валидацию — никогда.
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

THS = np.linspace(0.005, 0.995, 199)
CAT = "Легковоспламеняющиеся"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["fold"] = make_folds(df)
d = df[df["category"] == CAT].reset_index(drop=True)
syn = pd.read_parquet(ROOT + "exp/synth_flam.parquet")
print(f"реальных {len(d)} (поз {d['label'].sum()}), синтетических {len(syn)} "
      f"(поз {int(syn['label'].sum())}), концептов {syn['concept'].nunique()}")


def prep(frame):
    nc = frame["name"].fillna("").map(clean)
    dc = frame["description"].fillna("").map(clean)
    return (nc + " ") * 3 + dc, nc + " " + dc, nc


ti, full, nc = prep(d)
sti, sfull, snc = prep(syn)
y, fold = d["label"].values, d["fold"].values
sy = syn["label"].values
R = extract(full.values, nc.values, CAT)
SR = extract(sfull.values, snc.values, CAT)


def best_thr(o, yy):
    f = np.array([f1_score(yy, (o >= t).astype(int)) for t in THS])
    return THS[int(f.argmax())]


def run(use_syn, weight=1.0):
    o = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        txt_tr = list(ti.values[tr]) + (list(sti.values) if use_syn else [])
        ytr = np.concatenate([y[tr], sy]) if use_syn else y[tr]
        Xtr = v.fit_transform(txt_tr)
        Rtr = np.vstack([R[tr], SR]) if use_syn else R[tr]
        Xtr = hstack([Xtr, csr_matrix(Rtr)]).tocsr()
        Xte = hstack([v.transform(ti.values[te]), csr_matrix(R[te])]).tocsr()
        sw = np.concatenate([np.ones(len(tr)), np.full(len(sy), weight)]) if use_syn else None
        o[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, ytr, sample_weight=sw).predict_proba(Xte)[:, 1]
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= best_thr(o[tr], y[tr])).astype(int)
    return f1_score(y, pred), roc_auc_score(y, o), average_precision_score(y, o), o


print(f"\n  {'вариант':28s} {'F1':>7s} {'AUC':>8s} {'PR':>8s}")
f0, a0, p0, _ = run(False)
print(f"  {'без синтетики':28s} {f0:7.4f} {a0:8.4f} {p0:8.4f}")
for w in [0.25, 0.5, 1.0, 2.0]:
    f, a, p, _ = run(True, w)
    print(f"  {f'+синтетика, вес {w}':28s} {f:7.4f} {a:8.4f} {p:8.4f}   "
          f"({f-f0:+.4f} F1, {p-p0:+.4f} PR)", flush=True)

print("\n  Решение: если PR-AUC не растёт ни при одном весе, тратить 5 часов GPU")
print("  на дообучение с синтетикой не стоит — сигнала нет уже на простой модели.")
