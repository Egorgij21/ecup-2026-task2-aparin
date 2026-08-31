"""Единая сравнивалка вариантов LoRA на ОДНОМ фолде.

Заводится ради дисциплины: в хендоффе зафиксировано расхождение 0.9002 против 0.9006
для одной и той же конфигурации, померенной разными скриптами. Пока сравнение живёт
в пяти местах, любая разница в третьем знаке необъяснима. Здесь один способ на всех.

Читает все exp/lora_oof_fold<F>*.parquet и печатает для каждого:
  AUC, PR-AUC, лучший F1 — по адаптеру в одиночку;
  то же для бленда 0.5/0.5 с текстовым компонентом сабмита;
  корреляцию с базовым адаптером (высокая корреляция = вариант ничего не добавил).

ЧТО СЧИТАТЬ РАЗЛИЧИЕМ. На одном фолде 40 позитивов флам, и смена одного лишь сида
даёт 4 пункта F1 и 7 пунктов PR-AUC. Поэтому F1 здесь справочный, решение принимается
по PR-AUC, и меньше 7 пунктов на одном фолде — это не результат, а шум. Порог тут
подбирается на самом фолде (вложенности взяться неоткуда), так что абсолютные числа
завышены у ВСЕХ вариантов одинаково и годятся только для сравнения между собой.

Запуск: python exp/eval_fold0_compare.py [номер_фолда]
"""
import glob
import os
import re
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

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
THS = np.linspace(0.005, 0.995, 199)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


files = sorted(glob.glob(ROOT + f"exp/lora_oof_fold{FOLD}*.parquet"))
print(f"фолд {FOLD}, найдено вариантов: {len(files)}")
for f in files:
    print("  ", os.path.basename(f))

for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[(df["category"] == cat)].reset_index(drop=True)
    tr = sub[sub["fold"] != FOLD].reset_index(drop=True)
    te = sub[sub["fold"] == FOLD].reset_index(drop=True)
    Rtr = extract(tr["full"].values, tr["nc"].values, cat)
    Rte = extract(te["full"].values, te["nc"].values, cat)
    Ltr, Lte = pf(tr["llm_score"].values), pf(te["llm_score"].values)
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([v.fit_transform(tr["ti"].values), csr_matrix(Rtr), csr_matrix(Ltr)]).tocsr()
    Xte = hstack([v.transform(te["ti"].values), csr_matrix(Rte), csr_matrix(Lte)]).tocsr()
    txt = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(Xtr, tr["label"].values).predict_proba(Xte)[:, 1]
    tmap = dict(zip(te["id"].values, txt))
    y_txt = te["label"].values
    print(f"\n########## {cat}: фолд {FOLD}, n={len(te)} pos={y_txt.sum()}")
    print(f"  текст один: AUC={roc_auc_score(y_txt, txt):.4f} "
          f"PR={average_precision_score(y_txt, txt):.4f}")
    print(f"  {'вариант':30s} {'AUC':>7s} {'PR':>7s} {'F1':>7s} | "
          f"{'бленд AUC':>9s} {'бленд PR':>8s} {'бленд F1':>8s} {'корр':>6s}")

    base = None
    for f in files:
        g = pd.read_parquet(f)
        g = g[g["category"] == cat]
        if not len(g):
            continue
        y = g["label"].values
        lo = g["lora_score"].values
        t = np.array([tmap[i] for i in g["id"].values])
        bl = 0.5 * t + 0.5 * lo
        if base is None:
            base = lo
        tag = re.sub(rf"^lora_oof_fold{FOLD}", "", os.path.basename(f)[:-8]) or "(база)"
        f1l = max(f1_score(y, (lo >= th).astype(int)) for th in THS)
        f1b = max(f1_score(y, (bl >= th).astype(int)) for th in THS)
        print(f"  {tag:30s} {roc_auc_score(y, lo):7.4f} {average_precision_score(y, lo):7.4f} "
              f"{f1l:7.4f} | {roc_auc_score(y, bl):9.4f} {average_precision_score(y, bl):8.4f} "
              f"{f1b:8.4f} {np.corrcoef(lo, base)[0, 1]:6.3f}")

print("\n  Решение принимается по PR-AUC бленда. Разница меньше 7 пунктов PR")
print("  на одном фолде — шум от сида, а не эффект. Победитель идёт на полный OOF.")
