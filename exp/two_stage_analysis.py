"""Двухстадийный предикт: сильнее ли какой-то компонент В СОЛО на подмножестве, и
можно ли роутингом (первой стадией) побить равномерный бленд 0.5*ens+0.5*lora.

Компоненты (полный OOF):
  ens   — текстовый ансамбль (TF-IDF + правила + zero-shot LogReg)
  lora  — чемпионский LoRA (флам: среднее двух баз; БАД: _mmtp = v20)
Проверяем:
  * F1 каждого в соло vs бленд (при фикс. пороге);
  * confidence-routing: берём тот компонент, что дальше от 0.5 (увереннее);
  * disagreement: на строках, где ens и lora расходятся в вердикте, кто прав чаще;
  * ORACLE-потолок: если бы роутер идеально выбирал компонент — сколько это даёт
    (верхняя граница; если мал — роутить нечего).
Фикс. порог как в сабмите: флам 0.45, БАД 0.47.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THR = {"Легковоспламеняющиеся": 0.45, "БАД": 0.47}


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    return pd.concat([pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{tag}.parquet")[["id", "lora_score"]]
                      for k in range(5)], ignore_index=True).drop_duplicates("id")


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
for tag, nm in [("", "q"), ("_gemma", "g"), ("_mmtp", "b")]:
    df = df.merge(load_tag(tag).rename(columns={"lora_score": f"lo_{nm}"}), on="id", how="left")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]


def f1_at(y, s, thr):
    return f1_score(y, (s >= thr).astype(int))


for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    R = extract(sub["full"].values, sub["nc"].values, cat)
    L = pf(sub["llm_score"].values)
    ens = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
        Bm = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
        ens[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(A, y[tr]).predict_proba(Bm)[:, 1]
    lora = 0.5 * (sub["lo_q"].values + sub["lo_g"].values) if cat == "Легковоспламеняющиеся" \
        else sub["lo_b"].values
    thr = THR[cat]
    blend = 0.5 * ens + 0.5 * lora

    print(f"\n########## {cat}  n={len(y)} поз={y.sum()} порог {thr}")
    print(f"  ens соло         F1={f1_at(y, ens, thr):.4f}")
    print(f"  lora соло        F1={f1_at(y, lora, thr):.4f}")
    print(f"  бленд 0.5/0.5    F1={f1_at(y, blend, thr):.4f}")

    # confidence routing: компонент дальше от 0.5
    pick_lora = np.abs(lora - 0.5) >= np.abs(ens - 0.5)
    routed = np.where(pick_lora, lora, ens)
    print(f"  confidence-routing F1={f1_at(y, routed, thr):.4f} "
          f"(lora выбран в {pick_lora.mean()*100:.0f}% строк)")

    # где расходятся вердикты — кто прав
    ep, lp = (ens >= thr), (lora >= thr)
    dis = ep != lp
    if dis.sum():
        ens_right = ((ep == y) & dis).sum()
        lora_right = ((lp == y) & dis).sum()
        print(f"  расхождение вердиктов: {dis.sum()} строк — ens прав {ens_right}, lora прав {lora_right}")

    # oracle: идеальный роутер между ens и lora (верхняя граница)
    best = np.where(np.abs(y - lora) <= np.abs(y - ens), lora, ens)
    print(f"  ORACLE-роутинг   F1={f1_at(y, best, thr):.4f}  (потолок, если роутер идеален)")
