"""F1 при ФИКСИРОВАННОМ пороге на одном фолде — так, как работает сабмит.

Заменяет сравнение по PR-AUC и по вложенному порогу, оба раза нас подведшие:
  * lr 2e-4 выигрывал 4 фолда из 5 по PR и проигрывал 4 из 5 по F1;
  * геометрическое среднее выигрывало по вложенному порогу и проиграло на паблике,
    потому что вложенная процедура штрафует за нестабильность выбора порога,
    которой в сабмите нет.

Считает конфигурацию сабмита: 0.5 * текстовый ансамбль + 0.5 * адаптер,
порог перебирается по сетке, но ОДИН на все строки фолда.

Запуск: python exp/eval_fold_fixed.py <фолд> <тег1> [тег2 ...]
Тег "" означает базовый вариант без суффикса.
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

FOLD = int(sys.argv[1])
TAGS = sys.argv[2:] or [""]
THS = np.arange(0.20, 0.81, 0.01)


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]

for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat]
    tr, te = sub[sub["fold"] != FOLD], sub[sub["fold"] == FOLD].reset_index(drop=True)
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(tr["ti"].values),
                csr_matrix(extract(tr["full"].values, tr["nc"].values, cat)),
                csr_matrix(pf(tr["llm_score"].values))]).tocsr()
    B = hstack([v.transform(te["ti"].values),
                csr_matrix(extract(te["full"].values, te["nc"].values, cat)),
                csr_matrix(pf(te["llm_score"].values))]).tocsr()
    txt = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, tr["label"].values).predict_proba(B)[:, 1]
    tmap = dict(zip(te["id"].values, txt))
    print(f"\n########## {cat}: фолд {FOLD}, n={len(te)}, позитивов {int(te['label'].sum())}")
    print(f"  {'вариант':22s} {'AUC':>7s} {'PR':>7s} {'F1 адаптер':>11s} "
          f"{'F1 БЛЕНД':>9s} {'порог':>6s}")
    base = None
    for tag in TAGS:
        try:
            g = pd.read_parquet(ROOT + f"exp/lora_oof_fold{FOLD}{tag}.parquet")
        except FileNotFoundError:
            print(f"  {tag or '(база)':22s} файла нет")
            continue
        g = g[g["category"] == cat]
        if not len(g):
            continue
        y = g["label"].values
        lo = g["lora_score"].values
        t = np.array([tmap[i] for i in g["id"].values])
        bl = 0.5 * t + 0.5 * lo
        f_lo = max(f1_score(y, (lo >= x).astype(int)) for x in THS)
        best = max((f1_score(y, (bl >= x).astype(int)), x) for x in THS)
        if base is None:
            base = best[0]
        d = "" if base is None else f"  ({best[0] - base:+.4f})"
        print(f"  {tag or '(база)':22s} {roc_auc_score(y, lo):7.4f} "
              f"{average_precision_score(y, lo):7.4f} {f_lo:11.4f} "
              f"{best[0]:9.4f} {best[1]:6.2f}{d}")
        # машинно-читаемо: разбор глазами и awk-ом уже подводил
        print(f"MACHINE\t{cat}\t{tag or 'base'}\t{best[0]:.6f}\t{best[1]:.2f}")
