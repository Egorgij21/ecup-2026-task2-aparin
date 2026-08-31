"""Из чего складывается скор: последовательное отключение компонентов.

Считаем вклад каждой части чемпиона по КАНОНИЧЕСКОЙ методике: полный OOF,
вложенный порог, конфигурация сабмита (флам = бленд, БАД = ансамбль в одиночку).
Ориентир: 0.8997.

Зачем. Мы полтора дня улучшали то, что вносит мало, и не трогали то, что вносит
много. Разложение показывает, где вообще есть запас, а где мы шлифуем шум.

Порядок наращивания выбран от самого дешёвого к самому дорогому по ресурсам:
TF-IDF (секунды CPU) -> правила (написаны руками один раз) -> zero-shot (час GPU)
-> LoRA (три часа GPU). Так видно не только вклад, но и его цену.
"""
import glob
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

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def nested(o, y, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        f = np.array([f1_score(y[tr], (o[tr] >= t).astype(int)) for t in THS])
        pred[te] = (o[te] >= THS[int(f.argmax())]).astype(int)
    return f1_score(y, pred)


def fit_oof(sub, y, fold, blocks, use_tfidf=True):
    o = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        ptr, pte = [], []
        if use_tfidf:
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                max_features=200_000)
            ptr.append(v.fit_transform(sub["ti"].values[tr]))
            pte.append(v.transform(sub["ti"].values[te]))
        for b in blocks:
            ptr.append(csr_matrix(b[tr]))
            pte.append(csr_matrix(b[te]))
        m = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")
        m.fit(hstack(ptr).tocsr(), y[tr])
        o[te] = m.predict_proba(hstack(pte).tocsr())[:, 1]
    return o


rows = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["nc"].values, cat)
    L = pf(sub["llm_score"].values)

    tfidf = fit_oof(sub, y, fold, [])
    tf_r = fit_oof(sub, y, fold, [R])
    tf_rl = fit_oof(sub, y, fold, [R, L])          # это и есть «ансамбль» сабмита
    rules_only = fit_oof(sub, y, fold, [R], use_tfidf=False)

    variants = {
        "1. только TF-IDF": tfidf,
        "2. только правила (без TF-IDF)": rules_only,
        "3. TF-IDF + правила": tf_r,
        "4. + zero-shot = ансамбль": tf_rl,
        "5. только LoRA": lo,
        "6. ансамбль + LoRA (сабмит)": 0.5 * tf_rl + 0.5 * lo,
    }
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)
    print(f"  {'вариант':34s} {'F1':>7s} {'AUC':>8s} {'PR':>8s}  прирост")
    prev = None
    for tag, o in variants.items():
        f = nested(o, y, fold)
        rows[(cat, tag)] = f
        d = "" if prev is None or tag.startswith(("2.", "5.")) else f"{f - prev:+.4f}"
        print(f"  {tag:34s} {f:7.4f} {roc_auc_score(y, o):8.4f} "
              f"{average_precision_score(y, o):8.4f}  {d}")
        if not tag.startswith(("2.", "5.")):
            prev = f

print("\n" + "=" * 92)
print("ВКЛАД В МЕТРИКУ СОРЕВНОВАНИЯ")
print("  (флам берётся из своей строки, БАД — из строки 4: в сабмите LoRA к БАД не применяется)")
FL, BD = "Легковоспламеняющиеся", "БАД"
bad_fixed = rows[(BD, "4. + zero-shot = ансамбль")]
for tag in ["1. только TF-IDF", "2. только правила (без TF-IDF)", "3. TF-IDF + правила",
            "4. + zero-shot = ансамбль", "5. только LoRA", "6. ансамбль + LoRA (сабмит)"]:
    m = (rows[(FL, tag)] + bad_fixed) / 2
    print(f"  {tag:34s} {m:.4f}   (флам {rows[(FL, tag)]:.4f}, БАД {bad_fixed:.4f})")
print("\n  чемпион на Qwen: 0.8997 -> LB 0.87820 | на gemma: LB 0.89017")
