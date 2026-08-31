"""КАНОНИЧЕСКИЙ замер: метрика соревнования для одного варианта LoRA на полном OOF.

Единственный скрипт, которым сравниваем кандидатов с чемпионом. Заводится потому,
что раньше одна и та же конфигурация давала 0.9002 в одном скрипте и 0.9006 в другом,
а «0.9006» вообще оказалось строкой-легендой, ни разу не пересчитанной.

Воспроизводит РОВНО конфигурацию сабмита:
  флам = 0.5 * (TF-IDF + правила + zero-shot) + 0.5 * LoRA, порог вложенный
  БАД  = тот же ансамбль в одиночку,                        порог вложенный
  метрика = среднее F1 двух категорий

Ориентир чемпиона (submit_blend_v6): флам 0.8615, БАД 0.9378, метрика 0.8996 -> LB 0.87820.

Защиты, без которых замер врёт молча:
  * требует ВСЕ пять фолдов с этим тегом — иначе часть строк осталась бы без LoRA;
  * проверяет, что id из разных фолдов не пересекаются (иначе склеены разные варианты);
  * проверяет полное покрытие датасета.
Каждая из них уже была бы нарушена: eval_full_oof.py собирал файлы по маске
lora_oof_fold*.parquet и после появления тегов начал бы смешивать варианты.

Запуск: python exp/eval_oof_tagged.py <тег>        например: _r32
        python exp/eval_oof_tagged.py ""           базовый вариант без тега
"""
import os
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

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
THS = np.linspace(0.005, 0.995, 199)
BASE = {"Легковоспламеняющиеся": 0.8615, "БАД": 0.9378}

paths = [ROOT + f"exp/lora_oof_fold{k}{TAG}.parquet" for k in range(5)]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "нет фолдов: " + ", ".join(os.path.basename(m) for m in missing) +
        "\nполный OOF считается только по всем пяти — иначе часть строк без LoRA")

parts = []
seen = set()
for p in paths:
    g = pd.read_parquet(p)[["id", "lora_score"]]
    dup = seen & set(g["id"])
    if dup:
        raise ValueError(f"{os.path.basename(p)}: {len(dup)} id уже были в другом фолде — "
                         "похоже, склеены разные варианты")
    seen |= set(g["id"])
    parts.append(g)
lora = pd.concat(parts, ignore_index=True)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df = df.merge(lora, on="id", how="left")
if df["lora_score"].isna().any():
    raise ValueError(f"{df['lora_score'].isna().sum()} строк остались без скора LoRA")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]
print(f"тег {TAG!r}: покрыто {len(df)} строк, все пять фолдов на месте")


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def nested(o, y, fold):
    pred = np.zeros(len(y), dtype=int)
    thrs = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        f = np.array([f1_score(y[tr], (o[tr] >= t).astype(int)) for t in THS])
        t = THS[int(f.argmax())]
        thrs.append(t)
        pred[te] = (o[te] >= t).astype(int)
    return f1_score(y, pred), thrs


res = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["nc"].values, cat)
    L = pf(sub["llm_score"].values)
    txt = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]),
                    csr_matrix(L[tr])]).tocsr()
        B = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]),
                    csr_matrix(L[te])]).tocsr()
        txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(A, y[tr]).predict_proba(B)[:, 1]

    # ровно то, что делает сабмит: на флам бленд, на БАД только ансамбль
    score = 0.5 * txt + 0.5 * lo if cat == "Легковоспламеняющиеся" else txt
    f, thrs = nested(score, y, fold)
    res[cat] = f
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}")
    print(f"  LoRA одна:  F1={nested(lo, y, fold)[0]:.4f}  AUC={roc_auc_score(y, lo):.4f}  "
          f"PR={average_precision_score(y, lo):.4f}")
    print(f"  текст один: F1={nested(txt, y, fold)[0]:.4f}  AUC={roc_auc_score(y, txt):.4f}")
    print(f"  КОНФИГУРАЦИЯ САБМИТА: F1={f:.4f}  "
          f"(чемпион {BASE[cat]:.4f}, разница {f - BASE[cat]:+.4f})")
    print(f"  пороги по фолдам: {' '.join(f'{t:.2f}' for t in thrs)}")

m = np.mean([res[c] for c in BASE])
print("\n" + "=" * 84)
print(f"МЕТРИКА СОРЕВНОВАНИЯ (вложенно, полный OOF): {m:.4f}")
print(f"  чемпион submit_blend_v6:                   {np.mean(list(BASE.values())):.4f}  -> LB 0.87820")
print(f"  разница:                                   {m - np.mean(list(BASE.values())):+.4f}")
print(f"  цель:                                      0.9150")
