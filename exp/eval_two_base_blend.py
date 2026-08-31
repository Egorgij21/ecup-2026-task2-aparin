"""Бленд двух базовых моделей: спасает ли он от шейкапа и сколько стоит честно.

Повод. Локальный полный OOF и паблик разошлись: Qwen 0.8997 против геммы 0.8963
локально, но 0.87820 против 0.89017 на паблике. На паблике всего ~58 позитивов флам,
и вся разница в 0.012 — это около десяти строк. То есть паблик не различает варианты
слабее 0.012, а мы по нему уже приняли одно решение.

Если выбрать одну базу и ошибиться, приват перевернёт результат. Бленд двух даёт
среднее вместо подбрасывания монеты — это и есть страховка от шейкапа.

Корреляция OOF-скоров Qwen x gemma на полном OOF — **0.875** (в HANDOFF было
записано 0.95-0.97 по замерам с одного фолда, это оказалось неверно). При такой
корреляции ансамбль имеет право работать.

ГЛАВНОЕ ПРО ЧЕСТНОСТЬ: вес подбирается ВЛОЖЕННО — на фолдах != k, применяется к k.
Наивный выбор лучшего веса по тому же OOF уже давал нам завышение до +0.016.
Печатается и наивная оценка, разрыв между ними и есть цена подбора.
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

THS = np.linspace(0.005, 0.995, 199)
WS = np.round(np.arange(0.0, 1.01, 0.1), 2)
BAD = 0.9378          # БАД в сабмите считает ансамбль без LoRA, вариант не влияет


def load(tag):
    return pd.concat([pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{tag}.parquet")
                      [["id", "lora_score"]] for k in range(5)])


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df = df.merge(load("").rename(columns={"lora_score": "q"}), on="id")
df = df.merge(load("_gemma").rename(columns={"lora_score": "g"}), on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def best_thr(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    i = int(f.argmax())
    return THS[i], f[i]


cat = "Легковоспламеняющиеся"
sub = df[df["category"] == cat].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
q, g = sub["q"].values, sub["g"].values
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

print(f"корреляция Qwen x gemma на полном OOF: {np.corrcoef(q, g)[0, 1]:.3f}")
print(f"вердикты при 0.5 совпадают: {100 * ((q > .5) == (g > .5)).mean():.1f}%\n")

# 1) вес зафиксирован заранее — так решение и поедет в сабмит
print(f"  {'вес геммы':>10s} {'F1 флам':>9s} {'метрика':>9s}")
scores = {}
for w in WS:
    o = 0.5 * txt + 0.5 * ((1 - w) * q + w * g)
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= best_thr(o[tr], y[tr])[0]).astype(int)
    f = f1_score(y, pred)
    scores[w] = f
    mark = "  <- чемпион" if w == 0.0 else ("  <- поровну" if w == 0.5 else "")
    print(f"  {w:10.1f} {f:9.4f} {(f + BAD) / 2:9.4f}{mark}")

# 2) честная цена подбора веса
pred = np.zeros(len(y), dtype=int)
picked = []
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    best = None
    for w in WS:
        o = 0.5 * txt + 0.5 * ((1 - w) * q + w * g)
        t, f = best_thr(o[tr], y[tr])
        if best is None or f > best[1]:
            best = (w, f, t)
    w, _, t = best
    picked.append(w)
    o = 0.5 * txt + 0.5 * ((1 - w) * q + w * g)
    pred[te] = (o[te] >= t).astype(int)
honest = f1_score(y, pred)
naive_w = max(scores, key=lambda k: scores[k])

print(f"\n  фиксированный вес 0.5 (без подбора): {(scores[0.5] + BAD) / 2:.4f}")
print(f"  НАИВНО лучший вес {naive_w}:              {(scores[naive_w] + BAD) / 2:.4f}")
print(f"  ЧЕСТНО (вес подобран вложенно):      {(honest + BAD) / 2:.4f}")
print(f"  цена подбора веса: {(scores[naive_w] - honest) / 2:+.4f}   выбор по фолдам: {picked}")
print(f"\n  чемпион (только Qwen): {(scores[0.0] + BAD) / 2:.4f} -> LB 0.87820")
print(f"  только гемма:          {(scores[1.0] + BAD) / 2:.4f} -> LB 0.89017")
print("\n  Вывод про шейкап: если разница между базами внутри шума, бленд даёт среднее")
print("  вместо подбрасывания монеты. Смотреть надо на фиксированный вес 0.5, а не на")
print("  подобранный — подбор по 198 позитивам не переносится.")
