"""Не модель, а ОЦЕНЩИК ПОРОГА. Сколько мы теряем на постановке порога?

Наблюдение из eval_blend_space: изотоническая калибровка бленда даёт PR-AUC 0.8558
против 0.8418 у сабмита и AUC 0.9899 против 0.9837 — то есть РАНЖИРУЕТ лучше, — но
по F1 проигрывает 0.8338 против 0.8615. Ранжирование и F1 расходятся. Значит потеря
не в модели, а в том, где ставится порог.

Причина понятна: 198 позитивов на флам. Кривая F1(порог) зазубрена, и argmax на
train-фолдах садится на случайный зубец, который на test-фолде не воспроизводится.

Проверяются оценщики порога (все фитятся ТОЛЬКО на train-фолдах, вложенность цела):
  argmax      — что стоит сейчас
  плато       — середина области, где F1 >= 0.99 от максимума
  сглаженный  — argmax после скользящего среднего по порогу
  бутстрап    — усреднение argmax по 200 бутстрап-репликам train-фолдов
  квантиль    — порог, дающий на test ту же долю позитивов, что в train

ВЕРХНЯЯ ГРАНИЦА (oracle) считается только как ориентир, сколько вообще можно
отыграть; в сабмит она не идёт и в сравнение как достижимое число не принимается.

Оценщиков несколько, поэтому смотрю не только на средний выигрыш, но и на то,
выигрывает ли правило В ОБЕИХ категориях и на КАЖДОМ фолде. Правило, выигрывающее
в среднем за счёт одного фолда, — это тот же зубец, только этажом выше.
"""
import glob
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
RNG = np.random.default_rng(0)


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def f1_curve(o, y):
    return np.array([f1_score(y, (o >= t).astype(int)) for t in THS])


def thr_argmax(o, y):
    return THS[int(f1_curve(o, y).argmax())]


def thr_plateau(o, y):
    f = f1_curve(o, y)
    m = f >= 0.99 * f.max()
    return float(np.median(THS[m]))


def thr_smooth(o, y, w=9):
    f = f1_curve(o, y)
    k = np.ones(w) / w
    s = np.convolve(f, k, mode="same")
    return THS[int(s.argmax())]


def thr_boot(o, y, n=200):
    ts = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        ts.append(thr_argmax(o[i], y[i]))
    return float(np.median(ts))


def thr_quantile(o, y):
    rate = y.mean()
    return float(np.quantile(o, 1 - rate))


RULES = {"argmax": thr_argmax, "плато": thr_plateau, "сглаженный": thr_smooth,
         "бутстрап": thr_boot, "квантиль": thr_quantile}

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt_in"] = (df["name_c"] + " ") * 3 + df["desc_c"]

summary = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)

    txt = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt_in"].values[tr]), csr_matrix(R[tr]),
                      csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt_in"].values[te]), csr_matrix(R[te]),
                      csr_matrix(L[te])]).tocsr()
        txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

    score = 0.5 * txt + 0.5 * lo if cat == "Легковоспламеняющиеся" else txt
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}  "
          f"(источник: {'бленд 0.5/0.5' if cat != 'БАД' else 'только текст'})", flush=True)

    print(f"  {'оценщик':12s} {'F1 общий':>9s}   пофолдово (F1 на своём фолде) / порог")
    for tag, fn in RULES.items():
        pred = np.zeros(len(y), dtype=int)
        per, thrs = [], []
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            t = fn(score[tr], y[tr])
            thrs.append(t)
            pred[te] = (score[te] >= t).astype(int)
            per.append(f1_score(y[te], pred[te]))
        tot = f1_score(y, pred)
        summary[(cat, tag)] = (tot, per)
        print(f"  {tag:12s} {tot:9.4f}   " +
              " ".join(f"{p:.3f}@{t:.2f}" for p, t in zip(per, thrs)), flush=True)

    orc = np.zeros(len(y), dtype=int)
    for k in range(5):
        te = np.where(fold == k)[0]
        orc[te] = (score[te] >= thr_argmax(score[te], y[te])).astype(int)
    print(f"  {'[oracle]':12s} {f1_score(y, orc):9.4f}   <- недостижимо, только ориентир")

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ по оценщику порога")
cats = ["Легковоспламеняющиеся", "БАД"]
base = np.mean([summary[(c, "argmax")][0] for c in cats])
for tag in RULES:
    vals = [summary[(c, tag)][0] for c in cats]
    wins = sum(1 for c in cats
               for a, b in [(summary[(c, tag)][1], summary[(c, "argmax")][1])]
               for x, z in zip(a, b) if x > z)
    print(f"  {tag:12s} {np.mean(vals):.4f}  ({np.mean(vals)-base:+.4f})   "
          f"флам {vals[0]:.4f}, БАД {vals[1]:.4f}   фолдов лучше argmax: {wins}/10")
print("\n  цель: 0.915 | текущий сабмит: вложенно 0.9006 -> LB 0.87820")
