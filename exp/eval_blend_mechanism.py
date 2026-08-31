"""За счёт ЧЕГО работает бленд: информация или устойчивость порога?

Наблюдение, ради которого написано. На фолде 0 PR-AUC бленда максимален при весе
LoRA = 1.0 у ВСЕХ пяти адаптеров, то есть текст ухудшает ранжирование. А на полном
OOF по PR они вровень (0.8418 у бленда против 0.8437 у LoRA одной), но по вложенному
F1 бленд впереди на 4.4 пункта (0.8615 против 0.8179).

Ранжирование одинаковое, а F1 разный — значит выигрыш берётся не из информации,
а из ПОСТАНОВКИ ПОРОГА. LoRA выдаёт почти двоичные скоры (масса у 0 и 1), порог
на такой шкале хрупок и не переносится с train-фолдов на тестовый; примесь текста
шкалу размазывает.

Если гипотеза верна, тот же выигрыш достаётся калибровкой LoRA БЕЗ текста — и тогда
рычаг «усилить адаптер» перестаёт гаситься блендом, а это ровно то, что мешает
ночным улучшениям доезжать до метрики.

Проверка: доля скоров у краёв шкалы + вложенный F1 разных способов принять решение.
Всё вложенно, полный OOF, порог всегда с фолдов != k.
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
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


def nested(o, y, fold, rule="argmax"):
    pred = np.zeros(len(y), dtype=int)
    thrs = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        if rule == "quantile":
            t = float(np.quantile(o[tr], 1 - y[tr].mean()))
        else:
            f = np.array([f1_score(y[tr], (o[tr] >= x).astype(int)) for x in THS])
            t = THS[int(f.argmax())]
        thrs.append(t)
        pred[te] = (o[te] >= t).astype(int)
    return f1_score(y, pred), thrs


def calibrate(o, y, fold):
    """Изотоническая калибровка, обученная на train-фолдах."""
    c = np.zeros(len(o))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        c[te] = IsotonicRegression(out_of_bounds="clip").fit(o[tr], y[tr]).predict(o[te])
    return c


summary = {}
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

    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}")
    edge_lo = ((lo < 0.02) | (lo > 0.98)).mean()
    edge_bl = (((0.5 * txt + 0.5 * lo) < 0.02) | ((0.5 * txt + 0.5 * lo) > 0.98)).mean()
    print(f"  доля скоров у краёв шкалы (<0.02 или >0.98): LoRA {edge_lo:.1%}, "
          f"бленд {edge_bl:.1%}, текст {((txt < 0.02) | (txt > 0.98)).mean():.1%}")

    cal = calibrate(lo, y, fold)
    variants = {
        "LoRA одна": lo,
        "LoRA калиброванная": cal,
        "бленд 0.5/0.5 (сабмит)": 0.5 * txt + 0.5 * lo,
        "бленд калибр. LoRA": 0.5 * txt + 0.5 * cal,
        "текст один": txt,
    }
    print(f"  {'вариант':26s} {'AUC':>7s} {'PR':>7s} {'F1 argmax':>10s} {'F1 квантиль':>12s}"
          f"  разброс порога")
    for tag, o in variants.items():
        fa, ta = nested(o, y, fold, "argmax")
        fq, _ = nested(o, y, fold, "quantile")
        summary[(cat, tag)] = fa
        print(f"  {tag:26s} {roc_auc_score(y, o):7.4f} {average_precision_score(y, o):7.4f} "
              f"{fa:10.4f} {fq:12.4f}  {min(ta):.2f}..{max(ta):.2f}")

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ (вложенно, argmax)")
cats = ["Легковоспламеняющиеся", "БАД"]
for tag in ["LoRA одна", "LoRA калиброванная", "бленд 0.5/0.5 (сабмит)",
            "бленд калибр. LoRA", "текст один"]:
    v = [summary[(c, tag)] for c in cats]
    print(f"  {tag:26s} {np.mean(v):.4f}   (флам {v[0]:.4f}, БАД {v[1]:.4f})")
print("\n  Если калибровка одной LoRA догоняет бленд — выигрыш бленда был в пороге,")
print("  а не в информации, и усиление адаптера перестанет гаситься.")
