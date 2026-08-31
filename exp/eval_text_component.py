"""Можно ли усилить ТЕКСТОВЫЙ компонент бленда?

Сейчас в сабмите TF-IDF по словесным 1-2 граммам. Вложенный F1 = 0.6716 на флам
при AUC 0.9670 — ранжирует прилично, а порог рушится: 198 позитивов на 5502 строки.
Для русских товарных названий («розжигсредство», «WD40», «спрей-очиститель») слова
дробятся плохо; символьные n-граммы обычно устойчивее к слитным написаниям и опечаткам.

Компонент важен: он входит в бленд с весом 0.5 и это ЕДИНСТВЕННЫЙ источник,
некоррелированный с LoRA (0.732 против 0.95+ у любых LLM-признаков).

Честность: конфигураций несколько, поэтому печатается и наивный максимум,
и вложенный выбор конфигурации. Разрыв между ними — цена перебора.
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
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]

CONFIGS = {
    "слова 1-2 (в сабмите)": [dict(analyzer="word", ngram_range=(1, 2), min_df=2)],
    "символы 3-5": [dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3)],
    "символы 2-6": [dict(analyzer="char_wb", ngram_range=(2, 6), min_df=3)],
    "слова 1-2 + символы 3-5": [dict(analyzer="word", ngram_range=(1, 2), min_df=2),
                                dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3)],
    "слова 1-3 + символы 2-5": [dict(analyzer="word", ngram_range=(1, 3), min_df=2),
                                dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3)],
}
REPS = [3, 1]          # во сколько раз повторяется название относительно описания


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def best_thr(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    i = int(f.argmax())
    return THS[i], f[i]


def nested_fixed(o, y, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= best_thr(o[tr], y[tr])[0]).astype(int)
    return f1_score(y, pred)


def nested_select(cands, y, fold):
    pred = np.zeros(len(y), dtype=int)
    picked = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        best = None
        for tag, o in cands.items():
            t, f = best_thr(o[tr], y[tr])
            if best is None or f > best[1]:
                best = (tag, f, t)
        picked.append(best[0])
        pred[te] = (cands[best[0]][te] >= best[2]).astype(int)
    return f1_score(y, pred), picked


summary = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)
    print(f"  {'конфигурация':34s} {'F1':>7s} {'AUC':>8s} {'PR':>8s} {'corr LoRA':>10s} "
          f"{'бленд 0.5':>10s}")

    cands = {}
    for rep in REPS:
        txt_in = (sub["name_c"] + " ") * rep + sub["desc_c"]
        txt_in = txt_in.values
        for tag, specs in CONFIGS.items():
            o = np.zeros(len(y))
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                ptr, pte = [], []
                for sp in specs:
                    v = TfidfVectorizer(sublinear_tf=True, max_features=300_000, **sp)
                    ptr.append(v.fit_transform(txt_in[tr]))
                    pte.append(v.transform(txt_in[te]))
                ptr += [csr_matrix(R[tr]), csr_matrix(L[tr])]
                pte += [csr_matrix(R[te]), csr_matrix(L[te])]
                o[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
                    .fit(hstack(ptr).tocsr(), y[tr]).predict_proba(hstack(pte).tocsr())[:, 1]
            key = f"{tag} [имя x{rep}]"
            cands[key] = o
            bl = 0.5 * o + 0.5 * lo
            print(f"  {key:34s} {nested_fixed(o, y, fold):7.4f} {roc_auc_score(y, o):8.4f} "
                  f"{average_precision_score(y, o):8.4f} {np.corrcoef(o, lo)[0, 1]:10.3f} "
                  f"{nested_fixed(bl, y, fold):10.4f}", flush=True)

    blends = {k: 0.5 * v + 0.5 * lo for k, v in cands.items()}
    naive_tag = max(blends, key=lambda k: nested_fixed(blends[k], y, fold))
    naive = nested_fixed(blends[naive_tag], y, fold)
    honest, picked = nested_select(blends, y, fold)
    base = nested_fixed(0.5 * cands["слова 1-2 (в сабмите)" + " [имя x3]"] + 0.5 * lo, y, fold)
    summary[cat] = (base, naive, honest)
    print(f"  бленд сабмита:              {base:.4f}")
    print(f"  НАИВНО лучший бленд:        {naive:.4f}  [{naive_tag}]")
    print(f"  ЧЕСТНО (конфиг вложенно):   {honest:.4f}  выбор: {picked}", flush=True)

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ")
cats = ["Легковоспламеняющиеся", "БАД"]
for i, tag in enumerate(["сабмит", "наивно (нельзя верить)", "ЧЕСТНО"]):
    print(f"  {tag:24s} {np.mean([summary[c][i] for c in cats]):.4f}   "
          f"(флам {summary[cats[0]][i]:.4f}, БАД {summary[cats[1]][i]:.4f})")
print("\n  цель: 0.915 | текущий сабмит: вложенно 0.9006 -> LB 0.87820")
