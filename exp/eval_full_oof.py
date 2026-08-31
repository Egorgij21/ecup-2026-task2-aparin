"""Итоговая ЧЕСТНАЯ оценка, когда собран полный OOF по LoRA.

Все замеры — вложенные: порог берётся из фолдов != k и применяется к фолду k.
Наивные цифры больше не приводим, они дважды нас обманули (v1 и v3).

Сравниваем варианты по каждой категории отдельно, потому что метрика соревнования —
среднее двух независимых F1, и оптимальный состав признаков у категорий разный.
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

THS = np.linspace(0.01, 0.99, 197)

# ---------------------------------------------------------------- данные
df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
qq = pd.read_parquet(ROOT + "exp/llm_questions.parquet")
qcols = [c for c in qq.columns if c.startswith("q_")]
df = df.merge(qq[["id"] + qcols], on="id", how="left")

parts = [pd.read_parquet(p)[["id", "lora_score"]]
         for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))]
lora = pd.concat(parts).drop_duplicates("id")
df = df.merge(lora, on="id", how="left")
have = df["lora_score"].notna()
print(f"LoRA OOF покрывает {have.sum()} из {len(df)} строк "
      f"({have.mean()*100:.1f}%) — фолдов собрано: {len(parts)}")

df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * 3 + df["desc_c"]


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5).astype(float),
                 (s > .8).astype(float), (s < .2).astype(float)]


def pick(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    return THS[int(f.argmax())]


def plateau_width(o, y):
    f = np.array([f1_score(y, (o >= t).astype(int)) for t in THS])
    good = THS[f >= f.max() * 0.97]
    return good.max() - good.min()


results = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    m = (df["category"] == cat).values & have.values
    sub = df[m].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    cols = [c for c in qcols if sub[c].notna().any()]
    Q = np.hstack([pf(sub[c].fillna(0.5).values) for c in cols])
    LO = pf(sub["lora_score"].values)
    lora_raw = sub["lora_score"].values
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)

    BLOCKS = {"rules": R, "llm": L, "quest": Q, "lora": LO}
    VARIANTS = {
        "tfidf+rules": ["rules"],
        "tfidf+rules+llm (v2)": ["rules", "llm"],
        "tfidf+rules+quest": ["rules", "quest"],
        "tfidf+rules+lora": ["rules", "lora"],
        "tfidf+rules+llm+lora": ["rules", "llm", "lora"],
        "tfidf+rules+quest+lora": ["rules", "quest", "lora"],
        "всё вместе": ["rules", "llm", "quest", "lora"],
    }

    # чистая LoRA — без стекинга, порог фиксирован в 0.5 (центр плато)
    f_fixed = f1_score(y, (lora_raw >= 0.5).astype(int))
    pred_nested = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        if len(te) == 0:
            continue
        pred_nested[te] = (lora_raw[te] >= pick(lora_raw[tr], y[tr])).astype(int)
    print(f"  {'LoRA одна, t=0.5':26s} F1={f_fixed:.4f}  "
          f"AUC={roc_auc_score(y, lora_raw):.4f} PR={average_precision_score(y, lora_raw):.4f} "
          f"плато={plateau_width(lora_raw, y):.2f}", flush=True)
    print(f"  {'LoRA одна, вложенный t':26s} F1={f1_score(y, pred_nested):.4f}", flush=True)
    results[(cat, "LoRA одна, t=0.5")] = f_fixed
    results[(cat, "LoRA одна, вложенный t")] = f1_score(y, pred_nested)

    for tag, blocks in VARIANTS.items():
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            if len(te) == 0:
                continue
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                max_features=200_000)
            ptr = [v.fit_transform(sub["txt"].values[tr])]
            pte = [v.transform(sub["txt"].values[te])]
            for b in blocks:
                ptr.append(csr_matrix(BLOCKS[b][tr]))
                pte.append(csr_matrix(BLOCKS[b][te]))
            mdl = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")
            mdl.fit(hstack(ptr).tocsr(), y[tr])
            o[te] = mdl.predict_proba(hstack(pte).tocsr())[:, 1]
        pred = np.zeros(len(y), dtype=int)
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            if len(te) == 0:
                continue
            pred[te] = (o[te] >= pick(o[tr], y[tr])).astype(int)
        f = f1_score(y, pred)
        results[(cat, tag)] = f
        print(f"  {tag:26s} F1={f:.4f}  AUC={roc_auc_score(y, o):.4f} "
              f"PR={average_precision_score(y, o):.4f} плато={plateau_width(o, y):.2f}", flush=True)

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ = среднее лучших вариантов по категориям (вложенно)")
tags = sorted({t for (_, t) in results})
best = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    b = max((results[(cat, t)], t) for t in tags if (cat, t) in results)
    best[cat] = b
    print(f"  {cat:24s} лучший: {b[1]:26s} {b[0]:.4f}")
print(f"\n  ИТОГО: {np.mean([best[c][0] for c in best]):.4f}")
print("\n  ориентиры: v2 вложенно 0.7954 -> LB 0.8150 | v3 вложенно 0.8507 -> LB 0.80556")
