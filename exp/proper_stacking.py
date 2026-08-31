"""НАСТОЯЩИЙ стекинг вместо конкатенации признаков.

В eval_full_oof.py я подавал в LogReg сырую TF-IDF-матрицу вместе с LoRA-скором.
Это не стекинг: TF-IDF переобучается на обучающих фолдах, выглядит там сильным,
получает большой вес — и вес не переносится. Поэтому «tfidf+rules+lora» (0.766)
оказался хуже «LoRA одна» (0.819).

Здесь базовые модели дают ВНЕШНЕ-ФОЛДОВЫЕ предсказания (вложенный внутренний CV),
и мета-модель работает с 2-4 честными числами вместо 200k сырых признаков.

Всё меряется вложенно: порог берётся из фолдов != k.
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THS = np.linspace(0.01, 0.99, 197)

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
qq = pd.read_parquet(ROOT + "exp/llm_questions.parquet")
qcols = [c for c in qq.columns if c.startswith("q_")]
df = df.merge(qq[["id"] + qcols], on="id", how="left")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold*.parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
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


def nested_f1(o, y, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= pick(o[tr], y[tr])).astype(int)
    return f1_score(y, pred)


def base_text_oof(txt, y, fold, extra_blocks):
    """Внешне-фолдовое предсказание текстовой модели: для каждого фолда k модель
    учится на != k. Для строк ВНУТРИ обучающей части мета-модели нужен ещё один
    уровень out-of-fold, иначе мета-модель увидит переобученные значения."""
    o = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = v.fit_transform(txt[tr])
        Xte = v.transform(txt[te])
        ptr = [Xtr] + [csr_matrix(b[tr]) for b in extra_blocks]
        pte = [Xte] + [csr_matrix(b[te]) for b in extra_blocks]
        m = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")
        m.fit(hstack(ptr).tocsr(), y[tr])
        o[te] = m.predict_proba(hstack(pte).tocsr())[:, 1]
    return o


for cat in ["Легковоспламеняющиеся", "БАД"]:
    m = (df["category"] == cat).values
    sub = df[m].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    txt = sub["txt"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    cols = [c for c in qcols if sub[c].notna().any()]
    Q = np.hstack([pf(sub[c].fillna(0.5).values) for c in cols])
    lo = sub["lora_score"].values
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)

    text_oof = base_text_oof(txt, y, fold, [R, L])
    print(f"  базовые: текст-модель AUC={roc_auc_score(y,text_oof):.4f} "
          f"F1={nested_f1(text_oof,y,fold):.4f} | "
          f"LoRA AUC={roc_auc_score(y,lo):.4f} F1={nested_f1(lo,y,fold):.4f}", flush=True)
    print(f"  корреляция предсказаний: {np.corrcoef(text_oof, lo)[0,1]:.3f}", flush=True)

    META = {
        "LoRA одна (t=0.5)": None,
        "мета[текст, LoRA]": [text_oof, lo],
        "мета[текст, LoRA, вопросы]": [text_oof, lo],
    }
    print(f"  {'LoRA одна, фикс. t=0.5':30s} F1={f1_score(y,(lo>=0.5).astype(int)):.4f}")
    print(f"  {'LoRA одна, вложенный t':30s} F1={nested_f1(lo,y,fold):.4f}")

    for tag, use_q in [("мета[текст, LoRA]", False), ("мета[текст, LoRA, вопросы]", True)]:
        o = np.zeros(len(y))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            feats = [pf(text_oof), pf(lo)] + ([Q] if use_q else [])
            X = np.hstack(feats)
            mm = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
            mm.fit(X[tr], y[tr])
            o[te] = mm.predict_proba(X[te])[:, 1]
        print(f"  {tag:30s} F1={nested_f1(o,y,fold):.4f} AUC={roc_auc_score(y,o):.4f} "
              f"PR={average_precision_score(y,o):.4f}", flush=True)

    for w in [0.2, 0.3, 0.5, 0.7, 0.85]:
        bl = (1 - w) * text_oof + w * lo
        print(f"  {'бленд w_lora='+format(w,'.2f'):30s} F1={nested_f1(bl,y,fold):.4f} "
              f"AUC={roc_auc_score(y,bl):.4f}", flush=True)
