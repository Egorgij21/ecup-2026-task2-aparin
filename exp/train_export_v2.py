"""Обучение и экспорт ансамбля v2: TF-IDF + правила + zero-shot LLM.

Экспортируем в версионно-независимый формат (json + npz), как в v1.
Дополнительно обучаем FALLBACK-модель без LLM-признаков: если в контейнере
не удастся загрузить Qwen, решение не упадёт, а деградирует до v1+правила.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from folds import clean, make_folds  # noqa: E402
from rule_features import extract, feature_names  # noqa: E402

ROOT = "/workspace/counter/"
OUT = ROOT + "submit2/artifacts/"
CFG = dict(word_ng=[1, 2], min_df=2, max_feat=200_000, C=10.0, name_rep=3)
CATS = ["БАД", "Легковоспламеняющиеся"]
SLUG = {"БАД": "bad", "Легковоспламеняющиеся": "flam"}


def llm_feats(s):
    s = np.clip(np.asarray(s, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > 0.5).astype(float),
                 (s > 0.8).astype(float), (s < 0.2).astype(float)]


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]],
              on="id", how="left")
assert df["llm_score"].notna().all()
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt"] = (df["name_c"] + " ") * CFG["name_rep"] + df["desc_c"]
df["fold"] = make_folds(df)

meta = {"config": CFG, "categories": {}, "rule_names": {}}
os.makedirs(OUT, exist_ok=True)
summary = {}

for cat in CATS:
    sub = df[df["category"] == cat].reset_index(drop=True)
    y = sub["label"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = llm_feats(sub["llm_score"].values)
    print(f"\n### {cat}: n={len(y)} pos={y.sum()}  правил={R.shape[1]}")

    # --- OOF на честных (семейных) фолдах: и с LLM, и без ---
    oof = {"llm": np.zeros(len(y)), "nollm": np.zeros(len(y))}
    for k in range(5):
        tr, te = np.where(sub["fold"] != k)[0], np.where(sub["fold"] == k)[0]
        v = TfidfVectorizer(ngram_range=tuple(CFG["word_ng"]), min_df=CFG["min_df"],
                            sublinear_tf=True, max_features=CFG["max_feat"])
        Xtr, Xte = v.fit_transform(sub["txt"].values[tr]), v.transform(sub["txt"].values[te])
        A_tr = hstack([Xtr, csr_matrix(R[tr])]).tocsr()
        A_te = hstack([Xte, csr_matrix(R[te])]).tocsr()
        oof["nollm"][te] = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced")\
            .fit(A_tr, y[tr]).predict_proba(A_te)[:, 1]
        B_tr = hstack([A_tr, csr_matrix(L[tr])]).tocsr()
        B_te = hstack([A_te, csr_matrix(L[te])]).tocsr()
        oof["llm"][te] = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced")\
            .fit(B_tr, y[tr]).predict_proba(B_te)[:, 1]

    ths = np.linspace(0.05, 0.95, 181)
    entry = {"slug": SLUG[cat]}
    for key in ["llm", "nollm"]:
        o = oof[key]
        tb = max((f1_score(y, (o >= t).astype(int)), t) for t in ths)
        tm = max((f1_score(y, (o >= t).astype(int), average="macro"), t) for t in ths)
        comb = max(((f1_score(y, (o >= t).astype(int)) / tb[0]
                     + f1_score(y, (o >= t).astype(int), average="macro") / tm[0]), t) for t in ths)
        thr = round(float(comb[1]), 3)
        fb, fm = f1_score(y, (o >= thr).astype(int)), f1_score(y, (o >= thr).astype(int), average="macro")
        print(f"  [{key:5s}] best bin={tb[0]:.4f}@{tb[1]:.2f} mac={tm[0]:.4f}@{tm[1]:.2f}"
              f"  -> компромисс t={thr}: bin={fb:.4f} mac={fm:.4f}")
        entry[f"threshold_{key}"] = thr
        entry[f"oof_bin_{key}"] = float(fb)
        entry[f"oof_mac_{key}"] = float(fm)
        summary.setdefault(key, {})[cat] = (fb, fm)

    # --- финальные модели на всех данных ---
    vec = TfidfVectorizer(ngram_range=tuple(CFG["word_ng"]), min_df=CFG["min_df"],
                          sublinear_tf=True, max_features=CFG["max_feat"])
    Xall = vec.fit_transform(sub["txt"].values)
    A = hstack([Xall, csr_matrix(R)]).tocsr()
    B = hstack([A, csr_matrix(L)]).tocsr()
    clf_nl = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced").fit(A, y)
    clf_l = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced").fit(B, y)

    nt, nr = Xall.shape[1], R.shape[1]
    terms = [None] * nt
    for t, i in vec.vocabulary_.items():
        terms[i] = t
    with open(OUT + f"vocab_{SLUG[cat]}.json", "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False)
    np.savez_compressed(
        OUT + f"model_{SLUG[cat]}.npz",
        idf=vec.idf_.astype(np.float32),
        coef_tfidf_llm=clf_l.coef_[0][:nt].astype(np.float32),
        coef_rules_llm=clf_l.coef_[0][nt:nt + nr].astype(np.float32),
        coef_llm=clf_l.coef_[0][nt + nr:].astype(np.float32),
        coef_tfidf_nollm=clf_nl.coef_[0][:nt].astype(np.float32),
        coef_rules_nollm=clf_nl.coef_[0][nt:].astype(np.float32),
    )
    entry["intercept_llm"] = float(clf_l.intercept_[0])
    entry["intercept_nollm"] = float(clf_nl.intercept_[0])
    entry["n_features"] = int(nt)
    entry["n_rules"] = int(nr)
    meta["categories"][cat] = entry
    meta["rule_names"][cat] = feature_names(cat)

with open(OUT + "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 90)
for key, label in [("llm", "С LLM (основная)"), ("nollm", "без LLM (fallback)")]:
    mb = np.mean([summary[key][c][0] for c in CATS])
    mm = np.mean([summary[key][c][1] for c in CATS])
    print(f"  {label:22s} честная CV: mean bin={mb:.4f}  mean mac={mm:.4f}")
print("  для сравнения: сабмит v1 честная CV bin=0.7830, факт на LB 0.73867")
print("\nэкспортировано в", OUT)
for f in sorted(os.listdir(OUT)):
    print(f"  {f}  {os.path.getsize(OUT+f)/1024:.0f} KB")
