"""Обучение финальной TF-IDF+LogReg модели и экспорт в версионно-независимый формат.

Экспортируем vocabulary/idf/coef/intercept в .npz + .json, чтобы в докер-образе
не зависеть от версии sklearn при распиковке.
"""
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

ROOT = "/workspace/counter/"
OUTDIR = ROOT + "submit/artifacts/"

CFG = dict(word_ng=(1, 2), min_df=2, max_feat=200_000, C=10.0, name_rep=3)
CATS = ["БАД", "Легковоспламеняющиеся"]

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(s):
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip().lower()


def build_text(name, desc, name_rep=CFG["name_rep"]):
    n, d = clean(name), clean(desc)
    return ((n + " ") * name_rep + d).strip()


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["txt"] = [build_text(n, d) for n, d in zip(df["name"].fillna(""), df["description"].fillna(""))]
df["gid"] = df.groupby(df["txt"]).ngroup()

artifacts = {}
print("=" * 90)
for cat in CATS:
    sub = df[df["category"] == cat].reset_index(drop=True)
    X, y, g = sub["txt"].values, sub["label"].values, sub["gid"].values

    # --- OOF для выбора порога ---
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g):
        v = TfidfVectorizer(ngram_range=CFG["word_ng"], min_df=CFG["min_df"],
                            sublinear_tf=True, max_features=CFG["max_feat"])
        Xtr = v.fit_transform(X[tr])
        c = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced")
        c.fit(Xtr, y[tr])
        oof[te] = c.predict_proba(v.transform(X[te]))[:, 1]

    ths = np.linspace(0.05, 0.95, 181)
    tb = max((f1_score(y, (oof >= t).astype(int)), t) for t in ths)
    tm = max((f1_score(y, (oof >= t).astype(int), average="macro"), t) for t in ths)
    # компромиссный порог: максимизируем сумму нормированных F1 обеих версий метрики
    comb = max(((f1_score(y, (oof >= t).astype(int)) / tb[0]
                 + f1_score(y, (oof >= t).astype(int), average="macro") / tm[0]), t) for t in ths)
    thr = round(float(comb[1]), 3)
    print(f"\n### {cat}: n={len(y)} pos={y.sum()}")
    print(f"  best binary  F1={tb[0]:.4f} @ t={tb[1]:.3f}")
    print(f"  best macro   F1={tm[0]:.4f} @ t={tm[1]:.3f}")
    print(f"  ВЫБРАН компромиссный порог t={thr}: "
          f"binary={f1_score(y,(oof>=thr).astype(int)):.4f} "
          f"macro={f1_score(y,(oof>=thr).astype(int),average='macro'):.4f}")
    print("  профиль по порогу:")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        p = (oof >= t).astype(int)
        print(f"    t={t:.1f} bin={f1_score(y,p):.4f} mac={f1_score(y,p,average='macro'):.4f} pred1={p.sum()}")

    # --- финальная модель на всех данных категории ---
    vec = TfidfVectorizer(ngram_range=CFG["word_ng"], min_df=CFG["min_df"],
                          sublinear_tf=True, max_features=CFG["max_feat"])
    Xall = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=CFG["C"], class_weight="balanced")
    clf.fit(Xall, y)

    vocab = vec.vocabulary_
    terms = [None] * len(vocab)
    for t, i in vocab.items():
        terms[i] = t
    artifacts[cat] = dict(
        terms=terms,
        idf=vec.idf_.astype(np.float32),
        coef=clf.coef_[0].astype(np.float32),
        intercept=float(clf.intercept_[0]),
        threshold=thr,
        oof_bin=float(f1_score(y, (oof >= thr).astype(int))),
        oof_mac=float(f1_score(y, (oof >= thr).astype(int), average="macro")),
    )
    # sanity: обучающая выборка
    p_tr = clf.predict_proba(Xall)[:, 1]
    print(f"  train F1 (переобучение, для справки) bin={f1_score(y,(p_tr>=thr).astype(int)):.4f}")

print("\n" + "=" * 90)
print("ИТОГО на OOF с выбранными порогами:")
print(f"  mean binary F1 = {np.mean([artifacts[c]['oof_bin'] for c in CATS]):.4f}")
print(f"  mean macro  F1 = {np.mean([artifacts[c]['oof_mac'] for c in CATS]):.4f}")

# ---------------------------------------------------------------- экспорт
import os
os.makedirs(OUTDIR, exist_ok=True)
meta = {"config": {k: list(v) if isinstance(v, tuple) else v for k, v in CFG.items()},
        "categories": {}}
for cat in CATS:
    a = artifacts[cat]
    slug = "bad" if cat == "БАД" else "flam"
    np.savez_compressed(OUTDIR + f"model_{slug}.npz",
                        idf=a["idf"], coef=a["coef"])
    with open(OUTDIR + f"vocab_{slug}.json", "w", encoding="utf-8") as f:
        json.dump(a["terms"], f, ensure_ascii=False)
    meta["categories"][cat] = dict(slug=slug, intercept=a["intercept"], threshold=a["threshold"],
                                   n_features=len(a["terms"]),
                                   oof_f1_binary=a["oof_bin"], oof_f1_macro=a["oof_mac"])
with open(OUTDIR + "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("\nэкспортировано в", OUTDIR)
for f in sorted(os.listdir(OUTDIR)):
    print(f"  {f}  {os.path.getsize(OUTDIR+f)/1024:.0f} KB")
