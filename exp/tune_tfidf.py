"""Подбор конфигурации TF-IDF+LogReg на StratifiedGroupKFold. Только текст."""
import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

ROOT = "/workspace/counter/"
df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = _TAG.sub(" ", s)
    return _WS.sub(" ", s).strip().lower()


df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["gid"] = df.groupby(df["name_c"] + "|||" + df["desc_c"]).ngroup()


def make_text(sub, name_rep):
    return ((sub["name_c"] + " ") * name_rep + sub["desc_c"]).values


def evaluate(cfg, verbose=False):
    """cfg -> dict per category с F1bin/F1macro/AUC и лучшими порогами."""
    out = {}
    for cat in ["БАД", "Легковоспламеняющиеся"]:
        sub = df[df["category"] == cat].reset_index(drop=True)
        txt = make_text(sub, cfg["name_rep"])
        y = sub["label"].values
        g = sub["gid"].values
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(txt, y, g):
            vecs, Xtr_parts, Xte_parts = [], [], []
            vw = TfidfVectorizer(ngram_range=cfg["word_ng"], min_df=cfg["min_df"],
                                 sublinear_tf=True, max_features=cfg["max_feat"])
            Xtr_parts.append(vw.fit_transform(txt[tr]))
            Xte_parts.append(vw.transform(txt[te]))
            if cfg["char_ng"]:
                vc = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg["char_ng"],
                                     min_df=cfg["min_df"], sublinear_tf=True,
                                     max_features=cfg["max_feat_char"])
                Xtr_parts.append(vc.fit_transform(txt[tr]))
                Xte_parts.append(vc.transform(txt[te]))
            Xtr = hstack(Xtr_parts).tocsr() if len(Xtr_parts) > 1 else Xtr_parts[0]
            Xte = hstack(Xte_parts).tocsr() if len(Xte_parts) > 1 else Xte_parts[0]
            clf = LogisticRegression(max_iter=3000, C=cfg["C"],
                                     class_weight="balanced" if cfg["balanced"] else None)
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        ths = np.linspace(0.05, 0.95, 91)
        fb = max((f1_score(y, (oof >= t).astype(int)), t) for t in ths)
        fm = max((f1_score(y, (oof >= t).astype(int), average="macro"), t) for t in ths)
        out[cat] = dict(auc=roc_auc_score(y, oof), pr=average_precision_score(y, oof),
                        f1bin=fb[0], tbin=fb[1], f1mac=fm[0], tmac=fm[1], oof=oof)
    return out


BASE = dict(name_rep=1, word_ng=(1, 2), char_ng=None, min_df=2, C=4.0,
            max_feat=200_000, max_feat_char=300_000, balanced=True)

GRID = [
    ("base word(1,2) C=4", {}),
    ("word(1,1)", dict(word_ng=(1, 1))),
    ("word(1,3)", dict(word_ng=(1, 3))),
    ("C=1", dict(C=1.0)),
    ("C=10", dict(C=10.0)),
    ("C=30", dict(C=30.0)),
    ("min_df=1", dict(min_df=1)),
    ("name x3", dict(name_rep=3)),
    ("name x3 + C=10", dict(name_rep=3, C=10.0)),
    ("+char_wb(3,5)", dict(char_ng=(3, 5))),
    ("+char_wb(3,5) name x3", dict(char_ng=(3, 5), name_rep=3)),
    ("+char_wb(3,5) C=10", dict(char_ng=(3, 5), C=10.0)),
    ("no class_weight", dict(balanced=False)),
]

rows = []
for tag, upd in GRID:
    cfg = {**BASE, **upd}
    r = evaluate(cfg)
    mb = np.mean([r[c]["f1bin"] for c in r])
    mm = np.mean([r[c]["f1mac"] for c in r])
    rows.append(dict(cfg=tag, mean_bin=mb, mean_mac=mm,
                     bad_bin=r["БАД"]["f1bin"], flam_bin=r["Легковоспламеняющиеся"]["f1bin"],
                     bad_mac=r["БАД"]["f1mac"], flam_mac=r["Легковоспламеняющиеся"]["f1mac"],
                     bad_t=r["БАД"]["tbin"], flam_t=r["Легковоспламеняющиеся"]["tbin"],
                     bad_auc=r["БАД"]["auc"], flam_auc=r["Легковоспламеняющиеся"]["auc"]))
    print(f"{tag:28s} mean_bin={mb:.4f} mean_mac={mm:.4f} | "
          f"БАД {r['БАД']['f1bin']:.4f}/{r['БАД']['f1mac']:.4f} "
          f"ФЛАМ {r['Легковоспламеняющиеся']['f1bin']:.4f}/{r['Легковоспламеняющиеся']['f1mac']:.4f}",
          flush=True)

res = pd.DataFrame(rows).sort_values("mean_bin", ascending=False)
print("\n=== ПО mean binary F1 ===")
print(res.round(4).to_string(index=False))
print("\n=== ПО mean macro F1 ===")
print(res.sort_values("mean_mac", ascending=False).round(4).to_string(index=False))
res.to_csv(ROOT + "exp/tune_tfidf.csv", index=False)
