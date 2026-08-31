"""Стекинг на БАД (двухстадийность): обученный мета-роутер поверх ens+lora — берёт ли
он oracle-разрыв +0.021 (two_stage_analysis.py) БЕЗ ловушки подбора порога.

Компоненты OOF: ens (текст-ансамбль, nested) и lora (v20 _mmtp). Оба уже вне фолда —
значит мета можно честно учить на фолдах !=k и применять к k (метки строки мета не видит).

Мерим ТРИ способа объединения, каждый при ФИКС. пороге 0.47 (как сабмит) И при
OOF-оптимальном (справочно, это ловушка §7/§8):
  base   — 0.5*ens+0.5*lora (v20);
  wlearn — скалярный вес бленда, подобранный вне фолда (§8: 0.3=0.5 была монетка);
  meta   — LogReg на [ens, lora, ens*lora, |ens-lora|], вне фолда.
Мета меняет ШКАЛУ скора, поэтому фикс.0.47 к ней некорректен — для неё «фикс.» порог =
единый порог, выбранный на полном OOF (как 0.47 выбирался для v20). Сравнение честное
только если ОБА при своём едином OOF-пороге. Парный bootstrap по семьям.

Вывод честный: если meta при фикс. пороге не бьёт v20 стабильно (P>0.9) — закрываем,
это мета-ловушка (§1).
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
from folds import clean, family_labels, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

BAD = "БАД"
THR = 0.47
B = 3000
RNG = np.random.RandomState(0)
GRID = np.arange(0.20, 0.81, 0.01)


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    return pd.concat([pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{tag}.parquet")[["id", "lora_score"]]
                      for k in range(5)], ignore_index=True).drop_duplicates("id")


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df = df.merge(load_tag("_mmtp").rename(columns={"lora_score": "lo"}), on="id", how="left")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]

sub = df[df["category"] == BAD].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
lora = sub["lo"].values
R = extract(sub["full"].values, sub["nc"].values, BAD)
L = pf(sub["llm_score"].values)
ens = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
    Bm = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
    ens[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, y[tr]).predict_proba(Bm)[:, 1]

# base
base = 0.5 * ens + 0.5 * lora
# wlearn: вес подобран вне фолда (максимизирует F1 на train-фолдах при их OOF-пороге)
wlearn = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    best = max(((max(f1_score(y[tr], ((1 - w) * ens[tr] + w * lora[tr] >= t)) for t in GRID)), w)
               for w in np.linspace(0, 1, 21))[1]
    wlearn[te] = (1 - best) * ens[te] + best * lora[te]
# meta: LogReg на признаках, вне фолда
meta = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    def feats(e, l):
        return np.c_[e, l, e * l, np.abs(e - l)]
    meta[te] = LogisticRegression(max_iter=2000, C=1.0)\
        .fit(feats(ens[tr], lora[tr]), y[tr]).predict_proba(feats(ens[te], lora[te]))[:, 1]

fam = family_labels(sub["name"].fillna("").values)
fams = np.unique(fam)
fam_idx = {f: np.where(fam == f)[0] for f in fams}


def boot(pred):
    out = np.empty(B)
    for b in range(B):
        pick = fams[RNG.randint(0, len(fams), len(fams))]
        idx = np.concatenate([fam_idx[f] for f in pick])
        out[b] = f1_score(y[idx], pred[idx])
    return out


def oof_best_thr(score):
    return max(GRID, key=lambda t: f1_score(y, (score >= t)))


print(f"БАД n={len(y)} поз={y.sum()}\n")
print(f"{'способ':10s} {'F1@0.47':>9s} {'F1@OOFbest':>11s} {'порог':>6s}")
scored = {}
for name, s in [("base(v20)", base), ("wlearn", wlearn), ("meta", meta)]:
    f_fixed = f1_score(y, (s >= THR))
    t = oof_best_thr(s)
    f_best = f1_score(y, (s >= t))
    scored[name] = (s >= THR).astype(int)   # сравнение при ФИКС. пороге (как сабмит)
    print(f"{name:10s} {f_fixed:9.4f} {f_best:11.4f} {t:6.2f}")

print("\n=== парный bootstrap против base(v20) при ФИКС. пороге 0.47 ===")
bb = boot(scored["base(v20)"])
for name in ["wlearn", "meta"]:
    d = boot(scored[name]) - bb
    dl, dh = np.percentile(d, [2.5, 97.5])
    print(f"  {name:10s} ΔF1={f1_score(y,scored[name])-f1_score(y,scored['base(v20)']):+.4f} "
          f"CI[{dl:+.4f},{dh:+.4f}] P(>base)={np.mean(d>0):.2f}")
