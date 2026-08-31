"""Что чемпион v20 (прицельный промпт БАД) ещё путает на БАД — проектируем следующую
БАД-формулировку. Конфигурация сабмита: 0.5*текст + 0.5*LoRA(_mmtp), порог 0.47.

БАД-OOF узкий (CI ±0.009, bootstrap_ci.py), совпадает с пабликом — значит здесь
локальные правки различимы и осмысленны, в отличие от флам (±0.063).
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


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    parts = [pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{tag}.parquet")[["id", "lora_score"]]
             for k in range(5)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("id")


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
R = extract(sub["full"].values, sub["nc"].values, BAD)
L = pf(sub["llm_score"].values)
txt = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
    Bm = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, y[tr]).predict_proba(Bm)[:, 1]

score = 0.5 * txt + 0.5 * sub["lo"].values
pred = (score >= THR).astype(int)
fam = family_labels(sub["name"].fillna("").values)
print(f"БАД: n={len(y)} поз={y.sum()} F1={f1_score(y, pred):.4f} порог {THR}")
fn = np.where((y == 1) & (pred == 0))[0]
fp = np.where((y == 0) & (pred == 1))[0]
print(f"пропуски FN={len(fn)}  ложные FP={len(fp)}")

# в смешанных ли семьях ошибка (шум разметки) или в чистых (систематика)?
def mixed(i):
    m = fam == fam[i]
    return len(set(y[m])) > 1
print(f"FN в смешанных семьях (шум): {sum(mixed(i) for i in fn)}/{len(fn)}")
print(f"FP в смешанных семьях (шум): {sum(mixed(i) for i in fp)}/{len(fp)}")

print("\n" + "=" * 96)
print("ПРОПУСКИ FN (разметка=БАД, модель=нет), по возрастанию скора — чистые семьи первыми")
ford = sorted(fn, key=lambda i: (mixed(i), score[i]))
for i in ford[:30]:
    tag = "СМЕШ" if mixed(i) else "чист"
    print(f"  [{score[i]:.3f} txt={txt[i]:.2f} lora={sub['lo'].values[i]:.2f} {tag}] {str(sub['name'][i])[:80]}")

print("\n" + "=" * 96)
print("ЛОЖНЫЕ FP (разметка=не БАД, модель=БАД), по убыванию скора — чистые семьи первыми")
pord = sorted(fp, key=lambda i: (mixed(i), -score[i]))
for i in pord[:30]:
    tag = "СМЕШ" if mixed(i) else "чист"
    print(f"  [{score[i]:.3f} txt={txt[i]:.2f} lora={sub['lo'].values[i]:.2f} {tag}] {str(sub['name'][i])[:80]}")
