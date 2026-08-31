"""ОБУЧЕННЫЙ флам-бленд промптов: спасает ли flamtypes-адаптер газовые горелки в бленде.

Zero-shot бленд провалился (flam_blend_analysis.py: спасение +10 TP / +1286 FP). Но
обучение калибрует негативы. Проверяем ОБУЧЕННЫЙ flamtypes: (1) корреляция с базовыми
флам-адаптерами (потенциал декорреляции), (2) F1 при фикс. пороге 0.45 бленда,
(3) item-level спасение газовых горелок. Флам F1 неизмерим (bootstrap ±0.063), поэтому
главный сигнал — item-level счёт и корреляция, а не F1.

Базовый флам-бленд (чемпион): 0.5*текст + 0.5*среднее(Qwen-LoRA[""], gemma-LoRA["_gemma"]).
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

FLAM = "Легковоспламеняющиеся"
THR = 0.45


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    return pd.concat([pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{tag}.parquet")[["id", "lora_score"]]
                      for k in range(5)], ignore_index=True).drop_duplicates("id")


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
for tag, nm in [("", "q"), ("_gemma", "g"), ("_flamtypes", "t")]:
    df = df.merge(load_tag(tag).rename(columns={"lora_score": f"lo_{nm}"}), on="id", how="left")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]

sub = df[df["category"] == FLAM].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
R = extract(sub["full"].values, sub["nc"].values, FLAM)
L = pf(sub["llm_score"].values)
txt = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
    Bm = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, y[tr]).predict_proba(Bm)[:, 1]

loq, log_, lot = sub["lo_q"].values, sub["lo_g"].values, sub["lo_t"].values
burner = sub["name"].str.contains("горелк|баллон|газ|резак|паяльн", case=False, na=False).values

print(f"флам n={len(y)} поз={y.sum()} газовых-поз={int((burner&(y==1)).sum())}\n")
print("=== корреляции LoRA-скоров (spearman) ===")
S = pd.DataFrame({"qwen": loq, "gemma": log_, "flamtypes": lot}).corr(method="spearman")
print(f"  qwen×gemma (текущий бленд) = {S.loc['qwen','gemma']:.3f}")
print(f"  gemma×flamtypes           = {S.loc['gemma','flamtypes']:.3f}")
print(f"  qwen×flamtypes            = {S.loc['qwen','flamtypes']:.3f}")

print("\n=== F1 при фикс. пороге 0.45 (неизмеримо, ±0.063 — справочно) ===")
def rep(name, lo):
    score = 0.5 * txt + 0.5 * lo
    pred = (score >= THR).astype(int)
    print(f"  {name:28s} F1={f1_score(y,pred):.4f} AUC(lora)={roc_auc_score(y,lo):.4f}")
    return pred
p_base = rep("база: mean(qwen,gemma)", 0.5 * (loq + log_))
rep("+ flamtypes (3-среднее)", (loq + log_ + lot) / 3)
rep("mean(qwen,gemma) 0.7 + tt 0.3", 0.7 * 0.5 * (loq + log_) + 0.3 * lot)

print("\n=== item-level: спасает ли flamtypes газовые горелки, что база валит ===")
base_lora = 0.5 * (loq + log_)
base_score = 0.5 * txt + 0.5 * base_lora
base_pred = (base_score >= THR).astype(int)
# газовые позитивы, которые база пропускает
gas_fn = burner & (y == 1) & (base_pred == 0)
print(f"  газовых позитивов пропущено базой: {int(gas_fn.sum())}")
# бленд с flamtypes
for w in [0.2, 0.33, 0.5]:
    bl = (1 - w) * base_lora + w * lot
    sc = 0.5 * txt + 0.5 * bl
    pr = (sc >= THR).astype(int)
    rescued = int((gas_fn & (pr == 1)).sum())
    new_fp = int(((y == 0) & (base_pred == 0) & (pr == 1)).sum())
    lost_tp = int(((y == 1) & (base_pred == 1) & (pr == 0)).sum())
    print(f"  w_tt={w}: спасено газовых {rescued}/{int(gas_fn.sum())}, "
          f"новых FP {new_fp}, потеряно TP {lost_tp}")
