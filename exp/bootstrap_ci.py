"""Bootstrap-доверительные интервалы метрики на полном OOF — ШУМОВОЙ ПОЛ валидации.

Зачем. Мы теряли сабмиты, гоняясь за приростами меньше шума (геом. среднее +0.004
локально -> -0.015 паблик; вес БАД). Нужно ЗНАТЬ, какой прирост локального OOF
статистически различим, а какой — шум. Ресемплим СЕМЬИ (не строки: CV семейный,
почти-дубли внутри семьи скоррелированы), пересчитываем F1 при ФИКСИРОВАННОМ пороге,
берём перцентили.

Конфигурация РОВНО как в сабмите, фикс. пороги:
  флам = 0.5*текст-ансамбль + 0.5*среднее(Qwen-LoRA[""], gemma-LoRA["_gemma"]), порог 0.45
  БАД  = 0.5*текст-ансамбль + 0.5*LoRA(<тег>), порог 0.47
Сравниваем БАД-адаптеры v16 (_mm1) и v20 (_mmtp): попадает ли разница в CI.

Запуск: python exp/bootstrap_ci.py
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

FLAM, BAD = "Легковоспламеняющиеся", "БАД"
THR = {FLAM: 0.45, BAD: 0.47}
B = 3000
RNG = np.random.RandomState(0)


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


def load_tag(tag):
    parts = [pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{tag}.parquet")[["id", "lora_score"]]
             for k in range(5)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("id")


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]

# текстовый ансамбль (nested по фолду) + флам-LoRA бленд — считаем один раз
tags_flam = load_tag("").rename(columns={"lora_score": "lo_q"}) \
    .merge(load_tag("_gemma").rename(columns={"lora_score": "lo_g"}), on="id")
df = df.merge(tags_flam, on="id", how="left")


def build_score(cat, bad_tag):
    sub = df[df["category"] == cat].reset_index(drop=True)
    y, fold = sub["label"].values, sub["fold"].values
    R = extract(sub["full"].values, sub["nc"].values, cat)
    L = pf(sub["llm_score"].values)
    txt = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]), csr_matrix(L[tr])]).tocsr()
        Bm = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]), csr_matrix(L[te])]).tocsr()
        txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(A, y[tr]).predict_proba(Bm)[:, 1]
    if cat == FLAM:
        lo = 0.5 * (sub["lo_q"].values + sub["lo_g"].values)
    else:
        lt = load_tag(bad_tag).rename(columns={"lora_score": "lo"})
        lo = sub.merge(lt, on="id", how="left")["lo"].values
    score = 0.5 * txt + 0.5 * lo
    fam = family_labels(sub["name"].fillna("").values)
    return y, score, fam


def boot(y, score, fam, thr):
    pred = (score >= thr).astype(int)
    point = f1_score(y, pred)
    fams = np.unique(fam)
    fam_idx = {f: np.where(fam == f)[0] for f in fams}
    vals = np.empty(B)
    for b in range(B):
        pick = fams[RNG.randint(0, len(fams), len(fams))]
        idx = np.concatenate([fam_idx[f] for f in pick])
        vals[b] = f1_score(y[idx], pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, vals


print("считаю текст-ансамбль и бленды (это медленная часть)...", flush=True)
yf, sf, ff = build_score(FLAM, None)
yb, sb16, bb = build_score(BAD, "_mm1")
_, sb20, _ = build_score(BAD, "_mmtp")

pf_, lf, hf, vf = boot(yf, sf, ff, THR[FLAM])
p16, l16, h16, v16 = boot(yb, sb16, bb, THR[BAD])
p20, l20, h20, v20 = boot(yb, sb20, bb, THR[BAD])

print("\n" + "=" * 78)
print(f"ФЛАМ  F1={pf_:.4f}  95%CI [{lf:.4f}, {hf:.4f}]  ширина {hf-lf:.4f}  (порог {THR[FLAM]})")
print(f"БАД v16(_mm1)  F1={p16:.4f}  95%CI [{l16:.4f}, {h16:.4f}]  ширина {h16-l16:.4f}")
print(f"БАД v20(_mmtp) F1={p20:.4f}  95%CI [{l20:.4f}, {h20:.4f}]  ширина {h20-l20:.4f}")
# парная разница v20-v16 (тот же ресемпл-сид -> сопоставимо)
diff = v20 - v16
dl, dh = np.percentile(diff, [2.5, 97.5])
print(f"\nПАРНАЯ разница БАД v20-v16: {p20-p16:+.4f}  95%CI [{dl:+.4f}, {dh:+.4f}]  "
      f"P(v20>v16)={np.mean(diff>0):.2f}")
m16, m20 = 0.5 * (pf_ + p16), 0.5 * (pf_ + p20)
print(f"\nМетрика (среднее F1): v16={m16:.4f}  v20={m20:.4f}  разница {m20-m16:+.4f}")
print(f"ШУМОВОЙ ПОЛ: прирост метрики надо мерить против CI-ширины БАД ~{(h20-l20)/2:.4f} и флам ~{(hf-lf)/2:.4f}")
