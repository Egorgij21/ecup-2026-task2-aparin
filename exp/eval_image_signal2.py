"""Визуальный сигнал, замер номер два — с центрированием.

Первый замер (exp/eval_image_signal.log) был НЕВАЛИДЕН, а не отрицателен:
медианный попарный косинус эмбеддингов 0.9998, косинус с центроидом >=0.9974,
std по столбцам 0.0004 при норме вектора 1. Сырое среднее по патчам ViT почти
целиком состоит из общей для всех картинок компоненты; полезная дисперсия тонет
в ней на три порядка, и логрег вырождается в константу (AUC ровно 0.5000 на БАД).

Здесь общая компонента убирается: центрирование + стандартизация по столбцам,
обучаемые ТОЛЬКО на train-фолде (меток не используют, но правило есть правило).
Дополнительно PCA — она заодно отбрасывает ведущие направления, если те окажутся
шумом освещения/фона.

Критерий тот же, что был задан заранее: корреляция с LoRA заметно ниже 0.73,
и бленд обязан бить LoRA в одиночку.
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THS = np.linspace(0.01, 0.99, 197)

E = np.load(ROOT + "exp/img_emb.npy")
idx = pd.read_parquet(ROOT + "exp/img_emb_index.parquet")

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
assert (df["id"].values == idx["id"].values).all(), "порядок строк не совпал"
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
assert len(df) == len(E), f"после слияний строк {len(df)}, эмбеддингов {len(E)}"
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


def nested(o, y, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        pred[te] = (o[te] >= pick(o[tr], y[tr])).astype(int)
    return f1_score(y, pred)


summary = {}
for cat in ["Легковоспламеняющиеся", "БАД"]:
    m = (df["category"] == cat).values
    sub = df[m].reset_index(drop=True)
    Ecat = E[m]
    y, fold = sub["label"].values, sub["fold"].values
    lo = sub["lora_score"].values
    R = extract(sub["full"].values, sub["name_c"].values, cat)
    L = pf(sub["llm_score"].values)
    print(f"\n########## {cat}: n={len(y)} pos={y.sum()}", flush=True)

    variants = {}
    for tag, npc in [("центр+станд", 0), ("PCA-64", 64), ("PCA-128", 128), ("PCA-256", 256)]:
        for C in ([0.1, 1.0] if npc in (0, 128) else [1.0]):
            o = np.zeros(len(y))
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                sc = StandardScaler().fit(Ecat[tr])
                Xtr, Xte = sc.transform(Ecat[tr]), sc.transform(Ecat[te])
                if npc:
                    p = PCA(n_components=npc, random_state=0).fit(Xtr)
                    Xtr, Xte = p.transform(Xtr), p.transform(Xte)
                o[te] = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")\
                    .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
            variants[f"{tag} C={C}"] = o
            print(f"  картинки [{tag} C={C}]  F1={nested(o, y, fold):.4f} "
                  f"AUC={roc_auc_score(y, o):.4f} PR={average_precision_score(y, o):.4f} "
                  f"corr с LoRA={np.corrcoef(o, lo)[0, 1]:.3f}", flush=True)

    img = max(variants.items(), key=lambda kv: roc_auc_score(y, kv[1]))
    print(f"  -> лучший по AUC: {img[0]}", flush=True)
    img = img[1]

    txt = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([v.fit_transform(sub["txt"].values[tr]), csr_matrix(R[tr]),
                      csr_matrix(L[tr])]).tocsr()
        Xte = hstack([v.transform(sub["txt"].values[te]), csr_matrix(R[te]),
                      csr_matrix(L[te])]).tocsr()
        txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
            .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    print(f"  текст: F1={nested(txt, y, fold):.4f} AUC={roc_auc_score(y, txt):.4f} "
          f"corr с LoRA={np.corrcoef(txt, lo)[0, 1]:.3f}")
    print(f"  корреляция картинки x текст: {np.corrcoef(img, txt)[0, 1]:.3f}")

    print("  --- бленды ---")
    for tag, o in {
        "текст + LoRA (сабмит)": 0.5 * txt + 0.5 * lo,
        "картинки + LoRA": 0.5 * img + 0.5 * lo,
        "текст + картинки + LoRA": (txt + img + lo) / 3,
        "0.4 текст + 0.2 картинки + 0.4 LoRA": 0.4 * txt + 0.2 * img + 0.4 * lo,
        "0.45 текст + 0.1 картинки + 0.45 LoRA": 0.45 * txt + 0.1 * img + 0.45 * lo,
    }.items():
        f = nested(o, y, fold)
        summary[(cat, tag)] = f
        print(f"  {tag:40s} F1={f:.4f} AUC={roc_auc_score(y, o):.4f} "
              f"PR={average_precision_score(y, o):.4f}", flush=True)

print("\n" + "=" * 96)
print("МЕТРИКА СОРЕВНОВАНИЯ (вложенно)")
for tag in sorted({t for (_, t) in summary}):
    vals = [summary[(c, tag)] for c in ["Легковоспламеняющиеся", "БАД"]]
    print(f"  {tag:42s} {np.mean(vals):.4f}   (флам {vals[0]:.4f}, БАД {vals[1]:.4f})")
print("\n  цель: 0.915 | текущий сабмит: вложенно 0.9006 -> LB 0.87820")
