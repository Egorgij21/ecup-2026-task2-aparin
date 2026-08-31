"""Хватает ли одной эпохи? Бесплатный эксперимент на уже сохранённых чекпоинтах.

Когда чинилось логирование, заодно добавились чекпоинты после каждой эпохи.
Поэтому для фолдов 1-4 есть модель после 1-й эпохи, и вопрос решается скорингом,
без переобучения. Fold 0 обучался до этой правки, поэтому он здесь не участвует.

Сравнение честное: адаптер fold k не видел строк фолда k ни на одной эпохе.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
sys.path.insert(0, ROOT + "submit4")
from folds import make_folds  # noqa: E402
from lora_infer import score_df  # noqa: E402
from src.lora import merge_lora_  # noqa: E402

BASE = "/tmp/shared_models/Qwen/Qwen3.5-4B"
FOLDS = [1, 2, 3, 4]

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["fold"] = make_folds(df)

tok = AutoTokenizer.from_pretrained(ROOT + "exp/lora_fold1", padding_side="left")
yes_id = tok.encode("Да", add_special_tokens=False)[0]
no_id = tok.encode("Нет", add_special_tokens=False)[0]

rows = []
t0 = time.time()
for k in FOLDS:
    adapter = ROOT + f"exp/lora_fold{k}_ep0"
    if not os.path.isdir(adapter):
        print(f"нет чекпоинта {adapter}, пропускаю")
        continue
    sub = df[df["fold"] == k].reset_index(drop=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    merge_lora_(model, adapter)
    s = score_df(model, tok, sub, yes_id, no_id, maxlen=1280, bs=8)
    out = sub[["id", "category", "label", "fold"]].copy()
    out["lora_ep0"] = s
    rows.append(out)
    print(f"  фолд {k} отскорен ({len(sub)} строк, {time.time()-t0:.0f}с)", flush=True)
    del model
    torch.cuda.empty_cache()

ep0 = pd.concat(rows)
ep0.to_parquet(ROOT + "exp/lora_oof_ep0.parquet")
print(f"\nсохранено, строк {len(ep0)}", flush=True)

# сравниваем с финальными (2 эпохи) на ТЕХ ЖЕ строках
import glob  # noqa: E402
ep1 = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                 for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[1-4].parquet"))])
m = ep0.merge(ep1, on="id", how="inner")
print(f"строк для сравнения: {len(m)}")

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402
THS = np.linspace(0.01, 0.99, 197)
for cat, g in m.groupby("category"):
    y = g["label"].values
    print(f"\n### {cat}: n={len(g)} pos={y.sum()}")
    for tag, col in [("1 эпоха", "lora_ep0"), ("2 эпохи", "lora_score")]:
        s = g[col].values
        fb = max((f1_score(y, (s >= t).astype(int)), t) for t in THS)
        print(f"  {tag:9s} F1@0.5={f1_score(y, (s >= 0.5).astype(int)):.4f}  "
              f"лучший F1={fb[0]:.4f}@{fb[1]:.2f}  AUC={roc_auc_score(y, s):.4f}  "
              f"PR={average_precision_score(y, s):.4f}")
    s = 0.5 * g["lora_ep0"].values + 0.5 * g["lora_score"].values
    print(f"  {'среднее':9s} F1@0.5={f1_score(y, (s >= 0.5).astype(int)):.4f}  "
          f"AUC={roc_auc_score(y, s):.4f}  PR={average_precision_score(y, s):.4f}")
    print(f"  корреляция 1эп x 2эп: "
          f"{np.corrcoef(g['lora_ep0'].values, g['lora_score'].values)[0, 1]:.3f}")
