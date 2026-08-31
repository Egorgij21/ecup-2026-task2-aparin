"""Скоринг произвольного среза данных обученным LoRA-адаптером.

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/lora_score_all.py <adapter_dir> <out.parquet> [fold]
  fold: если указан — скорим только этот фолд (честная оценка: адаптер его не видел),
        иначе весь датасет.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
sys.path.insert(0, ROOT + "submit2")
from folds import make_folds  # noqa: E402
from lora_infer import score_df  # noqa: E402

ADAPTER = sys.argv[1]
OUT = sys.argv[2]
FOLD = int(sys.argv[3]) if len(sys.argv) > 3 else None
BASE = os.environ.get("QWEN_PATH", "Qwen/Qwen3.5-4B")

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["fold"] = make_folds(df)
if FOLD is not None:
    df = df[df["fold"] == FOLD].reset_index(drop=True)
print(f"скорим {len(df)} строк, fold={FOLD}", flush=True)

tok = AutoTokenizer.from_pretrained(ADAPTER, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, ADAPTER).eval()
yes_id = tok.encode("Да", add_special_tokens=False)[0]
no_id = tok.encode("Нет", add_special_tokens=False)[0]

t0 = time.time()
scores = score_df(model, tok, df, yes_id, no_id, maxlen=1280, bs=32)
out = df[["id", "category", "label", "fold"]].copy()
out["lora_score"] = scores
out.to_parquet(OUT)
print(f"готово за {time.time()-t0:.0f}с -> {OUT}", flush=True)

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402

zs = pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]]
out = out.merge(zs, on="id", how="left")
for cat, g in out.groupby("category"):
    y = g["label"].values
    ths = np.linspace(0.02, 0.98, 193)
    print(f"\n### {cat}: n={len(g)} pos={y.sum()}")
    for tag, col in [("zero-shot", "llm_score"), ("LoRA", "lora_score")]:
        s = g[col].values
        fb = max((f1_score(y, (s >= t).astype(int)), t) for t in ths)
        print(f"  {tag:10s} AUC={roc_auc_score(y, s):.4f} PR={average_precision_score(y, s):.4f} "
              f"F1bin={fb[0]:.4f}@{fb[1]:.2f}")
