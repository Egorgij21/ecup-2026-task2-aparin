"""Скоринг произвольного чекпоинта LoRA на валидационном фолде.

Нужен потому, что lora_train.py считает OOF только для ФИНАЛЬНОЙ модели, а прогон
combo идёт 3 эпохи с чекпоинтами по эпохам. Без этого скрипта промежуточные эпохи
пришлось бы выбрасывать, а именно они отвечают на вопрос «а третья эпоха вообще
нужна?» — переход 1->2 в своё время дал +4.5 пункта, и где плато, мы не знаем.

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/score_checkpoint.py <adapter_dir> <fold> [tag]
Пишет exp/lora_oof_fold<fold><tag>.parquet — тот же формат, что у lora_train.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from folds import make_folds  # noqa: E402
from lora_infer import score_df  # noqa: E402

ADAPTER = sys.argv[1]
FOLD = int(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else "_" + os.path.basename(ADAPTER.rstrip("/"))

cfg = json.load(open(os.path.join(ADAPTER, "adapter_config.json")))
BASE = cfg["base_model_name_or_path"]
print(f"адаптер={ADAPTER}\nбаза={BASE} r={cfg['r']} alpha={cfg['lora_alpha']}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
yes_id = tok.encode("Да", add_special_tokens=False)[0]
no_id = tok.encode("Нет", add_special_tokens=False)[0]

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df["fold"] = make_folds(df)
va = df[df["fold"] == FOLD].reset_index(drop=True)
print(f"валидационный фолд {FOLD}: {len(va)} строк", flush=True)

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, ADAPTER).eval()

s = score_df(model, tok, va, yes_id, no_id, 1280)
out = va[["id", "category", "label"]].copy()
out["lora_score"] = s
path = ROOT + f"exp/lora_oof_fold{FOLD}{TAG}.parquet"
out.to_parquet(path)

for cat, g in out.groupby("category"):
    y, sc = g["label"].values, g["lora_score"].values
    ths = np.linspace(0.02, 0.98, 193)
    fb = max((f1_score(y, (sc >= t).astype(int)), t) for t in ths)
    print(f"{cat}: n={len(g)} pos={y.sum()} AUC={roc_auc_score(y, sc):.4f} "
          f"PR={average_precision_score(y, sc):.4f} F1={fb[0]:.4f}@{fb[1]:.2f}", flush=True)
print(f"-> {path}")
