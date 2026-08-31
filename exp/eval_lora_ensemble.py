"""Даёт ли ансамбль LoRA-адаптеров прирост, и какой источник разнообразия сильнее.

Все три адаптера обучены на фолдах 1-4 и оцениваются на fold 0, который никто из них
не видел. Сравниваем два источника разнообразия:
  * другой сид на той же базовой модели (Qwen, сид 777) — дёшево;
  * другое семейство моделей (gemma-4-E4B) — дороже, но ошибки должны быть
    декоррелированы сильнее.

Смотрим не только F1, но и корреляцию между адаптерами: именно она определяет,
будет ли от ансамбля толк.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = "/workspace/counter/"
THS = np.linspace(0.01, 0.99, 197)

SOURCES = {
    "Qwen (сид 0)": "exp/lora_oof_fold0.parquet",
    "Qwen (сид 777)": "exp/lora_oof_fold0_s777.parquet",
    "gemma-4-E4B": "exp/lora_oof_fold0_gemma.parquet",
}

base = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])[["id", "category", "label"]]
cols = {}
for tag, path in SOURCES.items():
    full = ROOT + path
    if not os.path.exists(full):
        print(f"нет: {tag} ({path})")
        continue
    d = pd.read_parquet(full)[["id", "lora_score"]].rename(columns={"lora_score": tag})
    base = base.merge(d, on="id", how="inner")
    cols[tag] = tag
print(f"адаптеров в наличии: {list(cols)}; строк fold0: {len(base)}\n")
if len(cols) < 2:
    sys.exit("нужно минимум два адаптера для сравнения")


def best_f1(y, s):
    return max(f1_score(y, (s >= t).astype(int)) for t in THS)


for cat, g in base.groupby("category"):
    y = g["label"].values
    print(f"########## {cat}: n={len(g)} pos={y.sum()}")
    print(f"  {'адаптер':18s} {'F1':>7s} {'AUC':>7s} {'PR':>7s}")
    for tag in cols:
        s = g[tag].values
        print(f"  {tag:18s} {best_f1(y, s):7.4f} {roc_auc_score(y, s):7.4f} "
              f"{average_precision_score(y, s):7.4f}")

    names = list(cols)
    print("  корреляция скоров между адаптерами:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = np.corrcoef(g[names[i]].values, g[names[j]].values)[0, 1]
            print(f"     {names[i]} x {names[j]}: {c:.3f}")

    print("  --- ансамбли (среднее скоров) ---")
    import itertools
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            s = np.mean([g[t].values for t in combo], axis=0)
            print(f"     {' + '.join(combo):44s} F1={best_f1(y, s):.4f} "
                  f"AUC={roc_auc_score(y, s):.4f} PR={average_precision_score(y, s):.4f}")
    print()
