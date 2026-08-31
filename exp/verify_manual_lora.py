"""Проверка: ручной мерж LoRA даёт то же самое, что peft.

Сравниваем скоры на одних и тех же карточках:
  A) peft.PeftModel поверх базовой модели  (эталон, но в образе организаторов недоступен)
  B) ручной мерж W += (alpha/r) * B @ A     (то, что реально поедет в сабмит)
"""
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "submit4")
from src.lora import merge_lora_  # noqa: E402
from src.prompts import build_messages  # noqa: E402

BASE = "/tmp/shared_models/Qwen/Qwen3.5-4B"
ADAPTER = ROOT + "submit4/lora_adapter"
N = 200

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"]).sample(N, random_state=3)
tok = AutoTokenizer.from_pretrained(ADAPTER, padding_side="left")
yes_id = tok.encode("Да", add_special_tokens=False)[0]
no_id = tok.encode("Нет", add_special_tokens=False)[0]

prompts = []
for cat, n, d in zip(df["category"], df["name"].fillna(""), df["description"].fillna("")):
    msgs = build_messages(cat, n, d)
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompts.append(p)


@torch.no_grad()
def score(model):
    out = np.zeros(len(prompts), dtype=np.float32)
    for i in range(0, len(prompts), 16):
        enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True,
                  truncation=True, max_length=1280).to(model.device)
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        out[i:i + 16] = torch.sigmoid(lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
    return out


print("=== A) эталон: peft ===", flush=True)
from peft import PeftModel  # noqa: E402
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
m = PeftModel.from_pretrained(m, ADAPTER).eval()
ref = score(m)
del m
torch.cuda.empty_cache()

print("=== B) ручной мерж, без peft ===", flush=True)
m2 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
merged = merge_lora_(m2, ADAPTER)
print(f"вмержено модулей: {merged}", flush=True)
mine = score(m2)

d = np.abs(ref - mine)
print(f"\nmax|Δскор| = {d.max():.3e}   среднее = {d.mean():.3e}")
print(f"корреляция: {np.corrcoef(ref, mine)[0, 1]:.6f}")
# главное — не абсолютная разница (bf16 округляет), а совпадение решений
# на всём диапазоне порогов, где мы реально работаем
flips = {t: int((((ref >= t) != (mine >= t))).sum()) for t in
         [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
print("расхождений вердиктов по порогам:", flips)
ok = max(flips.values()) == 0 and np.corrcoef(ref, mine)[0, 1] > 0.9999
print("\nРЕЗУЛЬТАТ:", "OK — ручной мерж эквивалентен peft"
      if ok else "РАСХОЖДЕНИЕ!")
