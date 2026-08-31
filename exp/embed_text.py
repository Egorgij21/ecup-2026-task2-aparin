"""Эмбеддинги текста небольшим энкодером -> LogReg на той же group-CV.
Запуск: CUDA_VISIBLE_DEVICES=6 python exp/embed_text.py <model_id> <out_name> [max_len]
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = "/workspace/counter/"
MODEL = sys.argv[1]
OUT = ROOT + f"exp/emb_{sys.argv[2]}.npy"
MAXLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 512
PREFIX = os.environ.get("E5_PREFIX", "")  # "query: " для e5

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(s):
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip()


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
texts = [
    f"{PREFIX}Категория: {c}. Товар: {clean(n)}. Описание: {clean(d)}"
    for c, n, d in zip(df["category"], df["name"].fillna(""), df["description"].fillna(""))
]
print(f"{MODEL}: {len(texts)} текстов, max_len={MAXLEN}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

BS = 64
embs = []
with torch.no_grad():
    for i in range(0, len(texts), BS):
        batch = texts[i:i + BS]
        enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        embs.append(pooled.cpu().numpy())
        if i % (BS * 40) == 0:
            print(f"  {i}/{len(texts)}", flush=True)

E = np.vstack(embs).astype(np.float32)
np.save(OUT, E)
print("saved", OUT, E.shape, flush=True)
