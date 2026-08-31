"""Дообучение маленького текстового энкодера как компонента бленда.

Два вопроса разом, оба дёшево (минуты на фолд вместо 2.7 часов у LoRA на 4B):

1. Нужен ли ещё TF-IDF? Сейчас в бленде он даёт всего 0.6716 на флам, и держится
   не качеством, а декоррелированностью с LoRA (корреляция 0.732). Дообученный
   энкодер может быть сильнее — но может оказаться и коррелированнее, тогда
   выигрыша в бленде не будет. Проверяем.

2. Помогает ли OCR ОБУЧАЕМОЙ модели? Прошлая проверка была на TF-IDF, и это
   слабый аргумент про LoRA. Здесь модель обучается, поэтому ответ переносится
   на LoRA куда честнее — и стоит минуты, а не 13 часов.

Фолды — те же семейные. Запуск:
  CUDA_VISIBLE_DEVICES=6 python exp/train_encoder.py <with_ocr|no_ocr> [model]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "no_ocr"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "intfloat/multilingual-e5-base"
MAXLEN, BS, EPOCHS, LR = 512, 32, 3, 2e-5


class Cards(Dataset):
    def __init__(self, texts, labels, weights, tok):
        self.texts, self.labels, self.weights, self.tok = texts, labels, weights, tok

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.labels[i]), float(self.weights[i])


def collate(batch, tok):
    texts, labels, w = zip(*batch)
    enc = tok(list(texts), padding=True, truncation=True, max_length=MAXLEN,
              return_tensors="pt")
    return enc, torch.tensor(labels), torch.tensor(w, dtype=torch.float32)


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    ocr = pd.read_parquet(ROOT + "exp/ocr.parquet")[["id", "ocr"]]
    df = df.merge(ocr, on="id", how="left")
    df["fold"] = make_folds(df)
    df["ocr"] = df["ocr"].fillna("").map(clean)

    base = ("query: Категория: " + df["category"].astype(str)
            + ". Товар: " + df["name"].fillna("").map(clean)
            + ". Описание: " + df["description"].fillna("").map(clean))
    df["text"] = base + (" [упаковка] " + df["ocr"] if MODE == "with_ocr" else "")
    print(f"режим={MODE} модель={MODEL} длина текста медиана="
          f"{df['text'].str.len().median():.0f}", flush=True)

    # вес примера обратно пропорционален частоте класса ВНУТРИ категории
    w = np.ones(len(df), dtype=np.float32)
    for cat, g in df.groupby("category"):
        p = g["label"].mean()
        w[g.index] = np.where(g["label"] == 1, 0.5 / p, 0.5 / (1 - p))

    tok = AutoTokenizer.from_pretrained(MODEL)
    oof = np.zeros(len(df))
    t0 = time.time()

    for k in range(5):
        tr = np.where(df["fold"].values != k)[0]
        te = np.where(df["fold"].values == k)[0]
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=2).cuda()
        ds = Cards(df["text"].values[tr], df["label"].values[tr], w[tr], tok)
        dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate(b, tok))
        steps = len(dl) * EPOCHS
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda")

        model.train()
        for ep in range(EPOCHS):
            for enc, y, ww in dl:
                enc = {kk: v.cuda() for kk, v in enc.items()}
                y, ww = y.cuda(), ww.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits
                per = torch.nn.functional.cross_entropy(logits.float(), y, reduction="none")
                loss = (per * ww).sum() / ww.sum()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(te), 64):
                chunk = te[i:i + 64]
                enc = tok(list(df["text"].values[chunk]), padding=True, truncation=True,
                          max_length=MAXLEN, return_tensors="pt")
                enc = {kk: v.cuda() for kk, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(**enc).logits.float()
                probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        oof[te] = np.concatenate(probs)
        print(f"  фолд {k} готов, {time.time()-t0:.0f}с", flush=True)
        del model
        torch.cuda.empty_cache()

    out = df[["id", "category", "label"]].copy()
    out["enc_score"] = oof
    path = ROOT + f"exp/enc_{MODE}.parquet"
    out.to_parquet(path)
    print(f"\nготово за {time.time()-t0:.0f}с -> {path}", flush=True)

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    ths = np.linspace(0.01, 0.99, 197)
    for cat, g in out.groupby("category"):
        y, s = g["label"].values, g["enc_score"].values
        fb = max((f1_score(y, (s >= t).astype(int)), t) for t in ths)
        print(f"{cat}: AUC={roc_auc_score(y, s):.4f} PR={average_precision_score(y, s):.4f} "
              f"F1={fb[0]:.4f}@{fb[1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
