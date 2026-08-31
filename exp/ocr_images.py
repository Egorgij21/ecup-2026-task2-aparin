"""OCR упаковки через Qwen3-VL-2B-Instruct.

Гипотеза из разбора ошибок LoRA: пропуски — это целые незнакомые семьи товаров
(«Цветной дым Maxsem», «Брикеты Weber», «мапп газ»), у которых на упаковке видно
«класс опасности», «18+», «огнеопасно», а в тексте карточки — ничего.
Плюс правила БАД прямо говорят: маркировка может быть «в описании ИЛИ НА ИЗОБРАЖЕНИИ».

Забираем с картинки именно ТЕКСТ, а не описание сцены: текст втыкается в уже
работающий пайплайн (промпт LoRA, TF-IDF, правила) без переделки архитектуры.

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/ocr_images.py <категория|all> [n_images] [limit]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

Image.MAX_IMAGE_PIXELS = None
ROOT = "/workspace/counter/"
IMG = ROOT + "data/images/"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"

CAT = sys.argv[1] if len(sys.argv) > 1 else "all"
N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BS = 16
MAX_NEW = 96
MAX_PIXELS = 401408          # пресет "L" из бейзлайна: мелкий шрифт на упаковке важен

PROMPT = ("Выпиши весь текст, который видно на этом изображении товара: надписи на "
          "упаковке и этикетке, маркировки, предупреждения, знаки опасности, состав. "
          "Только сам текст, без пояснений. Если текста нет, ответь: НЕТ ТЕКСТА.")


def load_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * s) // 28 * 28), max(28, int(h * s) // 28 * 28)),
                       Image.LANCZOS)
    return im


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    if CAT != "all":
        df = df[df["category"] == CAT].reset_index(drop=True)
    if LIMIT:
        df = df.sample(min(LIMIT, len(df)), random_state=0).reset_index(drop=True)

    jobs = []                      # (row_idx, slot, path)
    for i, pid in enumerate(df["id"]):
        d = os.path.join(IMG, str(pid))
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
        for slot, f in enumerate(files[:N_IMG]):
            jobs.append((i, slot, os.path.join(d, f)))
    print(f"товаров: {len(df)}, изображений к обработке: {len(jobs)}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    # при генерации в батче правый паддинг портит короткие последовательности
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    texts = {}
    t0 = time.time()
    for i in range(0, len(jobs), BS):
        batch = jobs[i:i + BS]
        images, prompts = [], []
        for _, _, path in batch:
            images.append(load_image(path))
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": PROMPT}]}]
            prompts.append(proc.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True))
        enc = proc(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 repetition_penalty=1.15, no_repeat_ngram_size=4)
        gen = out[:, enc["input_ids"].shape[1]:]
        dec = proc.batch_decode(gen, skip_special_tokens=True)
        for (ri, slot, _), txt in zip(batch, dec):
            texts.setdefault(ri, {})[slot] = txt.strip()
        for im in images:
            im.close()
        if i % (BS * 20) == 0:
            el = time.time() - t0
            done = i + len(batch)
            print(f"  {done}/{len(jobs)}  {el:.0f}с  ~{el/max(done,1)*len(jobs):.0f}с всего",
                  flush=True)

    out_rows = []
    for i, pid in enumerate(df["id"]):
        parts = texts.get(i, {})
        joined = " | ".join(parts[k] for k in sorted(parts))
        out_rows.append(joined)
    res = df[["id", "category", "label"]].copy()
    res["ocr"] = out_rows
    tag = "" if CAT == "all" else ("_flam" if CAT.startswith("Легк") else "_bad")
    tag += f"_n{LIMIT}" if LIMIT else ""
    path = ROOT + f"exp/ocr{tag}.parquet"
    res.to_parquet(path)
    print(f"\nготово за {time.time()-t0:.0f}с -> {path}", flush=True)
    nonempty = res["ocr"].str.len() > 0
    print(f"непустой OCR: {nonempty.sum()}/{len(res)}")
    print(f"средняя длина: {res.loc[nonempty, 'ocr'].str.len().mean():.0f} символов")


if __name__ == "__main__":
    main()
