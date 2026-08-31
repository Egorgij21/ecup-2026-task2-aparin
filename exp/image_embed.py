"""Эмбеддинги изображений через визуальную башню Qwen3-VL-2B (407M параметров).

Зачем именно так. OCR провалился, потому что переводил картинку в ТЕКСТ и попадал
в то же пространство, что и карточка — информация оказалась избыточной с LoRA.
Здесь берутся патч-эмбеддинги ЧИСТО визуальной башни, без языковой части: форма
упаковки (аэрозольный баллон против коробки), пиктограммы, тип тары, цветовые
паттерны. Такого в тексте нет вовсе.

Критерий успеха задан заранее: корреляция с LoRA должна быть заметно ниже 0.73
(столько у TF-IDF, который в бленде работает). Иначе сигнал избыточен и бесполезен,
каким бы сильным он ни был сам по себе — это ровно то, что показал замер компонентов.

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/image_embed.py [n_images] [limit]
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

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BS = 8
MAX_PIXELS = 261120        # пресет "M" из бейзлайна


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
    if LIMIT:
        df = df.sample(min(LIMIT, len(df)), random_state=0).reset_index(drop=True)

    jobs = []
    for i, pid in enumerate(df["id"]):
        d = os.path.join(IMG, str(pid))
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
        for f in files[:N_IMG]:
            jobs.append((i, os.path.join(d, f)))
    print(f"товаров {len(df)}, изображений {len(jobs)} (до {N_IMG} на товар)", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    full = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
    visual = full.model.visual.cuda().eval()      # только визуальная башня
    del full.model.language_model, full.lm_head
    print(f"визуальная башня: {sum(p.numel() for p in visual.parameters())/1e6:.0f}M параметров",
          flush=True)

    dim = None
    acc = {}
    cnt = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(jobs), BS):
            batch = jobs[i:i + BS]
            images = [load_image(p) for _, p in batch]
            enc = proc.image_processor(images=images, return_tensors="pt")
            pv = enc["pixel_values"].to("cuda", dtype=torch.bfloat16)
            grid = enc["image_grid_thw"].to("cuda")
            out = visual(pv, grid_thw=grid)
            # башня может вернуть тензор, кортеж или объект с last_hidden_state —
            # разбираем все варианты до тензора патчей
            feats = out
            while not torch.is_tensor(feats):
                if hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state
                elif isinstance(feats, (tuple, list)):
                    feats = feats[0]
                else:
                    raise TypeError(f"не удалось достать тензор из {type(feats)}")
            if feats.dim() == 3 and feats.shape[0] == 1:
                feats = feats[0]
            # визуальная башня отдаёт все патчи подряд; режем по числу патчей на картинку
            merge = getattr(proc.image_processor, "merge_size", 2) ** 2
            sizes = (grid.prod(dim=-1) // merge).tolist()
            pos = 0
            for (ri, _), n in zip(batch, sizes):
                v = feats[pos:pos + n].float().mean(0).cpu().numpy()
                pos += n
                if dim is None:
                    dim = v.shape[0]
                acc[ri] = acc.get(ri, 0) + v
                cnt[ri] = cnt.get(ri, 0) + 1
            for im in images:
                im.close()
            if i % (BS * 60) == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}с  ~{el/max(i+BS,1)*len(jobs):.0f}с всего",
                      flush=True)

    E = np.zeros((len(df), dim), dtype=np.float32)
    for ri, v in acc.items():
        E[ri] = v / cnt[ri]
    n = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(n, 1e-9, None)            # L2-нормировка

    tag = f"_n{LIMIT}" if LIMIT else ""
    np.save(ROOT + f"exp/img_emb{tag}.npy", E)
    df[["id", "category", "label"]].to_parquet(ROOT + f"exp/img_emb_index{tag}.parquet")
    print(f"\nготово за {time.time()-t0:.0f}с -> exp/img_emb{tag}.npy {E.shape}", flush=True)
    print(f"товаров без картинок (нулевой вектор): "
          f"{sum(1 for i in range(len(df)) if i not in cnt)}", flush=True)


if __name__ == "__main__":
    main()
