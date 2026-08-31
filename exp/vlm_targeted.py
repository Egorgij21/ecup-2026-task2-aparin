"""Прицельный промпт для VLM: не «примени правила», а «прочитай упаковку».

Почему общий промпт провалился (прирост AUC −0.0005). Я подавал модели те же
опубликованные правила, что и в текстовом режиме, и просто добавлял картинку.
Модель не знала, ЧТО искать на изображении, и картинка работала как шум.

Почему это должно сработать на хлопушках — гипотеза проверена на данных:
  негативы называются «Хлопушка ПНЕВМАТИЧЕСКАЯ ...» — 79 из 142 прямо в тексте;
  позитивы это просто «Хлопушка для праздника» — 1 из 28 упоминает пневматику.
То есть у негативов признак в тексте есть, а у позитивов его нет. Отличить
неуточнённую хлопушку можно по УПАКОВКЕ: у пиротехники обязательна маркировка
(«пиротехническое изделие», класс опасности, ТР ТС 006/2011, возраст 12+/16+).

Замер идёт на смешанных по меткам группах, где различие заведомо существует:
хлопушки, свечи для торта, бенгальские огни, цветной дым. Сравниваются три режима
на ОДНИХ И ТЕХ ЖЕ строках: только текст, картинка с общим промптом, картинка
с прицельным промптом.

Критерий задан ЗАРАНЕЕ: прицельный промпт полезен, если на хлопушках его AUC
выше текстового не меньше чем на 0.10. Планка выше обычной, потому что здесь
мы точно знаем, что различие есть и оно визуальное, — слабый эффект означал бы,
что модель маркировку не читает.

Запуск: CUDA_VISIBLE_DEVICES=N python exp/vlm_targeted.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "submit2")
from src.prompts import RULES_FLAM, clean  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MODEL = os.environ.get("VLM", "google/gemma-4-E4B-it")
N_IMG = int(os.environ.get("N_IMG", "2"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "589824"))   # крупнее обычного: читаем МЕЛКИЙ ТЕКСТ
BS = int(os.environ.get("VLM_BS", "2"))

GROUPS = {
    "хлопушки": r"хлопушк",
    "свечи для торта": r"свеч.{0,14}(для торта|в торт|цифр)|торт.{0,8}свеч",
    "бенгальские огни": r"бенгальск",
    "цветной дым": r"цветной дым|дым.{0,4} шашк|дымов.{0,4} шашк",
}

TARGETED = """Ты проверяешь карточку товара маркетплейса. Тебе даны фотографии упаковки.

ГЛАВНОЕ: внимательно ПРОЧИТАЙ ТЕКСТ И ЗНАКИ НА УПАКОВКЕ на фотографиях.

Признаки того, что товар СОДЕРЖИТ ПИРОТЕХНИЧЕСКИЙ СОСТАВ (ответ «Да»):
* надписи «пиротехническое изделие», «пиротехника», «содержит пиротехнический состав»;
* маркировка ТР ТС 006/2011, «класс опасности», «I класс», «бытовая пиротехника»;
* возрастное ограничение 12+, 16+, 18+, значок «детям запрещено»;
* предупреждения «не направлять на людей», «поджигать фитиль», значок огня/пламени;
* у свечей — надписи «фонтан», «искры», «горит», фитиль на фото.

Признаки того, что пиротехнического состава НЕТ (ответ «Нет»):
* надписи «пневматическая», «сжатый воздух», «без пиротехники», «безопасно для детей»;
* механизм с верёвочкой/поршнем без фитиля;
* товар — сувенир, декор, электронная или светодиодная имитация.

Если на фотографиях таких надписей не видно, опирайся на тип товара:
хлопушка без пометки «пневматическая» обычно содержит пиротехнический состав."""

QUESTION = "Содержит ли этот товар пиротехнический состав или иное горючее вещество?"


def load_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * s) // 28 * 28), max(28, int(h * s) // 28 * 28)),
                       Image.LANCZOS)
    return im


def imgs_for(pid):
    d = os.path.join(ROOT, "data/images", str(pid))
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in
            sorted(os.listdir(d))[:N_IMG] if f.lower().endswith(".jpg")]


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    d = df[df["category"] == "Легковоспламеняющиеся"].reset_index(drop=True)
    d["n"] = d["name"].fillna("").str.lower().str.replace("ё", "е")
    d["grp"] = ""
    for g, rx in GROUPS.items():
        d.loc[d["n"].str.contains(rx, regex=True) & (d["grp"] == ""), "grp"] = g
    sel = d[d["grp"] != ""].reset_index(drop=True)
    sel["paths"] = [imgs_for(p) for p in sel["id"]]
    sel = sel[sel["paths"].map(len) > 0].reset_index(drop=True)
    print("группы:", {g: (int((sel.grp == g).sum()), int(sel.label[sel.grp == g].sum()))
                      for g in GROUPS}, flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    def ids_for(ws):
        out = set()
        for w in ws:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t:
                    out.add(t[0])
        return sorted(out)

    yes_ids, no_ids = ids_for(["Да", "да"]), ids_for(["Нет", "нет"])
    import inspect
    _f = inspect.signature(model.forward).parameters
    keep = ({"logits_to_keep": 1} if "logits_to_keep" in _f
            else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _f else {})
    if not keep:
        raise RuntimeError("нет logits_to_keep — полные логиты дадут OOM")

    def run(mode):
        out = np.full(len(sel), np.nan, dtype=np.float32)
        t0 = time.time()
        for i in range(0, len(sel), BS):
            ch = sel.iloc[i:i + BS]
            texts, imgs_b = [], []
            for _, r in ch.iterrows():
                card = (f"Название товара: {clean(r['name'])}\n\n"
                        f"Описание товара: {clean(r['description'])[:900]}")
                content, ims = [], []
                if mode != "text":
                    for p in r["paths"]:
                        content.append({"type": "image"})
                        ims.append(load_image(p))
                head = TARGETED if mode == "targeted" else RULES_FLAM
                q = QUESTION if mode == "targeted" else \
                    "Является ли этот товар легковоспламеняющимся по правилам выше?"
                content.append({"type": "text",
                                "text": f"{head}\n\n---\n\n{card}\n\n---\n\n"
                                        f"{q} Ответь одним словом: Да или Нет."})
                texts.append(proc.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True))
                imgs_b.append(ims)
            kw = {"text": texts, "return_tensors": "pt", "padding": True}
            if mode != "text":
                kw["images"] = imgs_b
            enc = proc(**kw).to("cuda")
            with torch.no_grad():
                lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
            out[i:i + len(ch)] = torch.sigmoid(
                torch.logsumexp(lp[:, yes_ids], -1) - torch.logsumexp(lp[:, no_ids], -1)).cpu().numpy()
            for ims in imgs_b:
                for im in ims:
                    im.close()
            if i % (BS * 30) == 0:
                el = time.time() - t0
                print(f"   {mode} {i}/{len(sel)} {el:.0f}с ~{el/max(i+BS,1)*len(sel):.0f}с", flush=True)
        return out

    for mode in ["text", "generic", "targeted"]:
        sel[mode] = run(mode)
    sel.drop(columns=["paths"]).to_parquet(ROOT + "exp/vlm_targeted.parquet")

    print(f"\n   {'группа':20s} {'n':>4s} {'поз':>4s} {'текст':>8s} {'+карт общий':>12s} "
          f"{'+карт ПРИЦЕЛЬНЫЙ':>17s}")
    for g in list(GROUPS) + ["ВСЕ"]:
        m = np.ones(len(sel), bool) if g == "ВСЕ" else (sel["grp"] == g).values
        y = sel["label"].values[m]
        if not (0 < y.sum() < m.sum()):
            print(f"   {g:20s} один класс")
            continue
        a = [roc_auc_score(y, sel[k].values[m]) for k in ["text", "generic", "targeted"]]
        print(f"   {g:20s} {int(m.sum()):4d} {int(y.sum()):4d} {a[0]:8.4f} {a[1]:12.4f} {a[2]:17.4f}")
    m = (sel["grp"] == "хлопушки").values
    y = sel["label"].values[m]
    dd = roc_auc_score(y, sel["targeted"].values[m]) - roc_auc_score(y, sel["text"].values[m])
    print(f"\n   прирост на ХЛОПУШКАХ: {dd:+.4f} (критерий заранее: нужно >= +0.10)")
    print(f"   вердикт: {'ЧИТАЕТ УПАКОВКУ' if dd >= 0.10 else 'не читает / не помогает'}")


if __name__ == "__main__":
    main()
