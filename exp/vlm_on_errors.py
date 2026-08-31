"""Различает ли ИЗОБРАЖЕНИЕ те строки, где текст ошибается?

Почему это вообще вопрос. Обе наши базы мультимодальные (Qwen3.5-4B —
Qwen3_5ForConditionalGeneration с vision_config, gemma-4-E4B — с vision и audio),
но мы всё время подавали им ТОЛЬКО текст. 44124 изображения не участвовали никогда.

Хуже того: официальное правило БАД гласит «в описании ИЛИ НА ИЗОБРАЖЕНИИ содержится
прямое указание». То есть мы просим модель применить критерий, который прямо
ссылается на упаковку, не показывая упаковку.

Замер узкий и дешёвый: берём строки, где текстовый ансамбль ОШИБСЯ, и столько же
случайных, где он прав. Скорим их одной моделью дважды — только текст и текст+картинка.
Смотрим, растёт ли AUC именно на ошибочных.

Критерий задан ЗАРАНЕЕ: картинки полезны, если на ошибочных строках AUC версии
с картинкой выше текстовой не меньше чем на 0.05. Меньше — в пределах шума
на выборке в несколько сотен строк.

Запуск: CUDA_VISIBLE_DEVICES=N python exp/vlm_on_errors.py [БАД|Легковоспламеняющиеся]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "submit2")
from src.prompts import RULES_BAD, RULES_FLAM, clean  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CAT = sys.argv[1] if len(sys.argv) > 1 else "БАД"
MODEL = os.environ.get("VLM", "google/gemma-4-E4B-it")
N_IMG = int(os.environ.get("N_IMG", "1"))
MAX_PIXELS = 261120          # пресет «M» из бейзлайна, ~83 vision-токена на картинку
BS = int(os.environ.get("VLM_BS", "4"))
QUESTION = {"Легковоспламеняющиеся": "Является ли этот товар легковоспламеняющимся?",
            "БАД": "Является ли этот товар биологически активной добавкой?"}


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
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
    return [os.path.join(d, f) for f in fs[:N_IMG]]


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    oof = pd.read_parquet(ROOT + "exp/text_oof.parquet")
    d = df.merge(oof[["id", "text_score", "err"]], on="id")
    d = d[d["category"] == CAT].reset_index(drop=True)
    err = d[d["err"] != "верно"]
    rng = np.random.default_rng(0)
    ok = d[d["err"] == "верно"].iloc[rng.choice((d["err"] == "верно").sum(),
                                                size=min(len(err), int((d["err"] == "верно").sum())),
                                                replace=False)]
    sel = pd.concat([err, ok]).reset_index(drop=True)
    sel["is_err"] = (sel["err"] != "верно").values
    sel["paths"] = [imgs_for(p) for p in sel["id"]]
    has = sel["paths"].map(len) > 0
    sel = sel[has].reset_index(drop=True)
    print(f"{CAT}: ошибочных {int(sel['is_err'].sum())}, контроль {int((~sel['is_err']).sum())}, "
          f"всего с картинками {len(sel)}", flush=True)

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

    rules = RULES_FLAM if CAT == "Легковоспламеняющиеся" else RULES_BAD

    def score(with_image):
        out = np.full(len(sel), np.nan, dtype=np.float32)
        t0 = time.time()
        for i in range(0, len(sel), BS):
            chunk = sel.iloc[i:i + BS]
            msgs_b, imgs_b = [], []
            for _, r in chunk.iterrows():
                card = (f"Название товара: {clean(r['name'])}\n\n"
                        f"Описание товара: {clean(r['description'])[:1100]}")
                content = []
                ims = []
                if with_image:
                    for p in r["paths"]:
                        content.append({"type": "image"})
                        ims.append(load_image(p))
                content.append({"type": "text",
                                "text": f"{rules}\n\n---\n\n{card}\n\n---\n\n"
                                        f"{QUESTION[CAT]} Ответь одним словом: Да или Нет."})
                msgs_b.append([{"role": "user", "content": content}])
                imgs_b.append(ims)
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in msgs_b]
            kw = {"text": texts, "return_tensors": "pt", "padding": True}
            if with_image:
                kw["images"] = imgs_b
            enc = proc(**kw).to("cuda")
            with torch.no_grad():
                lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
            y_ = torch.logsumexp(lp[:, yes_ids], dim=-1)
            n_ = torch.logsumexp(lp[:, no_ids], dim=-1)
            out[i:i + len(chunk)] = torch.sigmoid(y_ - n_).cpu().numpy()
            for ims in imgs_b:
                for im in ims:
                    im.close()
            if i % (BS * 25) == 0:
                el = time.time() - t0
                print(f"   {'с картинкой' if with_image else 'только текст'} {i}/{len(sel)} "
                      f"{el:.0f}с ~{el/max(i+BS,1)*len(sel):.0f}с", flush=True)
        return out

    sel["vlm_text"] = score(False)
    sel["vlm_img"] = score(True)
    sel.drop(columns=["paths"]).to_parquet(ROOT + f"exp/vlm_errors_{'bad' if CAT=='БАД' else 'flam'}.parquet")

    print(f"\n   {'срез':26s} {'n':>5s} {'поз':>5s} {'AUC текст':>10s} {'AUC +картинка':>14s} "
          f"{'PR текст':>9s} {'PR +карт':>9s}")
    for tag, m in [("ОШИБКИ текста", sel["is_err"].values),
                   ("контроль (верные)", ~sel["is_err"].values),
                   ("всё вместе", np.ones(len(sel), bool))]:
        yy = sel["label"].values[m]
        if not (0 < yy.sum() < m.sum()):
            print(f"   {tag:26s} один класс")
            continue
        a0 = roc_auc_score(yy, sel["vlm_text"].values[m])
        a1 = roc_auc_score(yy, sel["vlm_img"].values[m])
        p0 = average_precision_score(yy, sel["vlm_text"].values[m])
        p1 = average_precision_score(yy, sel["vlm_img"].values[m])
        print(f"   {tag:26s} {int(m.sum()):5d} {int(yy.sum()):5d} {a0:10.4f} {a1:14.4f} "
              f"{p0:9.4f} {p1:9.4f}")
    m = sel["is_err"].values
    yy = sel["label"].values[m]
    if 0 < yy.sum() < m.sum():
        dd = roc_auc_score(yy, sel["vlm_img"].values[m]) - roc_auc_score(yy, sel["vlm_text"].values[m])
        print(f"\n   прирост AUC на ОШИБКАХ: {dd:+.4f}  (критерий заранее: нужно >= +0.05)")
        print(f"   вердикт: {'КАРТИНКИ ПОЛЕЗНЫ' if dd >= 0.05 else 'в пределах шума'}")


if __name__ == "__main__":
    main()
