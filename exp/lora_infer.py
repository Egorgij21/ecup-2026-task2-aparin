"""Батчевый скоринг карточек LLM-кой: P(«Да») против P(«Нет») на позиции ответа.

Поддерживает мультимодальный режим: если передан processor и n_img > 0, в модель
идут картинки вместе с текстом. Без этого обучение с картинками мерилось бы
ТЕКСТОВЫМ скорингом — то есть замер был бы фикцией."""
import numpy as np
import torch

import sys, os
sys.path.insert(0, "/workspace/counter/exp/deps")
from prompts import build_messages
from imgutil import load_card_images


def build_prompts(tok, df):
    out = []
    for cat, n, d in zip(df["category"], df["name"].fillna(""), df["description"].fillna("")):
        msgs = build_messages(cat, n, d)
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(p)
    return out


@torch.no_grad()
def _score_mm(model, proc, df, yes_id, no_id, maxlen, bs, n_img, img_dir, max_pixels=0, tile=1):
    """Скоринг С КАРТИНКАМИ. Обязателен, если модель обучалась мультимодально:
    иначе меряли бы текстом то, что училось на изображениях. tile>1 — тайлинг (как в
    обучении), через единый deps/imgutil.load_card_images."""
    proc.tokenizer.padding_side = "left"
    scores = np.zeros(len(df), dtype=np.float32)
    was_training = model.training
    model.eval()
    import inspect
    _p = inspect.signature(model.forward).parameters
    keep = {"logits_to_keep": 1} if "logits_to_keep" in _p else (
        {"num_logits_to_keep": 1} if "num_logits_to_keep" in _p else {})
    rows = df.reset_index(drop=True)
    for i in range(0, len(rows), bs):
        ch = rows.iloc[i:i + bs]
        texts, imgs_b = [], []
        for _, r in ch.iterrows():
            # тот же загрузчик и то же ограничение размера, что в обучении (иначе замер
            # шёл бы на другом разрешении). tile>1 — тайлинг.
            ims = load_card_images(img_dir, r["id"], n_img, max_pixels, tile)
            base = build_messages(r["category"], r["name"], r["description"])
            content = [{"type": "image"} for _ in ims] + \
                      [{"type": "text", "text": base[-1]["content"]}]
            texts.append(proc.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True))
            imgs_b.append(ims)
        # НЕ обрезаем: у мультимодальных моделей truncation рвёт соответствие
        # между image-токенами в тексте и input_ids, процессор падает с
        # «Mismatch in image token count». Лучше упасть на длинном промпте.
        kw = {"text": texts, "return_tensors": "pt", "padding": True}
        if any(imgs_b):
            kw["images"] = imgs_b
        enc = proc(**kw)
        enc = {k: (v.to(model.device).to(torch.bfloat16) if v.dtype.is_floating_point
                   else v.to(model.device)) for k, v in enc.items()}
        lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
        scores[i:i + len(ch)] = torch.sigmoid(lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
        for ims in imgs_b:
            for im in ims:
                im.close()
    if was_training:
        model.train()
    return scores


@torch.no_grad()
def score_df(model, tok, df, yes_id, no_id, maxlen=1280, bs=16,
             proc=None, n_img=0, img_dir=None, max_pixels=0, tile=1) -> np.ndarray:
    if proc is not None and n_img > 0:
        return _score_mm(model, proc, df, yes_id, no_id, maxlen, bs, n_img, img_dir,
                         max_pixels, tile)
    prompts = build_prompts(tok, df)
    scores = np.zeros(len(prompts), dtype=np.float32)
    was_training = model.training
    model.eval()
    # режем СЛЕВА: вопрос и generation-prompt стоят в конце, правая обрезка съела бы их
    # и последний токен (по которому берём логит) оказался бы внутри описания. Аудит 26.08.
    tok.truncation_side = "left"
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(model.device)
        import inspect
        _p = inspect.signature(model.forward).parameters
        kw = {"logits_to_keep": 1} if "logits_to_keep" in _p else (
            {"num_logits_to_keep": 1} if "num_logits_to_keep" in _p else {})
        logits = model(**enc, **kw).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        scores[i:i + bs] = torch.sigmoid(lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
    if was_training:
        model.train()
    return scores
