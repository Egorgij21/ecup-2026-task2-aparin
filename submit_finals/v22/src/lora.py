"""Скоринг карточек LLM-ками из /shared_models — без peft и без молчаливых откатов.

Две модели в одном решении:
  * zero-shot для ансамбля БАД — строго Qwen3.5-4B. Коэффициенты ансамбля обучены
    именно на его скорах, подменять базу под ними нельзя.
  * LoRA для флам — база берётся ИЗ САМОГО АДАПТЕРА (base_model_name_or_path
    в adapter_config.json), поэтому можно менять базовую модель, не трогая код.

ПОЧЕМУ БЕЗ peft. В образе odsai/ecup26-quality-baseline:1.0 его нет: бейзлайн
организаторов использует только torch/transformers. Прошлые сабмиты падали на этом
импорте и тихо уходили в fallback — два разных решения дали одинаковый скор до
десятого знака. LoRA это W' = W + (alpha/r)*B@A, мержится вручную.

ПОЧЕМУ БЕЗ ОТКАТОВ. Молчаливая деградация опаснее падения: она выглядит рабочим
результатом и портит выводы. Здесь всё падает громко.
"""
import json
import os
import time
from typing import List, Optional

import numpy as np

SHARED_MODELS_DIR = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
ZERO_SHOT_MODEL = "Qwen/Qwen3.5-4B"
MAX_LEN = 1280
BATCH_SIZE = 32


def resolve_model(subpath: str) -> str:
    tried = []
    for d in (SHARED_MODELS_DIR, "/shared_models"):
        p = os.path.join(d, subpath)
        tried.append(p)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"модель {subpath!r} не найдена, проверены пути: {tried}")


def adapter_base_subpath(adapter_dir: str) -> str:
    """Базовая модель адаптера — из его собственного конфига."""
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"в {adapter_dir}/adapter_config.json нет base_model_name_or_path")
    return base


def _load_adapter_tensors(adapter_dir: str):
    st = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.isfile(bin_path):
        import torch
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"в {adapter_dir} нет ни adapter_model.safetensors, ни .bin")


def merge_lora_(model, adapter_dir: str) -> int:
    """Вмерживает LoRA в веса модели на месте. Возвращает число изменённых модулей."""
    import torch

    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    if cfg.get("use_dora") or cfg.get("fan_in_fan_out"):
        raise ValueError("поддержан только обычный LoRA (без dora/fan_in_fan_out)")

    pairs = {}
    for key, val in _load_adapter_tensors(adapter_dir).items():
        if ".lora_A." in key:
            base, side = key.split(".lora_A.")[0], "A"
        elif ".lora_B." in key:
            base, side = key.split(".lora_B.")[0], "B"
        else:
            continue
        pairs.setdefault(base.replace("base_model.model.", "", 1), {})[side] = val

    named = dict(model.named_modules())
    merged, missing = 0, []
    for name, ab in pairs.items():
        module = named.get(name)
        if module is None:
            missing.append(name)
            continue
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"у модуля {name} есть только одна из матриц LoRA")
        W = module.weight
        A = ab["A"].to(device=W.device, dtype=torch.float32)
        B = ab["B"].to(device=W.device, dtype=torch.float32)
        # копим в float32 и округляем один раз: при сложении в bf16 мелкая дельта теряется
        W.data = (W.data.float() + (B @ A) * scaling).to(W.dtype)
        merged += 1
    if missing:
        raise ValueError(f"{len(missing)} модулей адаптера не найдено в модели, "
                         f"например: {missing[:3]}")
    return merged


def score_cards(model_subpath: str, categories, names, descs, build_messages,
                idx, adapter_dir: Optional[str] = None) -> np.ndarray:
    """Скорит карточки с индексами idx. adapter_dir=None -> чистая модель (zero-shot)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model(model_subpath)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна — решение рассчитано на GPU")

    t0 = time.time()
    tok_src = adapter_dir if adapter_dir and os.path.isdir(adapter_dir) else path
    tok = AutoTokenizer.from_pretrained(tok_src, padding_side="left", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                 local_files_only=True).cuda().eval()
    label = os.path.basename(model_subpath) + (" + LoRA" if adapter_dir else " zero-shot")
    print(f"[llm] {label}: загружена за {time.time()-t0:.0f}с из {path}", flush=True)

    if adapter_dir:
        n = merge_lora_(model, adapter_dir)
        print(f"[llm] {label}: адаптер вмержен вручную в {n} модулей (без peft)", flush=True)

    yes_id = tok.encode("Да", add_special_tokens=False)[0]
    no_id = tok.encode("Нет", add_special_tokens=False)[0]

    prompts: List[str] = []
    for cat, nm, d in zip(categories, names, descs):
        msgs = build_messages(cat, nm, d)
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:      # старые версии transformers не знают enable_thinking
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    import inspect
    _fwd = inspect.signature(model.forward).parameters
    keep_last = ({"logits_to_keep": 1} if "logits_to_keep" in _fwd
                 else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _fwd else {})

    scores = np.full(len(prompts), np.nan, dtype=np.float32)
    order = np.asarray(idx, dtype=np.int64)
    if len(order):
        order = order[np.argsort([len(prompts[j]) for j in order])]
        with torch.no_grad():
            for i in range(0, len(order), BATCH_SIZE):
                chunk = order[i:i + BATCH_SIZE]
                enc = tok([prompts[j] for j in chunk], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LEN).to(model.device)
                # логиты только для последней позиции: полный тензор это
                # batch x seq x vocab, гигабайты впустую
                out = model(**enc, **keep_last)
                lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                scores[chunk] = torch.sigmoid(lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
        if np.isnan(scores[np.asarray(idx, dtype=np.int64)]).any():
            raise RuntimeError(f"{label}: часть карточек осталась без скора")
    print(f"[llm] {label}: {len(order)} карточек, {time.time()-t0:.0f}с", flush=True)

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def score_cards_mm(model_subpath: str, categories, names, descs, ids, build_messages,
                   idx, adapter_dir: str, images_dir, n_img: int = 1,
                   max_pixels: int = 261120) -> np.ndarray:
    """Скоринг С КАРТИНКАМИ. Отдельная функция, а не флаг в score_cards: там
    токенизатор, здесь процессор, и набор тензоров у моделей разный.

    Зачем вообще. Мультимодальный адаптер на БАД обошёл текстовый на ВСЕХ трёх
    посчитанных фолдах (+0.0060, +0.0007, +0.0071). Официальное правило БАД
    дословно ссылается на изображение: «в описании ИЛИ НА ИЗОБРАЖЕНИИ содержится
    прямое указание». На флам картинки, наоборот, вредят — там их не подаём.

    Тонкости, каждая проверена на своей поломке:
      * НЕ обрезаем по max_length: truncation рвёт соответствие между image-токенами
        в тексте и input_ids, процессор падает с «Mismatch in image token count»;
      * размер картинки ограничиваем ДО процессора — у Qwen-подобных моделей число
        визуальных токенов растёт с размером (до 16910 на карточку), у gemma
        фиксировано 280 и ограничение безвредно;
      * передаём ВСЕ ключи процессора: у gemma это image_position_ids,
        у Qwen image_grid_thw. Пропуск ключа ломает модель молча.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    Image.MAX_IMAGE_PIXELS = None
    path = resolve_model(model_subpath)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна — решение рассчитано на GPU")
    t0 = time.time()
    # процессор берётся из БАЗОВОЙ модели: в папке адаптера нет preprocessor_config.json,
    # да и обработку изображений LoRA не меняет — она правит только веса
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    try:
        proc.tokenizer.padding_side = "left"
    except AttributeError:
        pass
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16,
                                                        local_files_only=True).cuda().eval()
    label = os.path.basename(model_subpath) + " + LoRA(картинки)"
    print(f"[llm] {label}: загружена за {time.time()-t0:.0f}с", flush=True)
    n = merge_lora_(model, adapter_dir)
    print(f"[llm] {label}: адаптер вмержен вручную в {n} модулей (без peft)", flush=True)

    tok = proc.tokenizer
    yes_id = tok.encode("Да", add_special_tokens=False)[0]
    no_id = tok.encode("Нет", add_special_tokens=False)[0]
    import inspect
    _f = inspect.signature(model.forward).parameters
    keep = ({"logits_to_keep": 1} if "logits_to_keep" in _f
            else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _f else {})
    if not keep:
        raise RuntimeError("модель не поддерживает logits_to_keep — полные логиты дадут OOM")

    def load_imgs(pid):
        d = os.path.join(str(images_dir), str(pid))
        if not os.path.isdir(d):
            return []
        out = []
        for f in sorted(x for x in os.listdir(d) if x.lower().endswith(".jpg"))[:n_img]:
            try:
                im = Image.open(os.path.join(d, f)).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            if max_pixels and w * h > max_pixels:
                k = (max_pixels / (w * h)) ** 0.5
                im = im.resize((max(28, int(w * k) // 28 * 28),
                                max(28, int(h * k) // 28 * 28)), Image.LANCZOS)
            out.append(im)
        return out

    scores = np.full(len(categories), np.nan, dtype=np.float32)
    order = np.asarray(idx, dtype=np.int64)
    BS_MM = 4
    with torch.no_grad():
        for i in range(0, len(order), BS_MM):
            chunk = order[i:i + BS_MM]
            texts, imgs_b = [], []
            for j in chunk:
                ims = load_imgs(ids[j])
                base = build_messages(categories[j], names[j], descs[j])
                content = [{"type": "image"} for _ in ims] + \
                          [{"type": "text", "text": base[-1]["content"]}]
                texts.append(proc.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True))
                imgs_b.append(ims)
            kw = {"text": texts, "return_tensors": "pt", "padding": True}
            if any(imgs_b):
                kw["images"] = imgs_b
            enc = proc(**kw)
            enc = {k: (v.cuda().to(torch.bfloat16) if v.dtype.is_floating_point else v.cuda())
                   for k, v in enc.items()}
            lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
            scores[chunk] = torch.sigmoid(lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
            for ims in imgs_b:
                for im in ims:
                    im.close()
    if np.isnan(scores[order]).any():
        raise RuntimeError(f"{label}: часть карточек осталась без скора")
    print(f"[llm] {label}: {len(order)} карточек, {time.time()-t0:.0f}с", flush=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return scores
