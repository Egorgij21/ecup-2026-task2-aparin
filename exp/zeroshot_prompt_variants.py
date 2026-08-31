"""Варианты промпта для zero-shot: структурируем решение, НЕ платя за генерацию.

Что уже провалилось и почему это другое:
  * нативный режим рассуждения (enable_thinking=True) — +0.0036 при пороге +0.05;
  * атомарные вопросы — 0.8439 против 0.8615 в бленде.
Обе попытки требовали ГЕНЕРАЦИИ сотен токенов. Здесь ответ по-прежнему читается
одним forward-проходом, стоимость не меняется вовсе.

Зачем вообще: zero-shot в спорной зоне LoRA даёт AUC 0.4680 — ниже случайного, —
хотя вносит в флам-ансамбль +0.042. Компонент небесполезный, но в решающей зоне слепой.

Критерий задан ЗАРАНЕЕ, нужны ОБА условия (одно легко выбить зазубриной):
  1) AUC в спорной зоне > 0.52 (сейчас 0.4680);
  2) F1 бленда на полном OOF > 0.8665 (сейчас 0.8615).

Запуск: CUDA_VISIBLE_DEVICES=N python exp/zeroshot_prompt_variants.py <вариант>
        варианты: checklist | exclusions | short
Пишет exp/llm_scores_<вариант>.parquet в том же формате, что llm_scores.parquet.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "submit2")
from src.prompts import RULES_BAD, RULES_FLAM, clean  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "checklist"
MODEL = os.environ.get("ZS_MODEL", "Qwen/Qwen3.5-4B")
BS, MAXLEN, DESC = 24, 2048, 1100
FLAM = "Легковоспламеняющиеся"

# Короткие правила: проверяем, помогает ли полный текст ТЗ или размывает внимание.
SHORT = {
    FLAM: ("Товар легковоспламеняющийся, если он сам создаёт огонь (спички, зажигалки, "
           "средства розжига), либо содержит горючее вещество, газ или пиротехнический "
           "состав, либо горючее входит в его комплект. НЕ является: устройство без "
           "горючего (мангал, гриль, плита, горелка без баллона), встроенный пьезоподжиг, "
           "горючий материал как компонент другого изделия."),
    "БАД": ("Товар является БАД, если в описании есть прямое указание на это (БАД, "
            "dietary supplement, добавка к пище). НЕ является: спортивное питание "
            "(протеин, BCAA, креатин, гейнер, аминокислоты), явное отрицание, "
            "отсутствие маркировки БАД."),
}

CHECKLIST = {
    FLAM: ("Проверь по порядку:\n"
           "1. Основное назначение товара — создавать или поддерживать открытый огонь?\n"
           "2. Товар содержит горючее вещество, легковоспламеняющуюся жидкость или газ?\n"
           "3. Легковоспламеняющийся предмет входит в КОМПЛЕКТ товара?\n"
           "Если хотя бы на один пункт ответ «да» — товар легковоспламеняющийся."),
    "БАД": ("Проверь по порядку:\n"
            "1. Есть ли прямое указание, что товар — БАД или добавка к пище?\n"
            "2. Не является ли товар спортивным питанием?\n"
            "Товар является БАД, только если на первый пункт «да», а на второй «да»."),
}

EXCLUSIONS = {
    FLAM: ("Сначала проверь ИСКЛЮЧЕНИЯ. Товар НЕ легковоспламеняющийся, если это "
           "устройство для использования с огнём, но без горючего в комплекте "
           "(мангал, гриль, плита, горелка без баллона), если горючее — лишь компонент "
           "другого изделия, или если поджиг встроен в изделие.\n"
           "Только если ни одно исключение не подходит, проверь: создаёт ли товар огонь, "
           "содержит ли горючее вещество, входит ли горючее в комплект."),
    "БАД": ("Сначала проверь ИСКЛЮЧЕНИЯ. Товар НЕ является БАД, если это спортивное "
            "питание (протеин, BCAA, креатин, гейнер, аминокислоты, предтренировочный "
            "комплекс), если есть явное отрицание, или если маркировки БАД нет.\n"
            "Только если ни одно исключение не подходит, проверь наличие прямого "
            "указания на БАД."),
}

QUESTION = {FLAM: "Является ли этот товар легковоспламеняющимся?",
            "БАД": "Является ли этот товар биологически активной добавкой?"}


def build(cat, name, desc):
    rules = SHORT[cat] if VARIANT == "short" else (RULES_FLAM if cat == FLAM else RULES_BAD)
    extra = {"checklist": CHECKLIST, "exclusions": EXCLUSIONS}.get(VARIANT, {}).get(cat, "")
    body = f"{rules}\n\n{extra}\n\n" if extra else f"{rules}\n\n"
    card = f"Название: {clean(name)}\nОписание: {clean(desc)[:DESC]}"
    return [{"role": "user",
             "content": f"{body}Карточка товара:\n{card}\n\n{QUESTION[cat]} "
                        f"Ответь одним словом: Да или Нет."}]


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    def ids_for(ws):
        out = set()
        for w in ws:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t:
                    out.add(t[0])
        return sorted(out)

    yes_ids, no_ids = ids_for(["Да", "да", "ДА"]), ids_for(["Нет", "нет", "НЕТ"])
    import inspect
    _f = inspect.signature(model.forward).parameters
    keep = ({"logits_to_keep": 1} if "logits_to_keep" in _f
            else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _f else {})
    if not keep:
        raise RuntimeError("модель не поддерживает logits_to_keep — будет OOM на полных логитах")

    prompts = []
    for c, n, d in zip(df["category"], df["name"].fillna(""), df["description"].fillna("")):
        msgs = build(c, n, d)
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    print(f"вариант={VARIANT}, карточек {len(prompts)}", flush=True)
    print("ХВОСТ ПРОМПТА:\n" + prompts[0][-700:], flush=True)

    order = np.argsort([len(p) for p in prompts])       # сортировка по длине: меньше паддинга
    scores = np.full(len(prompts), np.nan, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(order), BS):
            ch = order[i:i + BS]
            enc = tok([prompts[j] for j in ch], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN).to("cuda")
            lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
            y = torch.logsumexp(lp[:, yes_ids], dim=-1)
            n = torch.logsumexp(lp[:, no_ids], dim=-1)
            scores[ch] = torch.sigmoid(y - n).cpu().numpy()
            if i % (BS * 40) == 0:
                el = time.time() - t0
                print(f"  {i}/{len(order)} {el:.0f}с ~{el/max(i+BS,1)*len(order):.0f}с всего",
                      flush=True)
    if np.isnan(scores).any():
        raise RuntimeError("часть карточек без скора")

    out = df[["id", "category", "label"]].copy()
    out["llm_score"] = scores
    out.to_parquet(ROOT + f"exp/llm_scores_{VARIANT}.parquet")
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"\nготово за {time.time()-t0:.0f}с -> exp/llm_scores_{VARIANT}.parquet")
    for cat, g in out.groupby("category"):
        print(f"  {cat}: AUC={roc_auc_score(g['label'], g['llm_score']):.4f} "
              f"PR={average_precision_score(g['label'], g['llm_score']):.4f}", flush=True)


if __name__ == "__main__":
    main()
