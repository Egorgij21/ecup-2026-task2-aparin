"""Декомпозиция правил на атомарные вопросы к LLM.

Вместо одного тупого «является ли товар легковоспламеняющимся?» задаём отдельный
вопрос на КАЖДЫЙ пункт правил. Каждый ответ — отдельный признак для линейной модели,
которая сама решит, как их взвесить.

Плюс: вопрос самодостаточен, полный текст правил в промпт класть не нужно —
промпты втрое короче (~250 токенов против ~600), поэтому 6 вопросов стоят
примерно как 2.5 полных прогона.

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/llm_questions.py [limit]
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
from src.prompts import clean  # noqa: E402

MODEL = os.environ.get("QWEN_PATH", "Qwen/Qwen3.5-4B")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DESC_CHARS, BS, MAXLEN = 700, 64, 640

QUESTIONS = {
    "Легковоспламеняющиеся": [
        ("self_source", "Основное назначение этого товара — создавать или поддерживать "
                        "открытый огонь (спички, зажигалки, средства для розжига)?"),
        ("contains_fuel", "Содержит ли сам товар горючее вещество, легковоспламеняющуюся "
                          "жидкость или горючий газ?"),
        ("kit_fuel", "Входит ли в комплект поставки горючее вещество или источник огня — "
                     "уголь, газ, спички, топливо, пиротехнический состав?"),
        ("device_only", "Это устройство или конструкция для использования с огнём (мангал, "
                        "гриль, плита, горелка), которое само по себе топлива не содержит?"),
        ("sold_empty", "Продаётся ли этот товар пустым, без горючего содержимого?"),
        ("pyro", "Содержит ли товар пиротехнический состав (хлопушка, дымовая шашка, "
                 "фейерверк, бенгальский огонь)?"),
    ],
    "БАД": [
        ("bad_marked", "Есть ли в карточке прямое указание, что товар является биологически "
                       "активной добавкой (БАД, dietary supplement, добавка к пище)?"),
        ("sport", "Является ли этот товар спортивным питанием (протеин, BCAA, креатин, "
                  "L-карнитин, гейнер, предтренировочный комплекс)?"),
        ("denies_bad", "Указано ли в карточке явно, что товар НЕ является биологически "
                       "активной добавкой?"),
        ("oral_supplement", "Это витамины, минералы или растительный экстракт в форме "
                            "капсул, таблеток или порошка для приёма внутрь?"),
        ("not_ingestible", "Это аксессуар или товар не для приёма внутрь (таблетница, "
                           "шейкер, косметика, контейнер)?"),
    ],
}

SYSTEM = "Ты модератор карточек товаров на маркетплейсе. Отвечай ровно одним словом: Да или Нет."


def card(name, desc):
    return f"Название: {clean(name)}\nОписание: {clean(desc)[:DESC_CHARS]}"


def build(tok, name, desc, question):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"Карточка товара:\n{card(name, desc)}\n\nВопрос: {question} "
             "Ответь одним словом: Да или Нет."}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    if LIMIT:
        df = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(LIMIT, len(g)), random_state=0)).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    yes_id = tok.encode("Да", add_special_tokens=False)[0]
    no_id = tok.encode("Нет", add_special_tokens=False)[0]

    out = df[["id", "category", "label"]].copy()
    t0 = time.time()
    for cat, qs in QUESTIONS.items():
        m = (df["category"] == cat).values
        sub = df[m]
        if not len(sub):
            continue
        for key, q in qs:
            prompts = [build(tok, n, d, q)
                       for n, d in zip(sub["name"].fillna(""), sub["description"].fillna(""))]
            sc = np.zeros(len(prompts), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(prompts), BS):
                    enc = tok(prompts[i:i + BS], return_tensors="pt", padding=True,
                              truncation=True, max_length=MAXLEN).to("cuda")
                    lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
                    sc[i:i + BS] = torch.sigmoid(lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
            col = f"q_{key}"
            if col not in out:
                out[col] = np.nan
            out.loc[m, col] = sc
            print(f"  [{cat[:12]}] {key:16s} среднее={sc.mean():.3f} "
                  f"({time.time()-t0:.0f}с)", flush=True)

    tag = f"_n{LIMIT}" if LIMIT else ""
    out.to_parquet(ROOT + f"exp/llm_questions{tag}.parquet")
    print(f"\nготово за {time.time()-t0:.0f}с -> exp/llm_questions{tag}.parquet")

    # быстрая оценка одиночной полезности каждого вопроса
    from sklearn.metrics import roc_auc_score
    for cat, qs in QUESTIONS.items():
        g = out[out["category"] == cat]
        if not len(g):
            continue
        print(f"\n{cat}: AUC отдельных вопросов (0.5 = бесполезен)")
        for key, _ in qs:
            s = g[f"q_{key}"].values
            print(f"   {key:16s} AUC={roc_auc_score(g['label'], s):.4f}")


if __name__ == "__main__":
    main()
