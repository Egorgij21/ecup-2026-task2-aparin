"""Zero-shot Qwen3.5-4B с ОФИЦИАЛЬНЫМИ правилами категорий в промпте.

Ключевая идея: правила опубликованы организаторами. Модели не нужно видеть бренд
«Forester», чтобы понять, что набор для розжига содержит горючее вещество.
Именно этого обобщения лишён TF-IDF, у которого на флам-категории всего 87 семей позитивов.

Скор = P("Да") / (P("Да") + P("Нет")) по логитам первого сгенерированного токена.
Запуск: CUDA_VISIBLE_DEVICES=6 python exp/llm_zeroshot.py [limit]
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
MODEL = "Qwen/Qwen3.5-4B"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DESC_CHARS = 1100
BS = 32

_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def clean(s):
    s = str(s) if s is not None else ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip()


RULES_FLAM = """Правила площадки Ozon для категории «Легковоспламеняющиеся товары».

Товар ЯВЛЯЕТСЯ легковоспламеняющимся, если выполнено хотя бы одно условие:
1. Товар является самостоятельным источником воспламенения — его основное назначение \
создавать или поддерживать открытый огонь. Например: спички, зажигалки, средства для розжига.
2. Товар содержит горючее вещество, легковоспламеняющуюся жидкость или горючий газ. \
Например: газ для заправки, сухое горючее, уголь и топливные брикеты, пиротехнический состав.
3. В комплект товара входит легковоспламеняющийся товар. Например: одноразовый мангал \
с углём в комплекте, горелка, продаваемая вместе с газовым баллоном.

Товар НЕ является легковоспламеняющимся, если:
1. Он не содержит источника воспламенения или горючего вещества. Устройство, \
предназначенное для использования с огнём или горючими веществами, само по себе НЕ \
является легковоспламеняющимся. Например: мангалы, грили, газовые плиты, горелки без баллона.
2. Легковоспламеняющимся является содержимое, а не сама конструкция; без содержимого \
товар не считается легковоспламеняющимся.
3. Источник воспламенения встроен в изделие (например, пьезоподжиг).
4. Потенциально горючий материал используется лишь как компонент другого изделия и не \
является самостоятельным товаром. Например: активированный уголь в фильтрах, уголь для рисования.
5. Легковоспламеняющийся предмет НЕ входит в комплект."""

RULES_BAD = """Правила площадки Ozon для категории «Биологически активные добавки (БАД)».

Товар ЯВЛЯЕТСЯ биологически активной добавкой, если:
1. В описании или на изображении содержится прямое указание, что товар является \
биологически активной добавкой (БАД, dietary supplement, добавка к пище).

Товар НЕ является биологически активной добавкой, если:
1. Товар является спортивным питанием: аминокислоты, BCAA, L-карнитин, креатин, протеин, \
гейнер, предтренировочный комплекс или иной товар с прямым указанием на принадлежность \
к спортивному питанию.
2. В описании явно указано, что товар не является биологически активной добавкой.
3. Товар не содержит маркировок биологически активной добавки (БАД, dietary supplement)."""

QUESTION = {
    "Легковоспламеняющиеся": "Является ли этот товар легковоспламеняющимся по правилам выше?",
    "БАД": "Является ли этот товар биологически активной добавкой по правилам выше?",
}


def build_messages(cat, name, desc):
    rules = RULES_FLAM if cat == "Легковоспламеняющиеся" else RULES_BAD
    card = f"Название товара: {clean(name)}\n\nОписание товара: {clean(desc)[:DESC_CHARS]}"
    return [
        {"role": "system", "content":
         "Ты модератор карточек товаров на маркетплейсе. Ты строго следуешь правилам "
         "площадки. Отвечай ровно одним словом: Да или Нет."},
        {"role": "user", "content":
         f"{rules}\n\n---\n\n{card}\n\n---\n\n{QUESTION[cat]} Ответь одним словом: Да или Нет."},
    ]


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    if LIMIT:
        df = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(LIMIT, len(g)), random_state=0)).reset_index(drop=True)
    print(f"товаров: {len(df)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    # id токенов ответа (с ведущим пробелом и без, оба регистра)
    def ids_for(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t:
                    out.add(t[0])
        return sorted(out)

    yes_ids, no_ids = ids_for(["Да", "да", "ДА"]), ids_for(["Нет", "нет", "НЕТ"])
    print("yes_ids", yes_ids, "no_ids", no_ids, flush=True)

    prompts = []
    for cat, n, d in zip(df["category"], df["name"].fillna(""), df["description"].fillna("")):
        msgs = build_messages(cat, n, d)
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    print("пример промпта (хвост):\n", prompts[0][-600:], flush=True)

    scores = np.zeros(len(prompts), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), BS):
            enc = tok(prompts[i:i + BS], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to("cuda")
            logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            y = torch.logsumexp(lp[:, yes_ids], dim=-1)
            n = torch.logsumexp(lp[:, no_ids], dim=-1)
            scores[i:i + BS] = torch.sigmoid(y - n).cpu().numpy()
            if i % (BS * 20) == 0:
                el = time.time() - t0
                print(f"  {i}/{len(prompts)}  {el:.0f}s  ~{el/max(i+BS,1)*len(prompts):.0f}s всего",
                      flush=True)

    out = df[["id", "category", "label"]].copy()
    out["llm_score"] = scores
    tag = f"_n{LIMIT}" if LIMIT else ""
    out.to_parquet(ROOT + f"exp/llm_scores{tag}.parquet")
    print(f"\nготово за {time.time()-t0:.0f}s -> exp/llm_scores{tag}.parquet", flush=True)

    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    for cat, g in out.groupby("category"):
        y, s = g["label"].values, g["llm_score"].values
        ths = np.linspace(0.01, 0.99, 99)
        fb = max((f1_score(y, (s >= t).astype(int)), t) for t in ths)
        fm = max((f1_score(y, (s >= t).astype(int), average="macro"), t) for t in ths)
        print(f"{cat}: n={len(g)} pos={y.sum()} AUC={roc_auc_score(y,s):.4f} "
              f"PR={average_precision_score(y,s):.4f} F1bin={fb[0]:.4f}@{fb[1]:.2f} "
              f"F1mac={fm[0]:.4f}@{fm[1]:.2f}")


if __name__ == "__main__":
    main()
