"""Скрининг ПРОМПТОВ для флам БЕЗ обучения: zero-shot AUC/PR на полном флам-наборе.

Зачем. Прицельный промпт на БАД дал +0.0028 после обучения, а на хлопушках +0.12 AUC
уже в zero-shot — то есть zero-shot AUC это дешёвый и валидный СКРИНИНГ формулировки
ДО того, как тратить 5 фолдов × ~2.7ч. Здесь сравниваем несколько прицельных промптов
для флам, каждый на всём флам-наборе, одной загрузкой модели.

Прицельные промпты закодированы под РАЗМЕТКУ (разбор ошибок error_analysis_flam.py):
  * газовые горелки/баллоны = флам ДАЖЕ с пьезоподжигом (газ важнее поджига) — крупнейший
    кластер пропусков;
  * уголь и топливные брикеты, средства розжига, сухое горючее = флам;
  * хлопушки на сжатом воздухе (пневматика) = НЕ флам — крупнейший кластер ложных;
  * «вечная спичка»/огниво/кремень сам по себе = НЕ флам;
  * горючее лишь как компонент (набор выживания, фильтр) = НЕ флам.
Шум (Forester/Boyscout то так, то эдак) НЕ кодируем — под него подгонять нельзя.

ВАЖНО: это СКРИНИНГ, а не решение. Решение — полный OOF при фикс. пороге ПОСЛЕ обучения
лучшего промпта. Здесь AUC/PR только чтобы отобрать формулировку из нескольких.

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/flam_prompt_screen.py [base]
  base: qwen (по умолчанию) | gemma
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = "/workspace/counter/"
BASE = (sys.argv[1] if len(sys.argv) > 1 else "qwen").lower()
MODEL = "Qwen/Qwen3.5-4B" if BASE == "qwen" else "google/gemma-4-E4B-it"
CAT = "Легковоспламеняющиеся"
DESC_CHARS = 1100
BS = 32

_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def clean(s):
    s = str(s) if s is not None else ""
    if s == "nan":
        s = ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", _TAG.sub(" ", s)).strip()


# --- ВАРИАНТ base: официальные правила площадки (контроль, = текущий чемпион) ---
RULES_BASE = """Правила площадки Ozon для категории «Легковоспламеняющиеся товары».

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

# --- ВАРИАНТ types: прицельный список ТИПОВ товаров под разметку ---
RULES_TYPES = """Определи, является ли товар легковоспламеняющимся, по его типу.

Товар ЛЕГКОВОСПЛАМЕНЯЮЩИЙСЯ, если это:
— спички (кроме «вечных» и кремнёвых), зажигалки;
— газовая горелка, газовый резак, паяльная лампа или газовый баллон — ДАЖЕ если у горелки \
есть пьезоподжиг: наличие горючего газа важнее способа поджига;
— газ для заправки (бутан, пропан, мапп-газ, грин-газ, цанговый баллон);
— сухое горючее, таблетки и гель для розжига;
— средства розжига: жидкость, паста, роллы, палочки, кубики, брикеты для розжига;
— уголь древесный, топливные и угольные брикеты для мангала, гриля, барбекю;
— пиротехника с горючим составом: петарды, фейерверки, бенгальские огни, фонтаны, \
дымовые шашки с пиротехническим составом.

Товар НЕ легковоспламеняющийся, если это:
— «вечная спичка», огниво, кремень, ферроцериевый стержень — сами по себе не горят;
— хлопушка, пневмохлопушка, конфетти-пушка на сжатом воздухе (пневматические, БЕЗ \
пиротехнического состава);
— мангал, гриль, плита, горелка БЕЗ баллона в комплекте — сама конструкция без топлива;
— товар, где горючее лишь компонент (набор выживания с огнивом, фильтр с углём, \
уголь для рисования)."""

# --- ВАРИАНТ package: «прочитай маркировку на упаковке» (аналог badimg для БАД) ---
RULES_PACKAGE = """Внимательно прочитай название, описание и маркировку товара.

Товар ЛЕГКОВОСПЛАМЕНЯЮЩИЙСЯ, если есть признаки: знак пламени (GHS02), надписи \
«огнеопасно», «легковоспламеняющийся», «горючий газ/жидкость», «пиротехническое изделие», \
«ТР ТС 006/2011», указан класс опасности пиротехники, «сжиженный/сжатый горючий газ», \
состав содержит горючее вещество; либо товар по сути — источник огня (спичка, зажигалка, \
розжиг, сухое горючее) или содержит горючий газ (баллон, газовая горелка, в т.ч. с \
пьезоподжигом), уголь и топливные брикеты.

Товар НЕ легковоспламеняющийся, если таких признаков нет: изделие работает на сжатом \
ВОЗДУХЕ (пневматическая хлопушка/пушка), это «вечная спичка»/огниво/кремень (не горит \
само), либо горючее — лишь компонент (набор выживания, фильтр), либо это пустая \
конструкция без топлива (мангал, гриль, горелка без баллона)."""

# --- ВАРИАНТ combo: типы + маркировка вместе ---
RULES_COMBO = RULES_TYPES + "\n\nДополнительно смотри маркировку: знак пламени, " \
    "«огнеопасно», «пиротехническое изделие», «ТР ТС 006/2011», класс опасности, " \
    "«горючий газ» — это признаки легковоспламеняющегося; «пневматическая», " \
    "«на сжатом воздухе» — признак НЕ легковоспламеняющегося."

VARIANTS = {
    "base": RULES_BASE,
    "types": RULES_TYPES,
    "package": RULES_PACKAGE,
    "combo": RULES_COMBO,
}
QUESTION = "Является ли этот товар легковоспламеняющимся по правилам выше?"
SYSTEM = ("Ты модератор карточек товаров на маркетплейсе. Ты строго следуешь правилам "
          "площадки. Отвечай ровно одним словом: Да или Нет.")


def build_prompt(tok, rules, name, desc):
    card = f"Название товара: {clean(name)}\n\nОписание товара: {clean(desc)[:DESC_CHARS]}"
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"{rules}\n\n---\n\n{card}\n\n---\n\n{QUESTION} Ответь одним словом: Да или Нет."},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    df = df[df["category"] == CAT].reset_index(drop=True)
    y = df["label"].values
    print(f"[{BASE}] {MODEL}  флам-строк {len(df)}  позитивов {y.sum()}", flush=True)

    if BASE == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                     local_files_only=True).cuda().eval()
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        tok = proc.tokenizer
        tok.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                            local_files_only=True).cuda().eval()

    import inspect
    _f = inspect.signature(model.forward).parameters
    keep = ({"logits_to_keep": 1} if "logits_to_keep" in _f
            else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _f else {})

    def ids_for(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t:
                    out.add(t[0])
        return sorted(out)

    yes_ids, no_ids = ids_for(["Да", "да"]), ids_for(["Нет", "нет"])

    names, descs = df["name"].fillna("").tolist(), df["description"].fillna("").tolist()
    rows = []
    for vname, rules in VARIANTS.items():
        prompts = [build_prompt(tok, rules, n, d) for n, d in zip(names, descs)]
        # сортируем по длине для эффективного паддинга, помним обратный порядок
        order = np.argsort([len(p) for p in prompts])
        scores = np.zeros(len(prompts), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(order), BS):
                ch = order[i:i + BS]
                enc = tok([prompts[j] for j in ch], return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to("cuda")
                lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
                yy = torch.logsumexp(lp[:, yes_ids], dim=-1)
                nn = torch.logsumexp(lp[:, no_ids], dim=-1)
                scores[ch] = torch.sigmoid(yy - nn).cpu().numpy()
        df[f"s_{vname}"] = scores
        ths = np.linspace(0.01, 0.99, 99)
        fb, tb = max((f1_score(y, (scores >= t).astype(int)), t) for t in ths)
        fm, tm = max((f1_score(y, (scores >= t).astype(int), average="macro"), t) for t in ths)
        auc, pr = roc_auc_score(y, scores), average_precision_score(y, scores)
        dt = time.time() - t0
        rows.append((vname, auc, pr, fb, tb, fm, tm))
        print(f"MACHINE\t{BASE}\t{vname}\tAUC={auc:.4f}\tPR={pr:.4f}\t"
              f"F1bin={fb:.4f}@{tb:.2f}\tF1mac={fm:.4f}@{tm:.2f}\t{dt:.0f}s", flush=True)

    df.to_parquet(ROOT + f"exp/flam_prompt_screen_{BASE}.parquet")
    print(f"\n=== ИТОГ [{BASE}] по AUC (скрининг, не решение) ===", flush=True)
    for vname, auc, pr, fb, tb, fm, tm in sorted(rows, key=lambda r: -r[1]):
        print(f"  {vname:8s} AUC={auc:.4f} PR={pr:.4f} F1bin={fb:.4f} F1mac={fm:.4f}", flush=True)
    print(f"\nскоры сохранены -> exp/flam_prompt_screen_{BASE}.parquet", flush=True)


if __name__ == "__main__":
    main()
