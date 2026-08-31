"""Скрининг БАД-промптов: v20 работает (паблик +0.001), ищем следующую формулировку.

Разбор ошибок (error_analysis_bad.py, конфиг v20) показал: клауза «Товар НЕ БАД, если
это спортивное питание» КОНФЛИКТУЕТ с разметкой — спортпит размечен как БАД в ~55%,
и весь спортпит-кластер (L-карнитин, BCAA, креатин, аргинин) уходит в пропуски.
Аналогично «не БАД если косметика» бьёт по кремам с меткой БАД.

Проверяем варианты (zero-shot AUC/PR на полном БАД, оба базовых). БАД-разметка
непоследовательна (NOW «Бад» -> метка 0), поэтому AUC-потолок ограничен, но
РАНЖИРОВАНИЕ вариантов сравнимо. БАД-OOF узкий (CI ±0.009) — что победит здесь,
осмысленно потом обучить.

Запуск: CUDA_VISIBLE_DEVICES=2 python exp/bad_prompt_screen.py [qwen|gemma]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, "/workspace/counter/exp")
from flam_prompt_screen import BS, ROOT, SYSTEM, clean  # noqa: E402

BASE = (sys.argv[1] if len(sys.argv) > 1 else "qwen").lower()
MODEL = "Qwen/Qwen3.5-4B" if BASE == "qwen" else "google/gemma-4-E4B-it"
CAT = "БАД"
DESC_CHARS = 1100

# --- официальные правила (до badimg) ---
OFFICIAL = """Правила площадки Ozon для категории «Биологически активные добавки (БАД)».

Товар ЯВЛЯЕТСЯ биологически активной добавкой, если:
1. В описании или на изображении содержится прямое указание, что товар является \
биологически активной добавкой (БАД, dietary supplement, добавка к пище).

Товар НЕ является биологически активной добавкой, если:
1. Товар является спортивным питанием: аминокислоты, BCAA, L-карнитин, креатин, протеин, \
гейнер, предтренировочный комплекс или иной товар с прямым указанием на принадлежность \
к спортивному питанию.
2. В описании явно указано, что товар не является биологически активной добавкой.
3. Товар не содержит маркировок биологически активной добавки (БАД, dietary supplement)."""

# --- v20 (чемпион, прицельный) ---
V20 = ("Внимательно ПРОЧИТАЙ ТЕКСТ НА УПАКОВКЕ на фотографиях. Товар является "
       "БАД, если на упаковке или в описании есть: «БАД», «биологически активная "
       "добавка», «добавка к пище», «dietary supplement», «не является лекарственным "
       "средством», номер свидетельства о государственной регистрации (СГР, RU.77). "
       "Товар НЕ БАД, если это спортивное питание (протеин, BCAA, креатин, гейнер), "
       "лекарство, косметика или обычный продукт питания без такой маркировки.")

# --- без исключений: только позитивные признаки ---
NOEXCL = ("Внимательно ПРОЧИТАЙ ТЕКСТ НА УПАКОВКЕ на фотографиях и в описании. Товар "
          "является БАД, если на упаковке или в описании есть: «БАД», «биологически "
          "активная добавка», «добавка к пище», «dietary supplement», «не является "
          "лекарственным средством», номер свидетельства о государственной регистрации "
          "(СГР, RU.77), или это витамины, минералы, аминокислоты, травяные экстракты и "
          "подобные добавки к пище. Если никаких признаков добавки к пище нет — не БАД.")

# --- спортпит как БАД (по разметке 55% спортпита = БАД) ---
SPORTBAD = ("Внимательно ПРОЧИТАЙ ТЕКСТ НА УПАКОВКЕ на фотографиях. Товар является БАД, "
            "если на упаковке или в описании есть: «БАД», «биологически активная добавка», "
            "«добавка к пище», «dietary supplement», «не является лекарственным средством», "
            "номер СГР (RU.77). Спортивное питание (протеин, BCAA, аминокислоты, креатин, "
            "гейнер, L-карнитин, изотоник) ТОЖЕ обычно относят к БАД. Товар НЕ БАД только "
            "если это лекарство (регистрационное удостоверение, показания к применению), "
            "косметика для наружного применения или обычный продукт питания.")

VARIANTS = {"official": OFFICIAL, "v20": V20, "noexcl": NOEXCL, "sportbad": SPORTBAD}
QUESTION = "Является ли этот товар биологически активной добавкой по правилам выше?"


def build_prompt(tok, rules, name, desc):
    card = f"Название товара: {clean(name)}\n\nОписание товара: {clean(desc)[:DESC_CHARS]}"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"{rules}\n\n---\n\n{card}\n\n---\n\n{QUESTION} Ответь одним словом: Да или Нет."}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    df = df[df["category"] == CAT].reset_index(drop=True)
    y = df["label"].values
    sport = df["name"].str.contains("BCAA|карнитин|креатин|протеин|гейнер|аминокислот|изотоник|аргинин",
                                    case=False, na=False).values
    print(f"[{BASE}] {MODEL}  БАД-строк {len(df)}  поз {y.sum()}  спортпит {int(sport.sum())}", flush=True)

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
        auc, pr = roc_auc_score(y, scores), average_precision_score(y, scores)
        sport_recall = ((scores[sport & (y == 1)] >= 0.5).mean()) if (sport & (y == 1)).any() else 0
        rows.append((vname, auc, pr, fb, tb, sport_recall))
        print(f"MACHINE\t{BASE}\t{vname}\tAUC={auc:.4f}\tPR={pr:.4f}\tF1bin={fb:.4f}@{tb:.2f}\t"
              f"спортпит-recall={sport_recall:.3f}\t{time.time()-t0:.0f}s", flush=True)

    df.to_parquet(ROOT + f"exp/bad_prompt_screen_{BASE}.parquet")
    print(f"\n=== ИТОГ [{BASE}] по AUC (скрининг) ===")
    for vname, auc, pr, fb, tb, sr in sorted(rows, key=lambda r: -r[1]):
        print(f"  {vname:10s} AUC={auc:.4f} PR={pr:.4f} F1bin={fb:.4f} спортпит-recall={sr:.3f}")


if __name__ == "__main__":
    main()
