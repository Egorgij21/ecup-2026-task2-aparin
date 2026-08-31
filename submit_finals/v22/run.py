"""E-CUP 2026 «Контроль качества» — v20: ПРИЦЕЛЬНЫЙ ПРОМПТ на БАД.

Единственное отличие от чемпиона v16: БАД-адаптер с картинками заменён на обученный
с прицельным промптом (_BAD_IMG — прямо называет надписи на упаковке: «не является
лекарственным средством», номер СГR RU.77), и на прогоне ему подаётся тот же промпт
(build_messages_badimg только в score_cards_mm). Порог, веса, флам, zero-shot ансамбль —
идентичны v16. На полном OOF при фикс. пороге: БАД 0.9487 против 0.9459 (+0.0028).

База решения (из v16): бленд ДВУХ базовых моделей на флам-категории.
Флам-скор собирается из двух LoRA-адаптеров на разных базах (gemma-4-E4B и
Qwen3.5-4B) вместо одного.

ЗАЧЕМ. Локальный полный OOF и паблик разошлись в противоположные стороны на одном
и том же сравнении: Qwen 0.8997 против геммы 0.8963 локально, но 0.87820 против
0.89017 на паблике. На паблике всего ~58 позитивов флам, и вся разница в 0.012 —
это около десяти строк. Различить базы мы не можем, и выбор одной означает
подбрасывание монеты на привате.

Бленд убирает выбор. И он не просто страховка: на полном OOF **девять смешанных
весов из одиннадцати обходят обе чистые базы**. Работает потому, что корреляция
OOF-скоров двух баз 0.875 — заметно ниже, чем у адаптеров одной базы (0.97).

Числа (полный OOF, вложенный порог, конфигурация сабмита):
  только Qwen              0.8997   -> LB 0.87820
  только gemma             0.8963   -> LB 0.89017
  поровну (эта версия)     0.9008
  вес подобран вложенно    0.9035   <- честная оценка с подбором
  наивно лучший вес 0.8    0.9110   <- НЕ брать, цена подбора +0.0076

Вес зафиксирован 50/50 БЕЗ подбора: весь диапазон 0.3-0.9 внутри шума друг друга,
а подбор по 198 позитивам не переносится — так мы уже потеряли сабмит v3.

Порог 0.45 — медиана вложенных порогов по фолдам (0.425..0.46), плато 0.38..0.49.

Три модели, все из /shared_models:
  * Qwen3.5-4B — zero-shot для ансамбля. Коэффициенты ансамбля обучены именно на
    его скорах, подменять базу под ними нельзя.
  * gemma-4-E4B + LoRA (lora_adapter)  — половина флам-скора.
  * Qwen3.5-4B + LoRA (lora_adapter2)  — вторая половина.
База каждого адаптера читается из его собственного adapter_config.json.

Никаких откатов: нет модели, адаптера, CUDA — падаем громко.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comments import format_result, make_comment   # noqa: E402
from src.lora import (ZERO_SHOT_MODEL, adapter_base_subpath,  # noqa: E402
                      resolve_model, score_cards, score_cards_mm)
from src.model import build_text, clean_text, llm_features, load_models  # noqa: E402
from src.prompts import (build_messages, build_messages_badimg,  # noqa: E402
                         build_messages_noexcl, build_messages_sportbad)
from src.rules import extract                          # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ADAPTERS = [HERE / "lora_adapter", HERE / "lora_adapter2"]
# Отдельный МУЛЬТИМОДАЛЬНЫЙ адаптер только для БАД. На всех трёх посчитанных фолдах
# он обошёл текстовый: +0.0060, +0.0007, +0.0071 к F1 бленда. На флам картинки
# наоборот вредят (в среднем −0.010), поэтому там их не подаём.
# Основание из правил площадки: критерий БАД дословно ссылается на изображение
# («в описании ИЛИ НА ИЗОБРАЖЕНИИ содержится прямое указание»), у флам такого нет.
ADAPTER_BAD = HERE / "lora_adapter_bad"
# v22: 3-бленд БАД-промптов. Три gemma+картинки адаптера, обученные с РАЗНЫМИ промптами
# (badimg=v20, noexcl, sportbad). На OOF бленд дал 0.9492 против v20 0.9487 — ВНУТРИ шума
# (±0.009, P=0.52), т.е. диверсификация, не подтверждённый прирост. Корр адаптеров 0.97
# (промпты слабо декоррелируют). Каждый скорится СВОИМ промптом.
ADAPTER_BAD2 = HERE / "lora_adapter_bad2"   # noexcl
ADAPTER_BAD3 = HERE / "lora_adapter_bad3"   # sportbad
FLAM = "Легковоспламеняющиеся"
BAD = "БАД"

BLEND_W = float(os.environ.get("BLEND_W", "0.5"))          # вес LoRA-части в бленде
BLEND_THRESHOLD = float(os.environ.get("BLEND_THRESHOLD", "0.45"))
# порог БАД для бленда: вложенные пороги по фолдам 0.36 0.36 0.36 0.32 0.39
# 0.47 вместо 0.36: на полном OOF (фикс. порог, как в сабмите) F1 БАД растёт
# 0.9432 -> 0.9462. Это ПЛАТО — 0.46, 0.47, 0.48 дают почти одно, значит выбор
# устойчив, а не подгонка под зубец. Единственное отличие от чемпиона v15.
BAD_THRESHOLD = float(os.environ.get("BAD_THRESHOLD", "0.47"))


def parse_args():
    p = argparse.ArgumentParser(description="Product quality predictor")
    p.add_argument("--test_data_path", "--test-data-path", "-i", dest="test_data_path",
                   required=True)
    p.add_argument("--output_path", "--output-path", "-o", dest="output_path", required=True)
    return p.parse_args()


def preflight():
    if not ART.joinpath("meta.json").exists():
        raise FileNotFoundError(f"нет обязательного файла: {ART / 'meta.json'}")
    bases = []
    for a in ADAPTERS:
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            if not a.joinpath(f).exists():
                raise FileNotFoundError(f"нет обязательного файла: {a / f}")
        base = adapter_base_subpath(str(a))
        resolve_model(base)                 # упадём здесь, а не после часа работы
        bases.append(base)
        print(f"[preflight] адаптер {a.name}: база {base}", flush=True)
    if len(set(bases)) != len(bases):
        raise ValueError(f"адаптеры на одной базе {bases} — бленд двух баз теряет смысл")
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not ADAPTER_BAD.joinpath(f).exists():
            raise FileNotFoundError(f"нет обязательного файла: {ADAPTER_BAD / f}")
    bad_base = adapter_base_subpath(str(ADAPTER_BAD))
    resolve_model(bad_base)
    print(f"[preflight] адаптер БАД (с картинками): база {bad_base}", flush=True)
    for extra in (ADAPTER_BAD2, ADAPTER_BAD3):
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            if not extra.joinpath(f).exists():
                raise FileNotFoundError(f"нет обязательного файла: {extra / f}")
        resolve_model(adapter_base_subpath(str(extra)))
        print(f"[preflight] доп. БАД-адаптер {extra.name}: ок", flush=True)
    import torch  # noqa: F401
    import transformers  # noqa: F401
    print(f"[preflight] zero-shot: {resolve_model(ZERO_SHOT_MODEL)}", flush=True)
    print(f"[preflight] бленд: {1 - BLEND_W:.2f}*ансамбль + {BLEND_W:.2f}*среднее("
          f"{len(bases)} LoRA), порог {BLEND_THRESHOLD}", flush=True)
    return bases, bad_base


def main():
    t0 = time.time()
    args = parse_args()
    bases, bad_base = preflight()

    df = pd.read_csv(Path(args.test_data_path))
    print(f"[data] {len(df)} строк", flush=True)
    for col in ("id", "name", "category"):
        if col not in df.columns:
            raise ValueError(f"во входном csv нет колонки {col!r}: {list(df.columns)}")
    if "description" not in df.columns:
        df["description"] = ""
    names = df["name"].fillna("").tolist()
    descs = df["description"].fillna("").tolist()
    cats = df["category"].fillna(BAD).tolist()
    unknown = set(cats) - {FLAM, BAD}
    if unknown:
        raise ValueError(f"неизвестные категории во входных данных: {unknown}")

    models, name_rep, _ = load_models(ART)
    for c in (FLAM, BAD):
        if c not in models:
            raise ValueError(f"в артефактах нет модели для категории {c!r}")

    # ЗОНД: LoRA применяется к ОБЕИМ категориям, а не только к флам.
    # Локально это даёт БАД 0.9334 против 0.9378 у чистого ансамбля, то есть
    # ожидание -0.002 к метрике. Смысл отправки не в приросте, а в проверке:
    # на паблике БАД-строк ~920 с ~685 позитивами против 24 позитивов у флам,
    # то есть БАД там измеряется точно. Если паблик покажет заметно иное,
    # значит наша локальная оценка БАД неверна — а это половина метрики.
    flam_idx = [i for i, c in enumerate(cats) if c == FLAM]
    bad_idx = [i for i, c in enumerate(cats) if c != FLAM]
    # текстовый ансамбль нужен и на флам (он половина бленда), значит zero-shot нужен всем
    zero = score_cards(ZERO_SHOT_MODEL, cats, names, descs, build_messages,
                       list(range(len(cats))))
    L = llm_features(zero)

    lora_scores = []
    for a, base in zip(ADAPTERS, bases):
        lora_scores.append(score_cards(base, cats, names, descs, build_messages,
                                       flam_idx, adapter_dir=str(a)))
    # картинки лежат рядом с тестовым csv — так же устроен бейзлайн организаторов
    images_dir = Path(args.test_data_path).parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"нет папки с изображениями: {images_dir}")
    # ПРИЦЕЛЬНЫЙ ПРОМПТ (v20, единственное отличие от v16): БАД-адаптер обучался с
    # PROMPT_VARIANT=badimg, поэтому на прогоне обязан получать тот же _BAD_IMG.
    # Только здесь — zero-shot ансамбль и флам остаются на build_messages.
    # v22: 3-бленд БАД — три адаптера, КАЖДЫЙ со своим обучающим промптом.
    bad_parts = []
    for adir, bmsg in [(ADAPTER_BAD, build_messages_badimg),
                       (ADAPTER_BAD2, build_messages_noexcl),
                       (ADAPTER_BAD3, build_messages_sportbad)]:
        ab = adapter_base_subpath(str(adir))
        bad_parts.append(score_cards_mm(ab, cats, names, descs, df["id"].tolist(),
                                        bmsg, bad_idx, adapter_dir=str(adir),
                                        images_dir=images_dir))
    bad_lora = [sum(p[i] for p in bad_parts) / len(bad_parts) for i in range(len(cats))]

    full = [clean_text(n) + " " + clean_text(d) for n, d in zip(names, descs)]
    name_only = [clean_text(n) for n in names]
    rules = {}
    for cat in set(cats):
        idx = [i for i, c in enumerate(cats) if c == cat]
        R = extract([full[i] for i in idx], [name_only[i] for i in idx], cat)
        for k, i in enumerate(idx):
            rules[i] = R[k]

    results, counts, src_counts = [], {}, {}
    for i in range(len(df)):
        cat = cats[i]
        model = models[cat]
        text = build_text(names[i], descs[i], name_rep=name_rep)
        ens = model.score(text, rules[i], L[i], True)
        if cat == FLAM:
            lora = sum(float(s[i]) for s in lora_scores) / len(lora_scores)
            prob = (1.0 - BLEND_W) * ens + BLEND_W * lora
            positive = prob >= BLEND_THRESHOLD
            src = f"blend{len(lora_scores)}"
        else:
            prob = (1.0 - BLEND_W) * ens + BLEND_W * float(bad_lora[i])
            positive = prob >= BAD_THRESHOLD
            src = "blend_mm"
        comment = make_comment(cat, full[i], positive, prob)
        results.append(format_result(comment, positive))
        counts[(cat, positive)] = counts.get((cat, positive), 0) + 1
        src_counts[(cat, src)] = src_counts.get((cat, src), 0) + 1

    out = pd.DataFrame({"id": df["id"].values, "result": results})
    if len(out) != len(df) or out["result"].isna().any():
        raise RuntimeError("результат неполон — не записываем заведомо битый файл")
    out.to_csv(args.output_path, index=False)

    for (cat, src), n in sorted(src_counts.items(), key=lambda kv: str(kv[0])):
        print(f"[src]  {cat} <- {src}: {n}", flush=True)
    for (cat, pos), n in sorted(counts.items(), key=lambda kv: str(kv[0])):
        print(f"[pred] {cat} -> {'не бан' if pos else 'бан'}: {n}", flush=True)
    print(f"[done] {len(df)} строк за {time.time()-t0:.1f}с -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
