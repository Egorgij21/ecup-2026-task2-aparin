"""Пригодна ли InternVL3_5-2B: проверка ДО обучения, чтобы не жечь карту впустую.

Nanbeige отвалилась именно здесь: ей нужен trust_remote_code, и её собственный код
упал на нашей версии transformers (KeyError: 'type' в rope_scaling). InternVL тоже
требует trust_remote_code, но она ЕСТЬ в списке организаторов — значит в их образе
работает. Проверяем в наших условиях.

Печатает «ПРИГОДНА» и строку «РЕГУЛЯРКА: ...» только если сошлось всё:
  1) модель грузится локально;
  2) «Да» и «Нет» дают РАЗНЫЕ первые токены (скор строится на одном токене ответа);
  3) находится непустой набор линейных модулей языковой части для LoRA.
Иначе печатает причину и очередь пропускает обучение.
"""
import re
import sys
import traceback

import torch

MODEL = "OpenGVLab/InternVL3_5-2B"


def main():
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
    y = tok.encode("Да", add_special_tokens=False)
    n = tok.encode("Нет", add_special_tokens=False)
    print(f"'Да' -> {y}, 'Нет' -> {n}")
    if not y or not n or y[0] == n[0]:
        print(f"НЕПРИГОДНА: первые токены совпадают ({y[:1]} == {n[:1]}), скор неразличим")
        return

    model = None
    for cls in (AutoModelForCausalLM, AutoModel):
        try:
            model = cls.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True)
            print(f"загружена как {cls.__name__}")
            break
        except Exception as e:
            print(f"{cls.__name__} не подошёл: {type(e).__name__}: {str(e)[:160]}")
    if model is None:
        print("НЕПРИГОДНА: ни один класс модели не загрузился")
        return

    lin = [nm for nm, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"линейных модулей всего: {len(lin)}")
    print("примеры:", lin[:4])

    # ищем языковую часть: у InternVL это language_model.*, внутри Qwen3ForCausalLM
    cands = [r"language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
             r"language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
             r".*\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"]
    for rx in cands:
        hit = [nm for nm in lin if re.fullmatch(rx, nm)]
        print(f"  регулярка {rx[:52]}... -> {len(hit)} модулей")
        if len(hit) >= 50:
            print("ПРИГОДНА")
            print(f"РЕГУЛЯРКА: {rx}")
            return
    print("НЕПРИГОДНА: не нашлось набора модулей языковой части для LoRA")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("НЕПРИГОДНА: исключение при проверке")
        sys.exit(0)      # выходим мягко: очередь сама пропустит обучение
