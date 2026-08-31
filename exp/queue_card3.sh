#!/bin/bash
# Всё на ОДНОЙ карте (3), строго по очереди. Других карт не занимать.
#
# Порядок выбран по цене вопроса:
#   1) Qwen3-VL-2B, фолды 1-4 — по 20 минут каждый, всего ~1.5 часа.
#      Прошла ворота по корреляции: 0.792 с Qwen и 0.814 с геммой при пороге 0.85.
#      Для сравнения, пара Qwen x gemma на том же фолде даёт 0.949, а на полном
#      OOF 0.875 — фолдовые корреляции завышены, значит у VL-2B на полном OOF
#      ожидается ~0.72-0.74, то есть заметно независимее уже работающей пары.
#   2) семплер family, фолды 3-4 — по 2.3 часа, всего ~4.5.
#      Чинит измеренную поломку: 0.48 позитива флам на эффективный батч против 6.8.
#
# Дешёвый вопрос идёт первым: за полтора часа получаем полный OOF третьей базы,
# а не ждём пять часов ради гипотезы, которую и так можно доделать следом.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
VLT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'

for f in 1 2 3 4; do
    echo "$(date +%H:%M:%S) [карта 3] VL-2B фолд $f" >> queue_card3.log
    LORA_TAG=_vl2b LORA_BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct LORA_TARGETS="$VLT" \
        $PY -u lora_train.py "$f" 2 > "lora_fold${f}_vl2b.out" 2>&1
done
echo "$(date +%H:%M:%S) [карта 3] VL-2B: все пять фолдов -> eval_oof_tagged.py _vl2b" >> queue_card3.log

for f in 3 4; do
    echo "$(date +%H:%M:%S) [карта 3] семплер family, фолд $f" >> queue_card3.log
    LORA_TAG=_famsmp LORA_SAMPLER=family LORA_SEED=0 \
        $PY -u lora_train.py "$f" 2 > "lora_fold${f}_famsmp.out" 2>&1
done
echo "$(date +%H:%M:%S) [карта 3] семплер: все пять фолдов -> eval_oof_tagged.py _famsmp" >> queue_card3.log
