#!/bin/bash
# Последовательная очередь на остаток ночи. Строго по одной задаче за раз:
# combo и famsmp уже делят карту пополам, третий параллельный прогон замедлил бы всё.
#
# (1) gemma + ранг 32, 3 эпохи. На фолде 0 гемма дала лучший бленд (PR 0.8173 против
#     0.7540 у базы), а ранг 32 — лучший одиночный адаптер (PR 0.9229). Комбинация
#     напрашивается. Три эпохи с чекпоинтами по эпохам заодно закрывают вопрос
#     «сколько эпох нужно» на этой конфигурации без отдельных прогонов.
#     ВАЖНО: у геммы модули задаются регуляркой строго по языковой части — вне её
#     лежат Gemma4ClippableLinear, которые peft не поддерживает.
# (2) адаптер только на флам + семплер family. В сабмите LoRA обслуживает лишь флам,
#     а обучается на обеих категориях: больше половины ёмкости уходит на БАД.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GEMMA_TARGETS='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'

running() {
    local pids real="" n=0 ppid comm
    pids=$(pgrep -f "lora_train.py" 2>/dev/null)
    for p in $pids; do
        comm=$(cat "/proc/$p/comm" 2>/dev/null)
        case "$comm" in python*) real="$real $p" ;; esac
    done
    for p in $real; do
        ppid=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
        echo "$real" | grep -qw "$ppid" || n=$((n + 1))
    done
    echo "$n"
}
wait_free() { sleep 300; while [ "$(running)" -gt 0 ]; do sleep 300; done; }

wait_free
echo "$(date +%H:%M:%S) gemma + ранг 32, фолд 0, 3 эпохи" >> queue_night3.log
LORA_TAG=_gemmar32 LORA_RANK=32 LORA_BASE_MODEL=google/gemma-4-E4B-it \
    LORA_TARGETS="$GEMMA_TARGETS" LORA_SEED=0 \
    $PY -u lora_train.py 0 3 > lora_fold0_gemmar32.out 2>&1
echo "$(date +%H:%M:%S) gemma r32 готово" >> queue_night3.log

wait_free
echo "$(date +%H:%M:%S) флам-онли + семплер family, фолд 0" >> queue_night3.log
LORA_TAG=_flamonly LORA_SAMPLER=family LORA_ONLY_CATEGORY="Легковоспламеняющиеся" \
    LORA_SEED=0 $PY -u lora_train.py 0 2 > lora_fold0_flamonly.out 2>&1
echo "$(date +%H:%M:%S) флам-онли готово" >> queue_night3.log
