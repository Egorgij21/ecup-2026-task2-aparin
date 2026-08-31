#!/bin/bash
# Третья очередь: чтобы GPU не простаивала под утро, когда combo и семплер доедут.
#
# Адаптер ТОЛЬКО на флам. Основание: в сабмите LoRA применяется исключительно к
# флам-категории, а обучается на обеих. БАД — это 7469 строк из 12971, то есть больше
# половины ёмкости адаптера уходит на данные, которые в бою этим адаптером не
# обслуживаются (БАД в сабмите считает текстовый ансамбль). Плюс на флам всего
# 87 положительных семей, и делить внимание модели там не с чем.
#
# Совмещено с семплером family — по той же причине, по которой собран combo:
# шумовой пол на одном фолде 4 пункта F1, поодиночке эффекты неразличимы.
# Если связка сработает, вклад разложим отдельными прогонами.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python

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
sleep 600                                          # дать очереди семплера стартовать
while [ "$(running)" -gt 0 ]; do sleep 300; done   # ждём полного освобождения

echo "$(date +%H:%M:%S) флам-онли + семплер family, фолд 0" >> queue_night2.log
LORA_TAG=_flamonly LORA_SAMPLER=family LORA_ONLY_CATEGORY="Легковоспламеняющиеся" \
    LORA_SEED=0 $PY -u lora_train.py 0 2 > lora_fold0_flamonly.out 2>&1
echo "$(date +%H:%M:%S) готово" >> queue_night2.log
