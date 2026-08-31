#!/bin/bash
# Семплирование по СЕМЬЯМ вместо взвешивания лосса.
#
# Основание — два замера, не догадки:
#   1) при shuffle=True на эффективный батч (BS 4 x ACC 8 = 32) приходится 0.48
#      позитива флам: половина шагов оптимизатора не видит целевой класс вообще.
#      С семплером family — 6.8 штук на батч.
#   2) разбор ошибок: 0 из 30 пропусков имели свою семью среди позитивов
#      train-фолда. Модель валится ровно на НОВЫХ семьях, значит учить надо
#      разнообразию семей, а не количеству строк.
#
# Конфигурация в остальном строго базовая (r=16, 2 эпохи, те же модули), чтобы
# эффект семплера не смешался с эффектом ранга и эпох. Фолд 0 — есть с чем сравнить.
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
while [ "$(running)" -gt 1 ]; do sleep 120; done   # ждём, пока r32 доедет

echo "$(date +%H:%M:%S) семплер family, фолд 0" >> queue_sampler.log
LORA_TAG=_famsmp LORA_SAMPLER=family LORA_SEED=0 \
    $PY -u lora_train.py 0 2 > lora_fold0_famsmp.out 2>&1
echo "$(date +%H:%M:%S) готово" >> queue_sampler.log
