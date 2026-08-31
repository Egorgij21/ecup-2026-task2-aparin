#!/bin/bash
# Очередь после текущих обучений. Гипотезы меряем на ОДНОМ fold 0
# (адаптер учится на фолдах 1-4, оценивается на невиданном) — 2.7ч вместо 13.5ч.
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

while [ "$(running)" -gt 0 ]; do sleep 120; done

# Гипотеза 0: ранг упёрт. Спектр ДЕЛЬТ показал эффективный ранг 16 из 16 у 73-80%
# модулей, хвост 0.24 от ведущего. Эпохи держим = 2, чтобы отделить эффект ранга.
echo "$(date +%H:%M:%S) ранг 32, 2 эпохи" >> queue_next.log
LORA_TAG=_r32 LORA_RANK=32 LORA_SEED=0 $PY -u lora_train.py 0 2 > lora_fold0_r32.out 2>&1

# Гипотеза 0.5: Qwen3.5 гибридный — обычное внимание лишь в 8 слоях из 32, а в
# остальных 24 линейное внимание с in_proj_qkv/out_proj/in_proj_z. Сейчас они НЕ
# задеты вовсе: 1.01 млрд параметров (28% линейных весов) вне адаптации.
# in_proj_a/in_proj_b (выход 32) не берём — там r=16 это половина полного ранга,
# а параметры управляют затуханием.
echo "$(date +%H:%M:%S) + модули линейного внимания" >> queue_next.log
LORA_TAG=_linattn LORA_SEED=0 \
    LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj,in_proj_z" \
    $PY -u lora_train.py 0 2 > lora_fold0_linattn.out 2>&1

# Гипотеза 1: кривая по эпохам не вышла на плато (1->2 дало +4.5 пункта на флам)
echo "$(date +%H:%M:%S) 3 эпохи" >> queue_next.log
LORA_TAG=_ep3 LORA_SEED=0 $PY -u lora_train.py 0 3 > lora_fold0_ep3.out 2>&1

# Гипотеза 2: в сабмите LoRA работает ТОЛЬКО на флам, а учится на обеих
# категориях — половина ёмкости уходит на данные, которые в бою не используются
echo "$(date +%H:%M:%S) только флам, 3 эпохи" >> queue_next.log
LORA_TAG=_flamonly LORA_SEED=0 LORA_ONLY_CATEGORY=Легковоспламеняющиеся \
    $PY -u lora_train.py 0 3 > lora_fold0_flamonly.out 2>&1

echo "$(date +%H:%M:%S) очередь готова" >> queue_next.log
