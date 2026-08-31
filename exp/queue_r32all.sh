#!/bin/bash
# Qwen r32 на ВСЕХ данных — сабмитный артефакт.
#
# Зачем отдельно от lora_fold0_r32, который уже упакован в submit_qwen_r32_v9.zip.
# Тот обучен на фолде 0, то есть на 80% данных, а чемпион — на 100%. Сабмит из него
# отличается от чемпиона ДВУМЯ вещами сразу: рангом и объёмом обучающих данных.
# Проиграет — и непонятно, что виновато. Этот прогон убирает вторую переменную,
# после него сравнение с чемпионом становится чистым: только ранг 16 против 32.
#
# Приоритет высокий: ставится первым, как только освободится карта, раньше
# gemma-r32 и флам-онли из queue_night3.sh (они ждут полного освобождения и
# подхватятся следом).
#
# 2 эпохи, а не 3: у r32 замер на чекпоинтах дал 0.8097 после первой эпохи и 0.9229
# после второй, третьей у него не было. Ставить непроверенное число эпох в сабмитный
# артефакт нельзя — третью эпоху сейчас проверяет combo.
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
while [ "$(running)" -gt 1 ]; do sleep 180; done   # стартуем, как только останется одна задача

echo "$(date +%H:%M:%S) Qwen r32 на всех данных, 2 эпохи" >> queue_r32all.log
LORA_TAG=_r32 LORA_RANK=32 LORA_SEED=0 \
    $PY -u lora_train.py all 2 > lora_foldall_r32.out 2>&1
echo "$(date +%H:%M:%S) готово -> exp/lora_foldall_r32" >> queue_r32all.log
