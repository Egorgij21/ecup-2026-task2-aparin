#!/bin/bash
# Ночная очередь: дообучаем LoRA на всех фолдах, чтобы получить ЧЕСТНЫЙ OOF.
#
# Сейчас оценка LoRA держится на fold0 (40 позитивов флам) — этого мало для решений.
# Полный OOF даст: (а) надёжную цифру по кросс-валидации, (б) возможность честно
# стекать LoRA с ансамблем, потому что для стекинга нужны out-of-fold скоры.
#
# Параллелим по 2 задачи: одна задача не насыщает H100 (batch 4, seq ~700),
# две дают заметно больший суммарный throughput при 16 ГБ каждая.

cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
MAX_PARALLEL=2

# Воркеры DataLoader наследуют ту же командную строку, поэтому одна задача видна
# как три процесса. Считаем только головные — те, чей родитель сам не обучение.
# Считаем только головные процессы обучения:
#   * воркеры DataLoader наследуют ту же командную строку -> отсекаем по родителю;
#   * посторонние шеллы могут содержать паттерн в своей cmdline -> отсекаем по comm.
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

wait_slot() {
    while [ "$(running)" -ge "$MAX_PARALLEL" ]; do sleep 60; done
}

echo "$(date +%H:%M:%S) старт очереди, уже запущено: $(running)" >> night_queue.log

for f in 1 2 3 4; do
    if [ -f "lora_oof_fold${f}.parquet" ]; then
        echo "$(date +%H:%M:%S) fold${f}: OOF уже есть, пропускаю" >> night_queue.log
        continue
    fi
    wait_slot
    echo "$(date +%H:%M:%S) запускаю fold${f}" >> night_queue.log
    nohup $PY -u lora_train.py "$f" 2 > "lora_fold${f}.out" 2>&1 &
    sleep 120          # разносим старты, чтобы пики загрузки весов не совпадали
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M:%S) все фолды готовы" >> night_queue.log

# Полный OOF собран — считаем честную вложенную оценку со всеми признаками
$PY -u eval_full_oof.py > eval_full_oof.log 2>&1
echo "$(date +%H:%M:%S) итоговая оценка посчитана" >> night_queue.log
