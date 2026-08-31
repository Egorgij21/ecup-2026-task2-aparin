#!/bin/bash
# Полный OOF для ОДНОГО кандидата: пять фолдов подряд, ~25 часов.
#
# Зачем отдельным скриптом. Фолд 0 даёт 40 позитивов флам, и разброс между двумя
# прогонами ОДНОЙ конфигурации там 7 пунктов PR. Различить конфигурации на нём
# нельзя — проверено парой база/сид-777. Поэтому решение принимается так: по фолду 0
# отбирается один кандидат, и только он идёт на полный OOF, где и меряется настоящая
# вложенная метрика, сравнимая с 0.8996 у чемпиона.
#
# Запуск:
#   LORA_TAG=_combo LORA_RANK=32 LORA_TARGETS="..." bash exp/queue_full_oof.sh 3
# Первый аргумент — число эпох. Переменные LORA_* пробрасываются как есть, поэтому
# конфигурация задаётся ровно теми же переменными, что и в отборочном прогоне —
# иначе полный OOF померяет не то, что победило.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
EPOCHS="${1:-2}"
TAG="${LORA_TAG:?нужен LORA_TAG, иначе перезапишутся файлы базовой конфигурации}"

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

echo "$(date +%H:%M:%S) полный OOF: тег=$TAG эпох=$EPOCHS" >> queue_full_oof.log
for f in 0 1 2 3 4; do
    while [ "$(running)" -gt 0 ]; do sleep 300; done
    echo "$(date +%H:%M:%S)   фолд $f" >> queue_full_oof.log
    $PY -u lora_train.py "$f" "$EPOCHS" > "lora_fold${f}${TAG}.out" 2>&1
done
echo "$(date +%H:%M:%S) все пять фолдов готовы" >> queue_full_oof.log
echo "дальше: exp/eval_full_oof.py на файлах lora_oof_fold[0-4]${TAG}.parquet" >> queue_full_oof.log
