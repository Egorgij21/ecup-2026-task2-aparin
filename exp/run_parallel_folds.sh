#!/bin/bash
# Параллельный полный OOF: по фолду на карту вместо последовательной очереди.
#
# Один фолд занимает ~2.5 часа. Последовательно пять фолдов это 12 часов, параллельно
# на четырёх картах — два с половиной. При том, что сабмитов пять в день, разница
# между «кандидат к вечеру» и «кандидат завтра» стоит целого дня проверок.
#
# Запуск:
#   CARDS="0 1 2 3" FOLDS="1 2 3 4" LORA_TAG=_gemma <прочие LORA_*> \
#       bash exp/run_parallel_folds.sh
#
# Карты и фолды сопоставляются по порядку. Если фолдов больше, чем карт, лишние
# ждут своей очереди на последней карте — поэтому лучше задавать поровну.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
EPOCHS="${EPOCHS:-2}"
TAG="${LORA_TAG:?нужен LORA_TAG, иначе перезапишутся файлы базовой конфигурации}"
CARDS="${CARDS:?нужен CARDS, например \"0 1 2 3\"}"
FOLDS="${FOLDS:?нужен FOLDS, например \"1 2 3 4\"}"

set -- $CARDS
for f in $FOLDS; do
    card="$1"; shift; [ -z "$card" ] && { echo "карт меньше, чем фолдов"; exit 1; }
    echo "$(date +%H:%M:%S) [карта $card] $TAG фолд $f" >> "queue_par${TAG}.log"
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$TAG" \
        nohup $PY -u lora_train.py "$f" "$EPOCHS" > "lora_fold${f}${TAG}.out" 2>&1 &
    sleep 20                     # разносим загрузку весов, чтобы не толкаться в кэше
done
wait
echo "$(date +%H:%M:%S) [$TAG] параллельные фолды завершены" >> "queue_par${TAG}.log"
