#!/bin/bash
# Условная очередь на ночь: дожидается мультимодального прогона и контроля,
# САМА сравнивает их по заранее заданному критерию и запускает то, что осмысленно
# при этом исходе. Иначе к утру был бы либо простой карт, либо посчитанное впустую.
#
# Критерий (записан до запуска): картинки полезны, если F1 БЛЕНДА при фиксированном
# пороге на фолде 4 выше контроля на +0.01. Меньше — не окупает роста инференса.
#
# Ветка А (картинки помогли): фолды 0 и 1 с картинками — движение к полному OOF.
# Ветка Б (не помогли): варианты ПРОМПТА ОБУЧЕНИЯ — он не менялся ни разу с первого
#   прогона, а формулировка задачи у VLM давала +0.12 AUC, то есть влияет сильно.
#
# Обе ветки занимают ровно карты 6 и 7, других не трогают.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_night_auto.log

busy() {   # занята ли карта $1 моим процессом
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -cF "$u"
}
wait_free() { while [ "$(busy 6)" -gt 0 ] || [ "$(busy 7)" -gt 0 ]; do sleep 180; done; }

echo "$(date +%H:%M:%S) жду окончания mm1 и ctrl" >> $LOG
wait_free
echo "$(date +%H:%M:%S) обе карты свободны, сравниваю" >> $LOG

$PY eval_fold_fixed.py 4 _ctrl _mm1 > cmp_mm1.log 2>&1
cat cmp_mm1.log >> $LOG

# берём машинно-читаемые строки MACHINE<таб>категория<таб>тег<таб>F1<таб>порог,
# по флам-категории. Разбор глазами и колонками уже подводил.
CTRL=$(grep -P "^MACHINE\tЛегков.*\t_ctrl\t" cmp_mm1.log | cut -f4 | head -1)
MM=$(grep -P "^MACHINE\tЛегков.*\t_mm1\t" cmp_mm1.log | cut -f4 | head -1)
echo "$(date +%H:%M:%S) контроль=$CTRL мультимодальный=$MM" >> $LOG

WIN=$($PY -c "
try:
    c,m=float('${CTRL:-0}'),float('${MM:-0}')
    print(1 if m-c>=0.01 else 0)
except Exception:
    print(0)")

if [ "$WIN" = "1" ]; then
    echo "$(date +%H:%M:%S) ВЕТКА А: картинки помогли, считаю фолды 0 и 1" >> $LOG
    for pair in "6:0" "7:1"; do
        card=${pair%%:*}; f=${pair##*:}
        CUDA_VISIBLE_DEVICES=$card LORA_IMAGES=1 LORA_BS=2 LORA_ACC=16 LORA_TAG=_mm1 \
            LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" LORA_SEED=0 \
            nohup $PY -u lora_train.py "$f" 2 > "lora_fold${f}_mm1.out" 2>&1 &
        sleep 20
    done
    wait
    echo "$(date +%H:%M:%S) фолды 0 и 1 с картинками готовы" >> $LOG
else
    echo "$(date +%H:%M:%S) ВЕТКА Б: картинки не помогли, проверяю промпт обучения" >> $LOG
    # промпт меняется через переменную окружения, читаемую в src/prompts.py
    for pair in "6:short" "7:checklist"; do
        card=${pair%%:*}; v=${pair##*:}
        CUDA_VISIBLE_DEVICES=$card PROMPT_VARIANT=$v LORA_TAG="_pr$v" \
            LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" LORA_SEED=0 \
            nohup $PY -u lora_train.py 4 2 > "lora_fold4_pr$v.out" 2>&1 &
        sleep 20
    done
    wait
    echo "$(date +%H:%M:%S) варианты промпта готовы" >> $LOG
fi
echo "$(date +%H:%M:%S) очередь завершена" >> $LOG
