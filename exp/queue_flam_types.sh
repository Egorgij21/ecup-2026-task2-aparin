#!/bin/bash
# ФЛАМ-types адаптер для БЛЕНДА ПРОМПТОВ (идея: декорреляция по промптам, как v10 по
# базам). Zero-shot types вырожден (AUC 0.75), но декоррелирован с базой (0.727 против
# 0.734 между базами) — обучение должно калибровать негативы. Проверяем, даёт ли
# ОБУЧЕННЫЙ types-адаптер декоррелированный НЕ вырожденный сигнал: блендим с базовыми
# флам-адаптерами ("" Qwen, "_gemma") и смотрим item-level спасение газовых горелок.
#
# Конфиг = как у базового флам gemma адаптера: текст (картинки на флам вредят −0.010),
# обе категории, дефолтные MAXLEN/BS/LR. Отличие от базового РОВНО одно: флам-промпт
# (PROMPT_VARIANT=flamtypes). Стартует ПОСЛЕ освобождения карт от очереди БАД.
# Замер: eval_fold_fixed / flam_blend_trained.py при фикс. пороге 0.45.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_flam_types.log

busy() {
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -cF "$u"
}
wait_card() { while [ "$(busy "$1")" -gt 0 ]; do sleep 120; done; }

train() {  # train <карта> <фолд>
    local card=$1 fold=$2
    wait_card "$card"
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] _flamtypes фолд $fold" >> $LOG
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="_flamtypes" LORA_SEED=0 \
        LORA_IMAGES=0 PROMPT_VARIANT=flamtypes \
        LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}_flamtypes.out" 2>&1
    grep -aE "Легковосп|БАД:" "lora_fold${fold}_flamtypes.out" | tail -2 >> $LOG
}

# сначала дожидаемся ПОЛНОГО конца очереди БАД — иначе гонка за карты (обе очереди
# ждут одни и те же карты и могут схватить одну одновременно)
echo "$(date '+%m-%d %H:%M:%S') flam-types ждёт завершения очереди БАД" >> $LOG
while ! grep -qa "ОЧЕРЕДЬ БАД-ПРОМПТОВ ЗАВЕРШЕНА" queue_bad_prompt.log 2>/dev/null; do sleep 300; done
echo "$(date '+%m-%d %H:%M:%S') очередь БАД завершена, старт flam-types" >> $LOG
( train 0 0 ) & ( train 1 1 ) & ( train 2 2 ) & ( train 3 3 ) & ( train 4 4 ) &
wait
echo "$(date '+%m-%d %H:%M:%S') FLAM-TYPES ЗАВЕРШЁН" >> $LOG
echo "  замер: $PY flam_blend_trained.py" >> $LOG
