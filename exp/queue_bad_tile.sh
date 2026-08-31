#!/bin/bash
# ТАЙЛИНГ на БАД: 1 картинку -> 2×2=4 тайла (4×280 токенов, 4× разрешение на мелком
# тексте). Отличие от v20 РОВНО одно: LORA_TILE 1->2. Замер: eval_fold_fixed <fold> _mmtp _mmtile.
# Старты РАЗНЕСЕНЫ на 30с: 6 одновременных тяжёлых стартов ранее тихо умирали.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_bad_tile.log
train() {
    local card=$1 fold=$2 tag=$3
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] $tag фолд $fold" >> $LOG
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_IMAGES=1 LORA_TILE=2 LORA_MAX_PIXELS=261120 LORA_MAXLEN=2200 LORA_BS=1 LORA_ACC=32 \
        PROMPT_VARIANT=badimg LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|покрытие" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}
echo "$(date '+%m-%d %H:%M:%S') СТАРТ тайлинг БАД (staggered)" >> $LOG
( train 0 0 _mmtile ) & sleep 30
( train 1 1 _mmtile ) & sleep 30
( train 2 2 _mmtile ) & sleep 30
( train 3 3 _mmtile ) & sleep 30
( train 4 4 _mmtile ) & sleep 30
( train 6 all _mmtileall ) &
wait
echo "$(date '+%m-%d %H:%M:%S') ТАЙЛИНГ БАД ЗАВЕРШЁН" >> $LOG
