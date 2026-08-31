#!/bin/bash
# Обучение БАД-адаптеров с УТОЧНЁННЫМ промптом на картинках, полный OOF (5 фолдов).
# Мотивация — error_analysis_bad.py: клауза «не БАД если спортпит» конфликтует с
# разметкой (спортпит ~55% БАД), порождая спортпит-кластер пропусков. Проверяем два
# варианта промпта против чемпиона v20 (_mmtp):
#   noexcl   — убраны исключения, только позитивные признаки добавки;
#   sportbad — спортпит явно отнесён к БАД.
# Конфиг ИДЕНТИЧЕН адаптеру v20 (queue_bad_push.sh runp): gemma, 1 картинка,
# MAX_PIXELS 261120, MAXLEN 1600, BS2/ACC16 — чтобы отличие было РОВНО в промпте.
# Замер потом: eval_fold_fixed.py <fold> _mmtp _mmnoexcl _mmsport (фикс. порог 0.47).
#
# Никаких молчаливых фолбэков: при отсутствии картинок/базы lora_train падает.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_bad_prompt.log

busy() {
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -cF "$u"
}
wait_card() { while [ "$(busy "$1")" -gt 0 ]; do sleep 120; done; }

# train <карта> <фолд> <тег> <PROMPT_VARIANT>
train() {
    local card=$1 fold=$2 tag=$3 pv=$4
    wait_card "$card"
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] $tag фолд $fold (pv=$pv)" >> $LOG
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_IMAGES=1 LORA_MAX_PIXELS=261120 LORA_MAXLEN=1600 LORA_BS=2 LORA_ACC=16 \
        PROMPT_VARIANT="$pv" LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|Легковосп" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}

echo "$(date '+%m-%d %H:%M:%S') СТАРТ очереди БАД-промптов" >> $LOG

# 10 прогонов на 7 карт (0,1,2,3,4,6,7; карта 5 — соседний трек, не трогаем).
# noexcl: фолды 0-4 на картах 0,1,2,3,4 ; sportbad: фолды 0-4 на 6,7,0,1,2 (вторая волна).
( train 0 0 _mmnoexcl noexcl ; train 0 3 _mmsport  sportbad ) &
( train 1 1 _mmnoexcl noexcl ; train 1 4 _mmsport  sportbad ) &
( train 2 2 _mmnoexcl noexcl ; train 2 0 _mmsport  sportbad ) &
( train 3 3 _mmnoexcl noexcl ) &
( train 4 4 _mmnoexcl noexcl ) &
( train 6 1 _mmsport  sportbad ) &
( train 7 2 _mmsport  sportbad ) &
wait
echo "$(date '+%m-%d %H:%M:%S') ОЧЕРЕДЬ БАД-ПРОМПТОВ ЗАВЕРШЕНА" >> $LOG
echo "  замер: $PY eval_fold_fixed.py <fold> _mmtp _mmnoexcl _mmsport" >> $LOG
