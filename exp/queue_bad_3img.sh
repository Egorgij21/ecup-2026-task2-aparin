#!/bin/bash
# 3 КАРТИНКИ на БАД + badimg — НОВЫЙ сигнал (не рекомбинация). Мотивация: номер СГР и
# «не является лекарственным средством» — на ОБОРОТЕ упаковки; v20 подаёт 1 картинку
# (перёд) и их не видит. Прошлый ночной запуск _mm3/_mm3tp УПАЛ на баге bash
# (PROMPT_VARIANT после команды) — то есть 3 картинки НИ РАЗУ не обучались честно.
# Отличие от v20 РОВНО одно: IMAGES 1->3. Замер: eval_fold_fixed.py <fold> _mmtp _mm3tp2.
#
# gemma: фикс. 280 токенов/картинку => 3 карт ~840 токенов, MAXLEN 2400, BS1/ACC32.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_bad_3img.log

busy() {
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -cF "$u"
}
wait_card() { while [ "$(busy "$1")" -gt 0 ]; do sleep 120; done; }

train() {  # <карта> <фолд> <тег>
    local card=$1 fold=$2 tag=$3
    wait_card "$card"
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] $tag фолд $fold" >> $LOG
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_IMAGES=3 LORA_MAX_PIXELS=261120 LORA_MAXLEN=2400 LORA_BS=1 LORA_ACC=32 \
        PROMPT_VARIANT=badimg LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|Легковосп|покрытие" "lora_fold${fold}${tag}.out" | tail -3 >> $LOG
}

echo "$(date '+%m-%d %H:%M:%S') СТАРТ 3 картинки БАД (5 фолдов + all-data)" >> $LOG
( train 0 0 _mm3tp2 ) & ( train 1 1 _mm3tp2 ) & ( train 2 2 _mm3tp2 ) &
( train 3 3 _mm3tp2 ) & ( train 4 4 _mm3tp2 ) & ( train 6 all _mm3tp2all ) &
wait
echo "$(date '+%m-%d %H:%M:%S') 3 КАРТИНКИ БАД ЗАВЕРШЕНЫ" >> $LOG
echo "  замер: $PY eval_fold_fixed.py <fold> _mmtp _mm3tp2" >> $LOG
