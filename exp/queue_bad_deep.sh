#!/bin/bash
# ГЛУБОКАЯ очередь БАД на карты 6,7 — стартует ПОСЛЕ queue_bad_push.sh (ждёт
# освобождения карт) и грузит их на много часов. Всё сохраняется:
# адаптеры lora_fold*_<тег>/, логи .out, OOF .parquet, сводка в этот .log.
#
# Порядок по убыванию ожидания. Замеры — потом на CPU через eval_fold_fixed.py,
# фиксированный порог (как в сабмите), сравнение с _mm1 (одна картинка, в чемпионе).
#
# 3) АНСАМБЛЬ ДВУХ БАЗ с картинками (гемма + Qwen). Декорреляция разных баз — наш
#    ГЛАВНЫЙ рабочий механизм (v10 бленд двух баз дал +0.009). На БАД в связке не
#    пробован. Не хватает Qwen-картинок на фолдах 2,3 — добираем, потом обучаем
#    Qwen-картинки на всех данных для сабмита.
# 4) ДВЕ КАРТИНКИ — промежуточная точка между 1 и 3 кадрами. Если 3 картинки лучше
#    1, интересно, монотонно ли: 2 картинки покажут форму зависимости.
# 5) ПРИЦЕЛЬНЫЙ ПРОМПТ + 3 КАРТИНКИ — комбо двух лучших идей, если обе сработают
#    по отдельности. Полный OOF.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
QT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_bad_deep.log

busy() {  # занята ли карта $1 хоть кем-то
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -cF "$u"
}
wait_cards() { while [ "$(busy 6)" -gt 0 ] || [ "$(busy 7)" -gt 0 ]; do sleep 300; done; }

echo "$(date '+%m-%d %H:%M:%S') глубокая очередь ждёт освобождения карт 6,7" >> $LOG
wait_cards
echo "$(date '+%m-%d %H:%M:%S') карты свободны, старт" >> $LOG

# gen <карта> <фолд> <тег> <база> <targets> <images> <maxlen> <bs> <acc> [PROMPT_VARIANT]
gen() {
    local card=$1 fold=$2 tag=$3 base=$4 tgt=$5 img=$6 ml=$7 bs=$8 acc=$9 pv=${10:-}
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] $tag фолд $fold (img=$img base=$base pv=$pv)" >> $LOG
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_IMAGES="$img" LORA_MAX_PIXELS=261120 LORA_MAXLEN="$ml" LORA_BS="$bs" LORA_ACC="$acc" \
        LORA_BASE_MODEL="$base" LORA_TARGETS="$tgt" ${pv:+PROMPT_VARIANT=$pv} \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|Легковосп" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}

# --- 3) ансамбль: добить Qwen-картинки фолды 2,3, затем на всех данных ---
( gen 6 2 _mmqwen Qwen/Qwen3.5-4B "$QT" 1 1900 2 16
  gen 6 3 _mmqwen Qwen/Qwen3.5-4B "$QT" 1 1900 2 16 ) &
( gen 7 all _mmqwenall Qwen/Qwen3.5-4B "$QT" 1 1900 2 16 ) &
wait
echo "$(date '+%m-%d %H:%M:%S') Qwen-картинки: полный OOF + all-data готовы" >> $LOG

# --- 4) две картинки, полный OOF ---
( gen 6 0 _mm2 google/gemma-4-E4B-it "$GT" 2 1800 2 16
  gen 6 1 _mm2 google/gemma-4-E4B-it "$GT" 2 1800 2 16
  gen 6 2 _mm2 google/gemma-4-E4B-it "$GT" 2 1800 2 16 ) &
( gen 7 3 _mm2 google/gemma-4-E4B-it "$GT" 2 1800 2 16
  gen 7 4 _mm2 google/gemma-4-E4B-it "$GT" 2 1800 2 16 ) &
wait
echo "$(date '+%m-%d %H:%M:%S') две картинки: полный OOF готов" >> $LOG

# --- 5) прицельный промпт + 3 картинки, полный OOF ---
( gen 6 0 _mm3tp google/gemma-4-E4B-it "$GT" 3 2400 1 32 badimg
  gen 6 1 _mm3tp google/gemma-4-E4B-it "$GT" 3 2400 1 32 badimg
  gen 6 2 _mm3tp google/gemma-4-E4B-it "$GT" 3 2400 1 32 badimg ) &
( gen 7 3 _mm3tp google/gemma-4-E4B-it "$GT" 3 2400 1 32 badimg
  gen 7 4 _mm3tp google/gemma-4-E4B-it "$GT" 3 2400 1 32 badimg ) &
wait
echo "$(date '+%m-%d %H:%M:%S') ГЛУБОКАЯ ОЧЕРЕДЬ ЗАВЕРШЕНА" >> $LOG
