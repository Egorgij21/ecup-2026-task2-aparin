#!/bin/bash
# Очередь на карты 6 и 7 — эксперименты по БАД в порядке приоритета.
# Всё сохраняется: адаптеры (lora_fold*_<тег>/), логи (.out), OOF-скоры (.parquet).
# Замеры делаются потом на CPU через exp/eval_fold_fixed.py — фиксированный порог,
# как в сабмите.
#
# ПОЧЕМУ ТОЛЬКО БАД. Пять улучшений подряд — все от картинок на БАД. БАД на паблике
# измеряется точно (~920 строк, ~685 позитивов) и трижды совпал с локальным OOF
# (v13, v16, v17). Флам насыщен и на паблике чистый шум — туда не бьём.
#
# ПРИОРИТЕТ 1: ТРИ КАРТИНКИ. Гемма видит одну картинку (фикс. 280 токенов на кадр,
#   разрешение поднять нельзя). Маркировка «не является лекарственным средством»
#   и номер СГР — на ОБОРОТЕ упаковки, то есть на 2-3 фото. Единственная линия
#   с НОВОЙ информацией. Ночью падала на лимите токенов — ставим MAXLEN 2400.
#   5 фолдов, сравнение с _mm1 (одна картинка) на тех же фолдах.
#
# ПРИОРИТЕТ 2: ПРИЦЕЛЬНЫЙ ПРОМПТ БАД + картинки. На хлопушках прицельный промпт
#   («прочитай на упаковке: пиротехническое изделие, класс опасности») дал +0.12 AUC
#   против общего. Для БАД аналог живёт в exp/deps/prompts_bad_targeted.py и
#   включается PROMPT_VARIANT=badimg. В обучении с картинками не пробован.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_bad_push.log

run() {   # run <карта> <фолд> <тег> <доп-переменные...>
    local card=$1 fold=$2 tag=$3; shift 3
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] $tag фолд $fold ($*)" >> $LOG
    env "$@" CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_IMAGES=3 LORA_MAX_PIXELS=261120 LORA_MAXLEN=2400 LORA_BS=1 LORA_ACC=32 \
        LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|Легковосп" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}

# --- ПРИОРИТЕТ 1: три картинки, полный OOF. Карта 6: фолды 0,1,2. Карта 7: 3,4 ---
( run 6 0 _mm3 ; run 6 1 _mm3 ; run 6 2 _mm3 ) &
( run 7 3 _mm3 ; run 7 4 _mm3 ) &
wait
echo "$(date '+%m-%d %H:%M:%S') ТРИ КАРТИНКИ: полный OOF готов" >> $LOG
echo "  замер: $PY eval_fold_fixed.py <fold> _mm1 _mm3" >> $LOG

# --- ПРИОРИТЕТ 2: прицельный промпт БАД + 1 картинка, полный OOF ---
runp() {  # то же, но 1 картинка и прицельный промпт
    local card=$1 fold=$2 tag=$3
    echo "$(date '+%m-%d %H:%M:%S') [карта $card] $tag фолд $fold (прицельный промпт БАД)" >> $LOG
    CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_IMAGES=1 LORA_MAX_PIXELS=261120 LORA_MAXLEN=1600 LORA_BS=2 LORA_ACC=16 \
        PROMPT_VARIANT=badimg LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|Легковосп" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}
( runp 6 0 _mmtp ; runp 6 1 _mmtp ; runp 6 2 _mmtp ) &
( runp 7 3 _mmtp ; runp 7 4 _mmtp ) &
wait
echo "$(date '+%m-%d %H:%M:%S') ПРИЦЕЛЬНЫЙ ПРОМПТ: полный OOF готов" >> $LOG
echo "  замер: $PY eval_fold_fixed.py <fold> _mm1 _mmtp" >> $LOG
echo "$(date '+%m-%d %H:%M:%S') ОЧЕРЕДЬ ЗАВЕРШЕНА" >> $LOG
