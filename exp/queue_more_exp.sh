#!/bin/bash
# Очередь экспериментов на те же ЧЕТЫРЕ карты (0, 1, 6, 7), строго после текущих.
# Других карт не занимать.
#
# ПОРЯДОК ПО ОЖИДАЕМОЙ ОТДАЧЕ, а не по дешевизне:
#
# 1) ТРИ КАРТИНКИ вместо одной. Главный кандидат, и вот почему.
#    У gemma визуальный бюджет на КАРТИНКУ жёстко фиксирован — 280 «мягких» токенов,
#    переопределить нельзя (проверено: процессор игнорирует аргумент size).
#    Значит единственный способ дать модели больше визуальной информации —
#    подать больше КАДРОВ. А их есть: в среднем 3.81 на товар, у 84-89% товаров
#    два и больше. Маркировка БАД («не является лекарственным средством», номер СГР)
#    почти всегда на ОБОРОТЕ упаковки, то есть на втором-третьем фото, которого
#    модель сейчас не видит вовсе.
#
# 2) FOCAL LOSS поверх мультимодального. focal сделал лучший флам-адаптер за всё
#    время (F1 адаптера 0.9167 против 0.8302 у базы) и дал +0.0024 на БАД.
#    С картинками не пробовался.
#
# 3) АУГМЕНТАЦИЯ КАРТИНОК. Раньше была бессмысленна — модель картинок не видела.
#    Теперь видит, и отражения/обрезки/цвет становятся осмысленной регуляризацией.
#    Реализуется отдельно, здесь только заготовка под LORA_IMG_AUG.
#
# 4) СЕМПЛЕР family поверх мультимодального. На тексте провалился (0.8689 против
#    0.8997), но чинил измеренную поломку; с картинками расклад мог измениться.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
LOG=queue_more_exp.log

free_card() {   # ждём, пока карта $1 освободится ОТ ЛЮБОГО процесса
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
    while [ "$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -cF "$u")" -gt 0 ]; do
        sleep 180
    done
}

run() {   # run <карта> <фолд> <тег> <доп. переменные...>
    local card=$1 fold=$2 tag=$3; shift 3
    free_card "$card"
    echo "$(date +%H:%M:%S) [карта $card] $tag фолд $fold ($*)" >> $LOG
    env "$@" CUDA_VISIBLE_DEVICES="$card" LORA_TAG="$tag" LORA_SEED=0 \
        LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "Легковоспламеняющиеся:|БАД:" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}

# --- карты 6 и 7: три картинки, два фолда (сравнение с _mm1 на тех же фолдах) ---
( run 6 4 _mm3 LORA_IMAGES=3 LORA_MAXLEN=2000 LORA_BS=1 LORA_ACC=32
  run 6 4 _mmfocal LORA_IMAGES=1 LORA_MAXLEN=1600 LORA_BS=2 LORA_ACC=16 LORA_LOSS=focal ) &
( run 7 0 _mm3 LORA_IMAGES=3 LORA_MAXLEN=2000 LORA_BS=1 LORA_ACC=32
  run 7 4 _mmsmp LORA_IMAGES=1 LORA_MAXLEN=1600 LORA_BS=2 LORA_ACC=16 LORA_SAMPLER=family ) &
wait
echo "$(date +%H:%M:%S) очередь экспериментов завершена" >> $LOG
