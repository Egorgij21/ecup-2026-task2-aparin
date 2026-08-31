#!/bin/bash
# Qwen-КАРТИНКИ на БАД — декорреляция ПО БАЗАМ (единственный подтверждённый рычаг:
# флам две базы дали +0.009 при corr 0.5; промпты декоррелируют слабо, corr 0.97).
# Блендим с gemma-картинками БАД (v20). БАД измерим (CI ±0.009, совпадает с пабликом).
#
# ФИКС (25.08): Qwen3.5-4B как AutoModelForCausalLM — ТЕКСТОВЫЙ (нет vision, forward
# без pixel_values), картинки молча игнорировались. load_base теперь при IMAGES>0
# выбирает AutoModelForImageTextToText (там model.language_model.layers + vision).
# Regex модулей покрывает linear_attn (24 гибридных слоя) + self_attn (8) + mlp (32).
# Проверено: loss 2.94->0.38 за 40 шагов (здоровый, картинки работают).
# Промпт badimg — как у gemma-адаптера v20 (отличие бленда РОВНО в базе).
#
# ~4.5ч/фолд (Qwen динамическое разрешение). Замер: eval_fold_fixed.py <fold> _mm1 _mmqwen2
# и бленд gemma+Qwen картинок на БАД. Карта 5 — соседний трек, не трогаем.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
QT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|out_proj)|mlp\.(gate|up|down)_proj)'
LOG=queue_qwen_mm.log

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
        LORA_IMAGES=1 LORA_MAX_PIXELS=261120 LORA_MAXLEN=1900 LORA_BS=2 LORA_ACC=16 \
        PROMPT_VARIANT=badimg LORA_BASE_MODEL=Qwen/Qwen3.5-4B LORA_TARGETS="$QT" \
        $PY -u lora_train.py "$fold" 2 > "lora_fold${fold}${tag}.out" 2>&1
    grep -aE "БАД:|Легковосп" "lora_fold${fold}${tag}.out" | tail -2 >> $LOG
}

echo "$(date '+%m-%d %H:%M:%S') СТАРТ Qwen-картинки БАД (5 фолдов + all-data)" >> $LOG
( train 0 0 _mmqwen2 ) &
( train 1 1 _mmqwen2 ) &
( train 2 2 _mmqwen2 ) &
( train 3 3 _mmqwen2 ) &
( train 4 4 _mmqwen2 ) &
( train 6 all _mmqwen2all ) &
wait
echo "$(date '+%m-%d %H:%M:%S') Qwen-КАРТИНКИ БАД ЗАВЕРШЕНЫ" >> $LOG
echo "  замер: $PY eval_fold_fixed.py <fold> _mm1 _mmqwen2  и bad_base_blend_eval.py" >> $LOG
