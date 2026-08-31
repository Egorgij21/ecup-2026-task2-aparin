#!/bin/bash
# noexcl и sportbad на ВСЕХ данных для сабмитов v22/v23. Конфиг = адаптер v20 (_mmtpall):
# gemma+картинки, MAXLEN 1600, BS2/ACC16, отличие только PROMPT_VARIANT.
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
run() {  # <карта> <тег> <PROMPT_VARIANT>
    CUDA_VISIBLE_DEVICES=$1 LORA_TAG="$2" LORA_SEED=0 \
        LORA_IMAGES=1 LORA_MAX_PIXELS=261120 LORA_MAXLEN=1600 LORA_BS=2 LORA_ACC=16 \
        PROMPT_VARIANT="$3" LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
        ../.venv/bin/python -u lora_train.py all 2 > "lora_foldall_$2.out" 2>&1
    echo "DONE $2" >> "lora_foldall_$2.out"
}
( run 1 _noexclall noexcl ) &
( run 2 _sportall sportbad ) &
wait
echo "BAD ALLDATA DONE"
