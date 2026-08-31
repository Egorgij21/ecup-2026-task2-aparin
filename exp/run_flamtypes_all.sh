#!/bin/bash
cd /workspace/counter/exp || exit 1
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
CUDA_VISIBLE_DEVICES=0 LORA_TAG="_flamtypesall" LORA_SEED=0 \
    LORA_IMAGES=0 PROMPT_VARIANT=flamtypes \
    LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GT" \
    ../.venv/bin/python -u lora_train.py all 2 > lora_foldall_flamtypes.out 2>&1
echo "FLAMTYPES ALL DONE" >> lora_foldall_flamtypes.out
