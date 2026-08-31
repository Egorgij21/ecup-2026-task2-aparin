#!/bin/bash
cd /workspace/counter/exp
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QT='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|out_proj)|mlp\.(gate|up|down)_proj)'
CUDA_VISIBLE_DEVICES=7 LORA_TAG="_mmqwen2all_s1" LORA_SEED=1 \
    LORA_IMAGES=1 LORA_MAX_PIXELS=261120 LORA_MAXLEN=1900 LORA_BS=2 LORA_ACC=16 \
    PROMPT_VARIANT=badimg LORA_BASE_MODEL=Qwen/Qwen3.5-4B LORA_TARGETS="$QT" \
    ../.venv/bin/python -u lora_train.py all 2 > lora_foldall_mmqwen2all_s1.out 2>&1
echo "SEED1 DONE" >> lora_foldall_mmqwen2all_s1.out
