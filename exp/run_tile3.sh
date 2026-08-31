cd /workspace/counter/exp
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t3() {
  CUDA_VISIBLE_DEVICES=$1 LORA_TAG="_mmtile3" LORA_SEED=0 LORA_IMAGES=1 LORA_TILE=3     LORA_MAX_PIXELS=261120 LORA_MAXLEN=3400 LORA_BS=1 LORA_ACC=32 PROMPT_VARIANT=badimg     LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'     ../.venv/bin/python -u lora_train.py $2 2 > "lora_fold$2_mmtile3.out" 2>&1
}
t3 5 0 & sleep 30 ; t3 7 1 & wait
