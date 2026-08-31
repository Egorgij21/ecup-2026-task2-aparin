#!/bin/bash
# Гемма + синтетика на ВСЕХ данных — кандидат в сабмит сегодня.
#
# Основание. Гемма на всех данных дала 0.89017 против 0.87820 у Qwen при побайтово
# одинаковом остальном: +0.012, самый крупный подтверждённый прирост за соревнование.
# Значит правильная база — гемма, а все ночные эксперименты шли на Qwen.
# Синтетика на фолде 0 (правда, на Qwen) дала лучший F1 бленда из девяти вариантов:
# 0.9302 против 0.8916 у базы, PR 0.9356 против 0.8651.
#
# ЧЕСТНО ПРО РИСК: сочетание «гемма + синтетика» на полном OOF не мерено. Ставка
# оправдана тем, что на лидерборде держится ЛУЧШИЙ результат, поэтому цена неудачной
# отправки — потраченный слот из пяти дневных, а не потеря позиции. Локальная
# проверка идёт параллельно, а не вместо.
#
# У геммы модули задаются регуляркой строго по языковой части: вне её лежат
# Gemma4ClippableLinear, которые peft не поддерживает.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
GEMMA_TARGETS='model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'

echo "$(date +%H:%M:%S) [карта 6] гемма + синтетика, все данные, 2 эпохи" >> queue_gemma_synth.log
LORA_TAG=_gemmasynth LORA_BASE_MODEL=google/gemma-4-E4B-it LORA_TARGETS="$GEMMA_TARGETS" \
    LORA_SYNTH=/workspace/counter/exp/synth_flam.parquet \
    LORA_SYNTH_W=1.0 LORA_SEED=0 \
    $PY -u lora_train.py all 2 > lora_foldall_gemmasynth.out 2>&1
echo "$(date +%H:%M:%S) [карта 6] готово -> exp/lora_foldall_gemmasynth" >> queue_gemma_synth.log
