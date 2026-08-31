#!/bin/bash
# Полный OOF для варианта _synth: фолды 1-4 (нулевой уже посчитан ночью).
#
# Почему именно эти два варианта. На фолде 0 они дали лучший F1 бленда из девяти:
# синтетика 0.9302 и семплер family 0.9286 против 0.8916 у базы. Но фолд 0 уже
# один раз обманул: r32 показывал там 0.9157 и оказался ХУЖЕ базы на полном OOF
# (0.8451 против 0.8615). Поэтому оба кандидата проверяются честно, до конца.
#
# У обоих есть механизм, а не только число:
#   synth  — 0 из 30 пропусков имели свою семью среди позитивов train-фолда,
#            синтетика добавляет 96 концептов против 87 реальных семей;
#   famsmp — на эффективный батч приходилось 0.48 позитива флам, стало 6.8.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=4
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
for f in 1 2 3 4; do
    echo "$(date +%H:%M:%S) [карта 4] _synth фолд $f" >> queue_oof_synth.log
    LORA_TAG=_synth LORA_SEED=0 \
        LORA_SYNTH=/workspace/counter/exp/synth_flam.parquet LORA_SYNTH_W=1.0 \
        $PY -u lora_train.py "$f" 2 > "lora_fold${f}_synth.out" 2>&1
    echo "$(date +%H:%M:%S) [карта 4] фолд $f готов" >> queue_oof_synth.log
done
echo "$(date +%H:%M:%S) [карта 4] _synth: полный OOF собран" >> queue_oof_synth.log
