#!/bin/bash
# Очередь энкодеров. e5-small по инсайду из соседней задачи бьёт e5-base при
# меньшем размере, поэтому оба варианта (с OCR и без) гоняем именно на нём.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
PY=../.venv/bin/python
while pgrep -f "train_encoder.py no_ocr intfloat/multilingual-e5-base" >/dev/null || \
      pgrep -f "train_encoder.py no_ocr$" >/dev/null; do sleep 30; done
for mode in no_ocr with_ocr; do
    echo "$(date +%H:%M:%S) e5-small $mode" >> enc_queue.log
    $PY -u train_encoder.py "$mode" intfloat/multilingual-e5-small \
        > "enc_small_${mode}.log" 2>&1
    mv "enc_${mode}.parquet" "enc_small_${mode}.parquet" 2>/dev/null
done
echo "$(date +%H:%M:%S) очередь энкодеров готова" >> enc_queue.log
