#!/bin/bash
# Комбинированный конфиг вместо перебора поодиночке.
#
# Причина: шумовой пол на одном фолде ~4 пункта F1 (замерено сменой одного сида),
# а ожидаемые эффекты 2-5 пунктов. Поодиночке они неразличимы. Поэтому сначала
# проверяем, есть ли эффект ВООБЩЕ, а раскладываем по вкладам только если он есть.
#
# Что объединено (у каждого своя диагностика, не догадки):
#   ранг 32   — спектр показал эффективный ранг 16 из 16 у 73-80% модулей
#   модули    — 28% линейных весов (24 слоя линейного внимания) вне адаптации
#   3 эпохи   — переход 1->2 дал +4.5 пункта, плато не достигнуто
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj,in_proj_z"

running() {
    local pids real="" n=0 ppid comm
    pids=$(pgrep -f "lora_train.py" 2>/dev/null)
    for p in $pids; do
        comm=$(cat "/proc/$p/comm" 2>/dev/null)
        case "$comm" in python*) real="$real $p" ;; esac
    done
    for p in $real; do
        ppid=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
        echo "$real" | grep -qw "$ppid" || n=$((n + 1))
    done
    echo "$n"
}
while [ "$(running)" -gt 1 ]; do sleep 120; done   # ждём, пока останется <=1 задача

for f in 0 1; do
    echo "$(date +%H:%M:%S) комбо на фолде $f" >> queue_combo.log
    LORA_TAG=_combo LORA_RANK=32 LORA_TARGETS="$TARGETS" LORA_SEED=0 \
        $PY -u lora_train.py "$f" 3 > "lora_fold${f}_combo.out" 2>&1
done
echo "$(date +%H:%M:%S) комбо готово на двух фолдах" >> queue_combo.log
