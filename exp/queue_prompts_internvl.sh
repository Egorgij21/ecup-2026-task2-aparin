#!/bin/bash
# Одна карта, две линии по очереди. Дешёвое и уже стоящее в сабмите — первым.
#
# ЛИНИЯ 1: варианты промпта для zero-shot (обучение не нужно, только скоринг).
#   Основание: zero-shot в спорной зоне LoRA даёт AUC 0.4680 — ниже случайного, —
#   хотя вносит в флам-ансамбль +0.042. В решающей зоне компонент слеп.
#   Отличие от провалившихся попыток: ответ читается ОДНИМ forward-проходом,
#   генерации нет, стоимость инференса не меняется вовсе.
#   Критерий (оба условия): AUC спорной зоны > 0.52 И F1 бленда > 0.8665.
#
# ЛИНИЯ 2: InternVL3_5-2B как третья база.
#   Сначала ПРОВЕРКА ЗАГРУЗКИ: модели нужен trust_remote_code — ровно то, на чём
#   отвалилась Nanbeige (её код упал на нашей версии transformers). Разница в том,
#   что InternVL есть в списке организаторов, то есть в их образе она работает.
#   Если не грузится — обучение не запускаем, карту не тратим.
#   Гейт после фолда 4 (59 позитивов, самый информативный): PR >= 0.84 И корр < 0.80.
#   Гейт требует СИЛЫ, а не только независимости: у Qwen3-VL-2B корреляция была
#   отличная (0.62-0.65), но PR 0.8029 против 0.8857 у наших — и в бленде не помогла.
#   Оговорка: языковая часть InternVL это Qwen3ForCausalLM, то есть «другое
#   семейство» тут слабее, чем звучит.
cd /workspace/counter/exp || exit 1
export CUDA_VISIBLE_DEVICES="${CARD:-6}"
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv/bin/python
LOG=queue_prompts_internvl.log

for v in checklist exclusions short; do
    echo "$(date +%H:%M:%S) промпт: $v" >> $LOG
    $PY -u zeroshot_prompt_variants.py "$v" > "zs_$v.out" 2>&1
    tail -3 "zs_$v.out" >> $LOG
done
echo "$(date +%H:%M:%S) промпты готовы -> eval_prompt_variants.py" >> $LOG

echo "$(date +%H:%M:%S) InternVL: проверка загрузки" >> $LOG
$PY -u check_internvl.py > check_internvl.out 2>&1
if grep -q "ПРИГОДНА" check_internvl.out; then
    echo "$(date +%H:%M:%S) InternVL пригодна, обучаю фолд 4" >> $LOG
    TARGETS=$(grep -oP 'РЕГУЛЯРКА: \K.*' check_internvl.out | head -1)
    LORA_TAG=_ivl LORA_BASE_MODEL=OpenGVLab/InternVL3_5-2B LORA_TARGETS="$TARGETS" \
        $PY -u lora_train.py 4 2 > lora_fold4_ivl.out 2>&1
    echo "$(date +%H:%M:%S) InternVL фолд 4 готов" >> $LOG
else
    echo "$(date +%H:%M:%S) InternVL НЕ пригодна — обучение пропущено, карта не потрачена" >> $LOG
fi
