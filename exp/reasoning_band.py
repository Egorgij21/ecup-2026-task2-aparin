"""Помогает ли режим рассуждения в спорной зоне?

Идея. Сейчас zero-shot читает логиты последней позиции — один forward, дёшево.
Режим рассуждения требует генерации сотен токенов; на 3800 строках привата это
30-40 минут при общем лимите 40, то есть в сабмит целиком не влезает.

Но целиком и не нужно. Разбор ошибок показал: **159 строк из 5502 лежат в спорной
зоне LoRA (скор 0.02..0.98), и в них 34% всех позитивов** (67 из 198). В зоне
уверенного «да» текст вообще хуже случайного (AUC 0.49), в зоне уверенного «нет»
позитивов 0.5%. Значит дорогое рассуждение имеет смысл пускать только на спорные —
это ~3% строк, секунды работы, и ровно там, где принимается решение.

Замер: на спорных строках сравниваем AUC/PR обычного zero-shot и zero-shot
с рассуждением. Контроль — случайная выборка уверенных строк того же размера,
чтобы отделить «рассуждение помогает» от «эти строки просто легче».

Критерий, заданный ЗАРАНЕЕ: рассуждение считается полезным, если на спорных
строках его AUC выше обычного zero-shot не меньше чем на 0.05. Меньший разрыв
на 159 строках неотличим от шума.

Запуск: CUDA_VISIBLE_DEVICES=1 python exp/reasoning_band.py [макс_строк]
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
sys.path.insert(0, ROOT + "submit2")
from src.prompts import build_messages  # noqa: E402

MODEL = os.environ.get("ZS_MODEL", "Qwen/Qwen3.5-4B")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_NEW = int(os.environ.get("ZS_MAXNEW", "1024"))
BS = int(os.environ.get("ZS_BS", "8"))
CAT = "Легковоспламеняющиеся"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
d = df[df["category"] == CAT].reset_index(drop=True)

band = ((d["lora_score"] >= 0.02) & (d["lora_score"] <= 0.98)).values
rng = np.random.default_rng(0)
conf_idx = rng.choice(np.where(~band)[0], size=min(band.sum(), (~band).sum()), replace=False)
sel = (np.where(band)[0] if os.environ.get("BAND_ONLY")
       else np.concatenate([np.where(band)[0], conf_idx]))
if LIMIT:
    sel = sel[:LIMIT]
sub = d.iloc[sel].reset_index(drop=True)
is_band = np.isin(sel, np.where(band)[0])
print(f"спорных строк {band.sum()} (позитивов {d['label'].values[band].sum()} из "
      f"{d['label'].sum()}), контроль {len(sel) - is_band.sum()}, всего к скорингу {len(sub)}",
      flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()


def ids_for(words):
    out = set()
    for w in words:
        for v in (w, " " + w):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                out.add(t[0])
    return sorted(out)


yes_ids, no_ids = ids_for(["Да", "да", "ДА"]), ids_for(["Нет", "нет", "НЕТ"])

import inspect
_fwd = inspect.signature(model.forward).parameters
keep_last = ({"logits_to_keep": 1} if "logits_to_keep" in _fwd
             else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _fwd else {})
print("режим последней позиции:", keep_last or "не поддерживается — риск OOM", flush=True)

prompts = []
for cat, n, dd in zip(sub["category"], sub["name"].fillna(""), sub["description"].fillna("")):
    msgs = build_messages(cat, n, dd)
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True))

scores = np.zeros(len(prompts), dtype=np.float32)
ntok = []
done = 0
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(prompts), BS):
        chunk = prompts[i:i + BS]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1600).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen.shape[1] - enc["input_ids"].shape[1]
        ntok.append(new)
        # первый прогон был НЕВАЛИДЕН: при MAX_NEW=320 в потолок упёрлись ВСЕ карточки,
        # рассуждение обрывалось и до ответа не доходило. Считаем, сколько завершилось.
        eos = tok.eos_token_id
        done += int(sum(1 for row in gen[:, enc["input_ids"].shape[1]:]
                        if eos is not None and (row == eos).any()))
        # после рассуждения дочитываем вероятность ответа на следующей позиции.
        # logits_to_keep=1 обязателен: без него тензор batch x seq x 151936 это
        # десятки гигабайт и гарантированный OOM (проверено, прогон упал на 28.8 ГБ)
        out = model(input_ids=gen, attention_mask=torch.ones_like(gen), **keep_last)
        lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        y = torch.logsumexp(lp[:, yes_ids], dim=-1)
        n = torch.logsumexp(lp[:, no_ids], dim=-1)
        scores[i:i + BS] = torch.sigmoid(y - n).cpu().numpy()
        if i % (BS * 4) == 0:
            el = time.time() - t0
            print(f"  {i}/{len(prompts)} {el:.0f}с ~{el/max(i+BS,1)*len(prompts):.0f}с всего",
                  flush=True)

sub["reason_score"] = scores
sub["is_band"] = is_band
sub.to_parquet(ROOT + "exp/reasoning_band.parquet")
print(f"\nсгенерировано в среднем {np.mean(ntok):.0f} токенов (потолок {MAX_NEW}), "
      f"завершилось само {done}/{len(prompts)}, всего {time.time()-t0:.0f}с", flush=True)
if done < 0.8 * len(prompts):
    print("  ВНИМАНИЕ: больше 20% обрывов — замер снова недостоверен", flush=True)

print(f"\n  {'срез':22s} {'n':>5s} {'поз':>5s} {'AUC обычный':>12s} {'AUC ризонинг':>13s} "
      f"{'PR обычный':>11s} {'PR ризонинг':>12s}")
for tag, m in [("СПОРНАЯ ЗОНА", is_band), ("контроль (уверенные)", ~is_band)]:
    y = sub["label"].values[m]
    if not (0 < y.sum() < m.sum()):
        print(f"  {tag:22s} — один класс, метрики не определены")
        continue
    a0 = roc_auc_score(y, sub["llm_score"].values[m])
    a1 = roc_auc_score(y, sub["reason_score"].values[m])
    p0 = average_precision_score(y, sub["llm_score"].values[m])
    p1 = average_precision_score(y, sub["reason_score"].values[m])
    print(f"  {tag:22s} {m.sum():5d} {y.sum():5d} {a0:12.4f} {a1:13.4f} {p0:11.4f} {p1:12.4f}")

mb = is_band
if 0 < sub["label"].values[mb].sum() < mb.sum():
    d_auc = (roc_auc_score(sub["label"].values[mb], sub["reason_score"].values[mb])
             - roc_auc_score(sub["label"].values[mb], sub["llm_score"].values[mb]))
    print(f"\n  прирост AUC в спорной зоне: {d_auc:+.4f}  "
          f"(критерий задан заранее: нужно >= +0.05)")
    print(f"  вердикт: {'ПОЛЕЗНО' if d_auc >= 0.05 else 'в пределах шума, не берём'}")
