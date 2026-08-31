"""LoRA-дообучение Qwen3.5-4B как бинарного классификатора «Да/Нет».

Обучаем только на одном токене ответа (Да/Нет) — это классификация, а не генерация,
поэтому лосс считаем строго по позиции ответа.

ВАЖНО: фолды режем по СЕМЬЯМ товаров (near-duplicates), иначе валидация врёт,
как это уже случилось с TF-IDF (CV 0.886 против 0.739 на LB).

Запуск: CUDA_VISIBLE_DEVICES=6 python exp/lora_train.py <fold|all> [epochs]
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ROOT = "/workspace/counter/"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# промпт живёт в exp/deps (копия submit16/src/prompts.py) — не зависим от папок сабмитов,
# которые чистятся. Промпт ИДЕНТИЧЕН тому, что в сабмите: это тот же build_messages.
sys.path.insert(0, ROOT + "exp/deps")
from prompts import build_messages  # noqa: E402
from folds import family_labels, make_folds  # noqa: E402

MODEL = os.environ.get("LORA_BASE_MODEL", "Qwen/Qwen3.5-4B")
FOLD = sys.argv[1] if len(sys.argv) > 1 else "0"
EPOCHS = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
# разные сиды дают разные адаптеры на одних и тех же данных — основа для ансамбля
SEED = int(os.environ.get("LORA_SEED", "0"))
TAG = os.environ.get("LORA_TAG", "")
# У мультимодальных моделей (gemma-4) вне языковой части лежат кастомные слои
# (Gemma4ClippableLinear), которые peft не поддерживает. Поэтому цель задаётся
# регуляркой строго по языковой части.
TARGETS = os.environ.get("LORA_TARGETS", "")
# В сабмите LoRA применяется только к флам-категории, а обучается на обеих.
# Эта опция позволяет отдать всю ёмкость адаптера той задаче, где он реально работает.
ONLY_CATEGORY = os.environ.get("LORA_ONLY_CATEGORY", "")
# Спектр обучённых добавок показал, что r=16 упёрт: 99% энергии требует все 16
# компонент у 73-80% модулей, хвост спектра 0.24 от ведущего (см. exp/lora_spectrum.py).
# alpha держим = 2r, чтобы масштаб добавки alpha/r не поехал.
RANK = int(os.environ.get("LORA_RANK", "16"))
# none / balanced / family — см. sampling_probs(). По умолчанию строго прежнее
# поведение, иначе уже посчитанные фолды станут несопоставимы с новыми.
SAMPLER = os.environ.get("LORA_SAMPLER", "none")
# Путь к синтетическим карточкам (exp/synth_flam.parquet). Добавляются ТОЛЬКО в train.
# Утечки нет по построению: генерация не смотрела в данные, лишь в опубликованные
# правила площадки (см. gen_synthetic.py). Но подстраховка всё равно нужна — синтетика
# кладётся после разбиения на фолды, поэтому в валидацию попасть физически не может.
SYNTH = os.environ.get("LORA_SYNTH", "")
SYNTH_W = float(os.environ.get("LORA_SYNTH_W", "1.0"))
_BASE_MAXLEN = int(os.environ.get("LORA_MAXLEN", "1280"))
BS = int(os.environ.get("LORA_BS", "4"))
ACC = int(os.environ.get("LORA_ACC", "8"))
# Скорость обучения и функция потерь до 15.08 не выводились наружу и ни разу
# не перебирались. У gemma-4-12B на кривой лосса было видно, что 1e-4 для крупной
# модели велика (лосс рос на отрезке 60-80%), — для 4B оптимум тоже мог не совпасть.
LR = float(os.environ.get("LORA_LR", "1e-4"))
# ce — как было; focal — при 3.6% позитивов на флам напрашивается, но не пробовали
LOSS = os.environ.get("LORA_LOSS", "ce")
FOCAL_GAMMA = float(os.environ.get("LORA_FOCAL_GAMMA", "2.0"))
# Мультимодальное обучение: картинки подаются в модель ВМЕСТЕ с текстом.
# Мы этого не делали ни разу — все замеры с картинками были в zero-shot, а разрыв
# zero-shot против обучения у нас четырёхкратный (PR 0.19 против 0.84 на тексте).
# Одна картинка у gemma стоит 258 токенов, поэтому MAXLEN поднимается автоматически:
# при обрезке первым отрезается именно изображение, и замер стал бы бессмысленным.
IMAGES = int(os.environ.get("LORA_IMAGES", "0"))          # сколько картинок на товар
IMG_DIR = ROOT + "data/images"
# Ограничение размера картинки ДО подачи в процессор. Критично для Qwen: у него
# динамическое разрешение, и число визуальных токенов растёт с размером картинки —
# замерено на 60 карточках: медиана 1684, 95-й процентиль 5787, максимум 16910.
# У gemma фиксированные 258 токенов на картинку, ей это безразлично.
# 261120 — пресет «M» из бейзлайна организаторов; больше даёт читаемость мелкого
# текста на упаковке, меньше экономит время. Параметр НИ РАЗУ не подбирался.
MAX_PIXELS = int(os.environ.get("LORA_MAX_PIXELS", "261120"))
# Тайлинг: первую картинку режем на TILE×TILE тайлов, каждый ~280 токенов => TILE²×
# разрешение на мелком тексте упаковки. TILE=1 — обычный режим. См. deps/imgutil.py.
TILE = int(os.environ.get("LORA_TILE", "1"))


class Cards(Dataset):
    """Вес примера обратно пропорционален частоте его класса ВНУТРИ категории:
    у «Легковоспламеняющихся» всего 3.6% позитивов, без взвешивания модель
    выучит константу «Нет».

    При SAMPLER != "none" балансировка уезжает в семплер, а веса становятся
    единичными — см. комментарий к SAMPLER."""

    def __init__(self, df, tok, yes_id, no_id, weighted=True, proc=None, maxlen=1280):
        self.df = df.reset_index(drop=True)
        self.tok, self.yes_id, self.no_id = tok, yes_id, no_id
        self.proc, self.maxlen = proc, maxlen
        self.w = np.ones(len(self.df), dtype=np.float32)
        if weighted:
            for cat, g in self.df.groupby("category"):
                p = g["label"].mean()
                for i, lab in zip(g.index, g["label"]):
                    self.w[i] = 0.5 / p if lab == 1 else 0.5 / (1 - p)
        # SYNTH_W применяем к весам ТОЛЬКО когда балансировка идёт через веса (weighted).
        # При включённом семплере веса единичны, а SYNTH_W учитывается в sampling_probs —
        # иначе двойной учёт (SYNTH_W²). Аудит 26.08.
        if weighted and "fold" in self.df.columns:      # синтетика помечена fold = -1
            self.w[(self.df["fold"] == -1).values] *= SYNTH_W


    def __len__(self):
        return len(self.df)

    def _images(self, pid):
        # единый загрузчик (обучение и скоринг), с тайлингом при TILE>1
        from imgutil import load_card_images
        return load_card_images(IMG_DIR, pid, IMAGES, MAX_PIXELS, TILE)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        target = self.yes_id if int(r["label"]) == 1 else self.no_id
        if not IMAGES:
            msgs = build_messages(r["category"], r["name"], r["description"])
            try:
                prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(prompt, truncation=True, max_length=self.maxlen - 1)["input_ids"]
            return {"input_ids": ids + [target], "answer_pos": len(ids) - 1,
                    "target": target, "weight": float(self.w[i])}
        ims = self._images(r["id"])
        base = build_messages(r["category"], r["name"], r["description"])
        text = base[-1]["content"]
        content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        kw = {"text": [prompt], "return_tensors": "pt"}
        if ims:
            kw["images"] = [ims]
        enc = self.proc(**kw)
        for im in ims:
            im.close()
        ids = enc["input_ids"][0].tolist()
        if len(ids) > self.maxlen - 1:            # обрезаем СЛЕВА: image-токены в начале
            raise RuntimeError(f"промпт {len(ids)} токенов > MAXLEN {self.maxlen}; "
                               "подними LORA_MAXLEN или уменьши LORA_IMAGES")
        item = {"input_ids": ids + [target], "answer_pos": len(ids) - 1,
                "target": target, "weight": float(self.w[i])}
        # image_grid_thw у Qwen, image_position_ids у gemma — набор ключей РАЗНЫЙ.
        # Пропуск image_grid_thw ломает Qwen молча: модель не может восстановить
        # сетку патчей, лосс уходит в 4.2 против 0.22 у рабочего прогона.
        for k in ("pixel_values", "image_position_ids", "image_grid_thw",
                  "mm_token_type_ids"):
            if k in enc:
                item[k] = enc[k]
        return item


def load_base(name):
    """Загрузка базы с явным выбором класса модели.

    Мультимодальные модели не всегда грузятся через AutoModelForCausalLM. У Qwen3.5-4B
    CausalLM — ТЕКСТОВЫЙ класс (Qwen3_5ForCausalLM: нет vision tower, forward без
    pixel_values), и при IMAGES>0 картинки ушли бы в никуда МОЛЧА. Мультимодальный Qwen
    живёт в AutoModelForImageTextToText (там model.language_model.layers + vision).
    У gemma-4-E4B, наоборот, сам CausalLM мультимодален (принимает pixel_values) — её
    путь НЕ меняем.

    Поэтому при IMAGES>0 проверяем, что выбранный класс реально принимает pixel_values;
    если нет — не молчим, пробуем следующий класс. Это НЕ откат: выбранный класс
    печатается, а если ни один не дал мультимодальную модель — падаем.
    """
    import inspect as _insp

    from transformers import AutoModelForImageTextToText
    errs = []
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            m = cls.from_pretrained(name, dtype=torch.bfloat16)
        except (ValueError, KeyError) as e:
            errs.append(f"{cls.__name__}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if IMAGES and "pixel_values" not in _insp.signature(m.forward).parameters:
            errs.append(f"{cls.__name__}: forward без pixel_values (текстовый), "
                        f"а IMAGES={IMAGES} — картинки ушли бы в никуда")
            del m
            continue
        print(f"база загружена как {cls.__name__}"
              f"{' (мультимодальный)' if IMAGES else ''}", flush=True)
        return m
    raise RuntimeError("не удалось получить мультимодальную базу:\n  " + "\n  ".join(errs))


def sampling_probs(df, mode):
    """Вероятность вытянуть строку. Балансировка через семплер, а не через лосс.

    Почему это вообще понадобилось. В режиме "none" на эффективный батч
    (BS 4 x ACC 8 = 32) приходится 0.48 позитива флам: половина шагов оптимизатора
    не видит ни одного положительного примера целевого класса. А нормировка
    (per*w).sum()/w.sum() считается ВНУТРИ микробатча из 4 строк, поэтому попавший
    туда позитив с весом 13.9 против негативов с 0.52 занимает ~90% лосса, а
    микробатч без позитива нормируется совсем иначе. Веса работают не как веса,
    а как случайный множитель скорости обучения.

      balanced — внутри каждой категории позитивы и негативы получают равную
                 суммарную массу. Батчи перестают голодать.
      family   — то же, но масса позитива делится на размер его СЕМЬИ. Разбор
                 ошибок показал: 0 из 30 пропусков имели свою семью среди
                 позитивов train-фолда, то есть модель валится ровно на новых
                 семьях. Значит учить надо разнообразию семей, а не количеству
                 строк — иначе семья из 8 почти-дублей перевешивает 8 одиночек.
    """
    p = np.zeros(len(df), dtype=np.float64)
    for cat, g in df.groupby("category"):
        r = g["label"].mean()
        for i, lab in zip(g.index, g["label"]):
            p[i] = 0.5 / r if lab == 1 else 0.5 / (1 - r)
    if mode == "family":
        fam = family_labels(df["name"].fillna("").values)
        size = pd.Series(fam).value_counts()
        p = p / np.array([size[f] for f in fam], dtype=np.float64)
    if "fold" in df.columns:               # синтетика помечена fold = -1
        p = p * np.where((df["fold"] == -1).values, SYNTH_W, 1.0)
    return p / p.sum()


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    mm = torch.zeros((len(batch), n), dtype=torch.long)
    pos, tgt, w = [], [], []
    have_mm = any("mm_token_type_ids" in b for b in batch)
    have_img = any("pixel_values" in b for b in batch)
    for i, b in enumerate(batch):           # левый паддинг
        L = len(b["input_ids"])
        ids[i, n - L:] = torch.tensor(b["input_ids"])
        att[i, n - L:] = 1
        if "mm_token_type_ids" in b:
            m = b["mm_token_type_ids"][0]
            mm[i, n - L:n - L + len(m)] = m
        pos.append(n - L + b["answer_pos"])
        tgt.append(b["target"])
        w.append(b["weight"])
    extra = {}
    if have_mm:
        extra["mm_token_type_ids"] = mm
    # image_position_ids задают положение патча ВНУТРИ картинки, а не в последовательности,
    # поэтому левый паддинг их не сдвигает — склеиваем как есть. Собираем НЕЗАВИСИМО от
    # mm_token_type_ids: база, дающая image_grid_thw без mm_token_type_ids, иначе молча
    # ушла бы в текст (loss ~4.2 против 0.2). Аудит 26.08.
    for k in ("pixel_values", "image_position_ids", "image_grid_thw"):
        vs = [b[k] for b in batch if k in b]
        if vs:
            extra[k] = torch.cat(vs, dim=0)
    # картинки были на входе, но pixel_values не дошёл до модели — это тихий уход в текст
    # в мультимодальном прогоне. Падаем громко, а не считаем не то.
    if have_img and "pixel_values" not in extra:
        raise RuntimeError("pixel_values в батче есть, но не собран в extra — "
                           "молчаливый уход в текст запрещён")
    return ids, att, torch.tensor(pos), torch.tensor(tgt), torch.tensor(w), extra


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"модель={MODEL} фолд={FOLD} сид={SEED} тег={TAG!r}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    # Текстовый путь режет по max_length. Вопрос и generation-prompt стоят В КОНЦЕ, а
    # truncation_side по умолчанию "right" срезал бы именно их -> answer_pos уезжал бы
    # внутрь описания (тихо, на ~3 длинных строках из 12971). Режем СЛЕВА: теряем начало
    # правил, но сохраняем вопрос+gen-prompt и корректную позицию ответа. Аудит 26.08.
    tok.truncation_side = "left"
    proc = None
    MAXLEN = _BASE_MAXLEN
    if IMAGES:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL)
        proc.tokenizer.padding_side = "left"
        MAXLEN = int(os.environ.get("LORA_MAXLEN", str(_BASE_MAXLEN + 300 * IMAGES)))
        print(f"мультимодальный режим: {IMAGES} картинок на товар, MAXLEN={MAXLEN}", flush=True)
    yes_id = tok.encode("Да", add_special_tokens=False)[0]
    no_id = tok.encode("Нет", add_special_tokens=False)[0]
    pad_id = tok.pad_token_id or tok.eos_token_id
    print(f"yes_id={yes_id} no_id={no_id}", flush=True)

    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    df["fold"] = make_folds(df)          # фолды считаем ДО фильтрации категорий
    if ONLY_CATEGORY:
        df = df[df["category"] == ONLY_CATEGORY].reset_index(drop=True)
        print(f"обучаемся только на категории {ONLY_CATEGORY!r}: {len(df)} строк", flush=True)
    if FOLD == "all":
        tr_df, va_df = df, df.iloc[:0]
    else:
        f = int(FOLD)
        tr_df, va_df = df[df["fold"] != f], df[df["fold"] == f]
    print(f"train={len(tr_df)} valid={len(va_df)}", flush=True)

    if SYNTH:
        # Синтетика подмешивается ПОСЛЕ разбиения на фолды — в валидацию попасть
        # физически не может. Разбор ошибок показал, что 0 из 30 пропусков имели
        # свою семью среди позитивов train-фолда, то есть модель валится на новых
        # семьях; синтетика добавляет 96 концептов против 87 реальных семей.
        s = pd.read_parquet(SYNTH)
        need = {"name", "description", "category", "label"}
        if not need.issubset(s.columns):
            raise ValueError(f"в {SYNTH} нет колонок {need - set(s.columns)}")
        if ONLY_CATEGORY:
            s = s[s["category"] == ONLY_CATEGORY]
        s = s.assign(fold=-1)
        if "id" not in s.columns:
            s["id"] = [f"synth_{i}" for i in range(len(s))]
        tr_df = pd.concat([tr_df, s[tr_df.columns]], ignore_index=True)
        print(f"синтетика: +{len(s)} строк ({int(s['label'].sum())} поз), "
              f"train стал {len(tr_df)}, вес семплирования x{SYNTH_W}", flush=True)

    model = load_base(MODEL).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # список через запятую -> список суффиксов, иначе строка трактуется как регулярка
    if TARGETS:
        targets = [t.strip() for t in TARGETS.split(",")] if "," in TARGETS else TARGETS
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(
        r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()

    ds = Cards(tr_df, tok, yes_id, no_id, weighted=(SAMPLER == "none"), proc=proc, maxlen=MAXLEN)
    if IMAGES:
        # Покрытие картинками: строки без папки/с битым jpg молча скорятся текстом. Логируем,
        # сколько мультимодального прогона РЕАЛЬНО мультимодально — иначе мерим не то. Аудит 26.08.
        have = sum(1 for pid in tr_df["id"]
                   if os.path.isdir(os.path.join(IMG_DIR, str(pid)))
                   and any(x.lower().endswith(".jpg")
                           for x in os.listdir(os.path.join(IMG_DIR, str(pid)))))
        print(f"покрытие картинками: {have}/{len(tr_df)} строк с изображениями "
              f"({have / max(len(tr_df), 1) * 100:.1f}%)", flush=True)
    if SAMPLER == "none":
        dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=(0 if IMAGES else 2),
                        collate_fn=lambda b: collate(b, pad_id))
    else:
        probs = sampling_probs(ds.df, SAMPLER)
        sampler = WeightedRandomSampler(torch.as_tensor(probs, dtype=torch.double),
                                        num_samples=len(ds), replacement=True)
        dl = DataLoader(ds, batch_size=BS, sampler=sampler, num_workers=(0 if IMAGES else 2),
                        collate_fn=lambda b: collate(b, pad_id))
        m = (ds.df["category"] == "Легковоспламеняющиеся") & (ds.df["label"] == 1)
        print(f"семплер={SAMPLER}: доля позитивов флам в выборке "
              f"{probs[m.values].sum():.3f} (было {m.mean():.3f}), "
              f"на эффективный батч {probs[m.values].sum()*BS*ACC:.1f} штук", flush=True)
    steps = int(len(dl) * EPOCHS) // ACC
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                            weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    print(f"optimizer steps: {steps} | lr={LR} | loss={LOSS}", flush=True)

    outdir = ROOT + f"exp/lora_fold{FOLD}{TAG}"
    logf = open(ROOT + f"exp/lora_fold{FOLD}{TAG}.log", "a", buffering=1)  # line-buffered

    def say(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    model.train()
    t0, done, run = time.time(), 0, 0.0
    stop = False
    for ep in range(int(np.ceil(EPOCHS))):
        if stop:
            break
        say(f"=== эпоха {ep} началась, {time.time()-t0:.0f}с от старта обучения")
        for it, (ids, att, pos, tgt, w, extra) in enumerate(dl):
            ids, att, pos, tgt, w = ids.cuda(), att.cuda(), pos.cuda(), tgt.cuda(), w.cuda()
            extra = {k: (v.cuda().to(torch.bfloat16) if v.dtype.is_floating_point else v.cuda())
                     for k, v in extra.items()}
            logits = model(input_ids=ids, attention_mask=att, **extra).logits
            sel = logits[torch.arange(len(pos), device=pos.device), pos].float()
            per = torch.nn.functional.cross_entropy(sel, tgt, reduction="none")
            if LOSS == "focal":
                # focal: (1-p_t)^gamma * CE, где p_t — вероятность верного класса.
                # Давит вклад лёгких примеров, которых при 3.6% позитивов подавляющее
                # большинство, и оставляет градиент на спорных.
                per = ((1.0 - torch.exp(-per)) ** FOCAL_GAMMA) * per
            elif LOSS != "ce":
                raise ValueError(f"неизвестный LORA_LOSS={LOSS!r}, ожидается ce или focal")
            loss = (per * w).sum() / w.sum() / ACC
            loss.backward()
            run += loss.item()
            if (it + 1) % ACC == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                done += 1
                if done % 20 == 0:
                    el = time.time() - t0
                    say(f"  ep{ep} step {done}/{steps} loss={run/20:.4f} "
                        f"{el:.0f}s eta={el/done*(steps-done):.0f}s")
                    run = 0.0
                if done >= steps:
                    stop = True
                    break
        model.save_pretrained(outdir + f"_ep{ep}")
        say(f"=== чекпоинт эпохи {ep} сохранён в {outdir}_ep{ep}")

    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print("сохранено в", outdir, flush=True)

    if len(va_df):
        from lora_infer import score_df
        model.eval()
        s = score_df(model, tok, va_df, yes_id, no_id, MAXLEN,
                     proc=proc, n_img=IMAGES, img_dir=IMG_DIR, max_pixels=MAX_PIXELS, tile=TILE)
        out = va_df[["id", "category", "label"]].copy()
        out["lora_score"] = s
        out.to_parquet(ROOT + f"exp/lora_oof_fold{FOLD}{TAG}.parquet")
        from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
        for cat, g in out.groupby("category"):
            y, sc = g["label"].values, g["lora_score"].values
            ths = np.linspace(0.02, 0.98, 193)
            fb = max((f1_score(y, (sc >= t).astype(int)), t) for t in ths)
            fm = max((f1_score(y, (sc >= t).astype(int), average="macro"), t) for t in ths)
            print(f"{cat}: n={len(g)} pos={y.sum()} AUC={roc_auc_score(y,sc):.4f} "
                  f"PR={average_precision_score(y,sc):.4f} "
                  f"F1bin={fb[0]:.4f}@{fb[1]:.2f} F1mac={fm[0]:.4f}@{fm[1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
