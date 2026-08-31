"""Насыщен ли ранг LoRA? Проверка через сингулярные числа обучённых добавок.

Если у ΔW = B·A все r сингулярных чисел значимы — ранг упёрт в потолок и его
стоит поднимать. Если энергия сосредоточена в первых нескольких — r=16 избыточен,
и увеличение ранга только добавит переобучения.

Полную матрицу out×in (до 9216x2560) для 128 модулей не строим: ранг ΔW и так <= r,
поэтому через QR задача сводится к SVD матрицы r x r:
    BA = Q_B (R_B R_A^T) Q_A^T,  у Q ортонормированные столбцы
    => сингулярные числа BA совпадают с числами (R_B R_A^T)

Запуск: python exp/lora_spectrum.py [adapter_dir ...]
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = "/workspace/counter/"
DEFAULT = [ROOT + "exp/lora_foldall", ROOT + "exp/lora_fold0"]
DIRS = sys.argv[1:] or DEFAULT


def spectrum(adapter_dir):
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    pairs = defaultdict(dict)
    for k, v in tensors.items():
        if ".lora_A." in k:
            pairs[k.split(".lora_A.")[0]]["A"] = v
        elif ".lora_B." in k:
            pairs[k.split(".lora_B.")[0]]["B"] = v

    rows = []
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].float()          # (r, in)
        B = ab["B"].float()          # (out, r)
        _, Rb = torch.linalg.qr(B)           # Rb: (r, r)
        _, Ra = torch.linalg.qr(A.T)         # Ra: (r, r)
        s = torch.linalg.svdvals(Rb @ Ra.T).numpy()
        rows.append((name, s))
    return r, rows


def eff_rank(s, frac):
    """Сколько компонент нужно, чтобы набрать frac энергии (суммы квадратов)."""
    e = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    return int(np.searchsorted(e, frac) + 1)


for d in DIRS:
    if not os.path.isdir(d):
        print(f"нет каталога {d}, пропускаю")
        continue
    r, rows = spectrum(d)
    if not rows:
        print(f"{d}: нет пар A/B")
        continue
    print("=" * 96)
    print(f"### {os.path.basename(d)}   r={r}, модулей={len(rows)}")

    r90 = np.array([eff_rank(s, 0.90) for _, s in rows])
    r99 = np.array([eff_rank(s, 0.99) for _, s in rows])
    tail = np.array([s[-1] / max(s[0], 1e-12) for _, s in rows])
    print(f"  эффективный ранг (90% энергии): медиана {np.median(r90):.1f} из {r}  "
          f"[{r90.min()}..{r90.max()}]")
    print(f"  эффективный ранг (99% энергии): медиана {np.median(r99):.1f} из {r}  "
          f"[{r99.min()}..{r99.max()}]")
    print(f"  доля модулей, где 99% энергии требует ВСЕ {r} компонент: "
          f"{(r99 >= r).mean()*100:.0f}%")
    print(f"  отношение последнего сингулярного к первому: медиана {np.median(tail):.3f}")

    mean_s = np.mean([s / max(s[0], 1e-12) for _, s in rows], axis=0)
    print("  усреднённый нормированный спектр (s_i / s_1):")
    print("    " + "  ".join(f"{v:.2f}" for v in mean_s))

    by_type = defaultdict(list)
    for name, s in rows:
        m = re.search(r"\.([a-z_]+_proj)$", name)
        by_type[m.group(1) if m else "?"].append(eff_rank(s, 0.99))
    print("  эффективный ранг (99%) по типам модулей:")
    for t, vals in sorted(by_type.items()):
        print(f"    {t:12s} медиана {np.median(vals):.1f} из {r}  (модулей {len(vals)})")

print("\n" + "=" * 96)
print("КАК ЧИТАТЬ: если медиана эффективного ранга по 99% энергии близка к r —")
print("ранг упёрт, есть смысл поднимать. Если заметно меньше — ёмкости хватает,")
print("и рост r даст только переобучение при 87 семьях позитивов на флам.")
