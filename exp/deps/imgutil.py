"""Единый загрузчик картинок карточки для обучения И скоринга — чтобы train/infer не
разошлись (§1). Поддерживает ТАЙЛИНГ: первую картинку режем на tile×tile тайлов.

Зачем тайлинг. gemma даёт ФИКС. ~280 токенов на картинку независимо от разрешения.
Мелкий текст на упаковке (номер СГР, «не является лекарственным средством») в 280
токенах на всю площадь нечитаем. Порезав картинку на tile×tile, каждый тайл получает
свои ~280 токенов на 1/tile² площади => tile²× разрешение на тексте. Это НОВЫЙ сигнал
(разрешение), а не рекомбинация: «3 картинки» давали разные ВИДЫ, но каждый всё равно
280 токенов на всю площадь (проверено: не помогло).
"""
import os

from PIL import Image


def _fit(im, max_pixels):
    """Ограничить площадь до max_pixels (как в обучении/скоринге исходно). Кратно 28."""
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        k = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * k) // 28 * 28),
                        max(28, int(h * k) // 28 * 28)), Image.LANCZOS)
    return im


def load_card_images(img_dir, pid, n_img, max_pixels=0, tile=1):
    """Список PIL-картинок карточки.
      tile>1  — берём ПЕРВУЮ картинку, режем на tile×tile тайлов (по строкам сверху вниз),
                каждый ужимаем до max_pixels. Итого tile² «картинок».
      tile==1 — первые n_img картинок как есть.
    Возвращает [] если папки нет / картинок нет / первая не открылась (для tile>1)."""
    d = os.path.join(str(img_dir), str(pid))
    if not os.path.isdir(d):
        return []
    files = sorted(x for x in os.listdir(d) if x.lower().endswith(".jpg"))
    if tile > 1:
        if not files:
            return []
        try:
            im = Image.open(os.path.join(d, files[0])).convert("RGB")
        except Exception:
            return []
        w, h = im.size
        out = []
        for r in range(tile):
            for c in range(tile):
                t = im.crop((c * w // tile, r * h // tile,
                             (c + 1) * w // tile, (r + 1) * h // tile))
                t.load()                      # материализуем ДО закрытия исходника
                out.append(_fit(t, max_pixels))
        im.close()
        return out
    out = []
    for f in files[:n_img]:
        try:
            im = Image.open(os.path.join(d, f)).convert("RGB")
        except Exception:
            continue
        out.append(_fit(im, max_pixels))
    return out
