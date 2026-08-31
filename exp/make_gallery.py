"""Галерея карточек с изображениями — чтобы оценить глазами, полезны ли картинки.

Собирает HTML со всеми изображениями товара, текстом карточки, истинной меткой
и скором текстового ансамбля. Сортирует по трудности: сначала самые уверенные
ошибки, то есть там, где текст ошибся сильнее всего.

Зачем: обе наши базы мультимодальные, но мы всё время подавали им только текст.
Официальное правило БАД при этом прямо ссылается на изображение («в описании
ИЛИ НА ИЗОБРАЖЕНИИ содержится прямое указание»). Прежде чем вкладываться
в обучение с картинками, стоит посмотреть, видно ли на них искомое человеку.

Запуск: python exp/make_gallery.py [БАД|Легковоспламеняющиеся] [сколько на группу]
Открывать: gallery_<кат>.html из корня проекта (пути к картинкам относительные).
"""
import base64
import html
import io
import os
import sys

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
THUMB = int(os.environ.get("THUMB", "360"))   # длинная сторона превью, пикселей
QUALITY = int(os.environ.get("QUALITY", "72"))
MAX_IMGS = int(os.environ.get("MAX_IMGS", "3"))


def embed(path):
    """Картинка вшивается в HTML как data-URI.

    Относительные ссылки на файлы не работают, когда HTML открывают через Jupyter
    или предпросмотр редактора — там доступ к файловой системе заблокирован.
    Самодостаточный файл открывается где угодно, в том числе скачанный.
    """
    try:
        im = Image.open(os.path.join(ROOT, path)).convert("RGB")
    except Exception:
        return None
    im.thumbnail((THUMB, THUMB), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY, optimize=True)
    im.close()
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

ROOT = "/workspace/counter/"
CAT = sys.argv[1] if len(sys.argv) > 1 else "БАД"
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SLUG = "bad" if CAT == "БАД" else "flam"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
oof = pd.read_parquet(ROOT + "exp/text_oof.parquet")
d = df.merge(oof[["id", "text_score", "text_pred", "err"]], on="id")
d = d[d["category"] == CAT].reset_index(drop=True)


def imgs(pid):
    p = os.path.join(ROOT, "data/images", str(pid))
    if not os.path.isdir(p):
        return []
    return [f"data/images/{pid}/{f}" for f in sorted(os.listdir(p)) if f.lower().endswith(".jpg")]


# группы по трудности: уверенные ошибки идут первыми
groups = [
    ("Ложные срабатывания — текст уверенно сказал ДА, а метка НЕТ",
     d[d["err"] == "ложное"].sort_values("text_score", ascending=False)),
    ("Пропуски — текст уверенно сказал НЕТ, а метка ДА",
     d[d["err"] == "пропуск"].sort_values("text_score")),
    ("Пограничные — скор около порога, модель не уверена",
     d[(d["text_score"] > 0.35) & (d["text_score"] < 0.6)].sort_values("text_score")),
    ("Контроль — решено верно и уверенно",
     d[(d["err"] == "верно") & ((d["text_score"] > 0.9) | (d["text_score"] < 0.05))].head(200)),
]

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;padding:14px 20px;border-bottom:1px solid #d2d2d7;z-index:10}
h1{margin:0;font-size:19px} h2{margin:26px 20px 10px;font-size:16px;color:#424245}
.card{display:flex;gap:14px;background:#fff;margin:10px 20px;padding:12px;border-radius:10px;
      border:1px solid #e5e5ea;align-items:flex-start}
.imgs{display:flex;gap:6px;flex:0 0 auto;flex-wrap:wrap;max-width:460px}
.imgs img{height:130px;width:auto;border-radius:6px;border:1px solid #e5e5ea;background:#fafafa;cursor:zoom-in}
.imgs img:hover{transform:scale(2.6);transform-origin:top left;z-index:5;position:relative;box-shadow:0 8px 30px rgba(0,0,0,.3)}
.txt{flex:1;min-width:280px;font-size:13px;line-height:1.45}
.nm{font-weight:600;margin-bottom:5px;font-size:14px}
.ds{color:#6e6e73;max-height:88px;overflow:auto}
.badges{margin-bottom:6px;display:flex;gap:6px;flex-wrap:wrap}
.b{padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}
.pos{background:#d1f0d8;color:#0b6b2b} .neg{background:#ffe0e0;color:#8b1a1a}
.sc{background:#e8e8ed;color:#3a3a3c} .noimg{background:#fff3cd;color:#7a5b00}
"""

parts = [f"<!doctype html><meta charset=utf-8><title>Галерея {CAT}</title><style>{CSS}</style>",
         f"<header><h1>Карточки категории «{html.escape(CAT)}» — по {PER} на группу</h1>",
         "<div style='font-size:12px;color:#6e6e73;margin-top:4px'>Наведи на картинку — увеличится. "
         "Скор — текстовый ансамбль (без изображений). Вопрос: видно ли на упаковке то, "
         "чего не хватило тексту?</div></header>"]

for title, g in groups:
    parts.append(f"<h2>{html.escape(title)} — {len(g)} шт, показано {min(PER, len(g))}</h2>")
    for _, r in g.head(PER).iterrows():
        ims = imgs(r["id"])
        lab = ("<span class='b pos'>метка: ДА</span>" if r["label"] == 1
               else "<span class='b neg'>метка: НЕТ</span>")
        badges = [lab, f"<span class='b sc'>скор текста {r['text_score']:.3f}</span>",
                  f"<span class='b sc'>id {r['id']}</span>"]
        if not ims:
            badges.append("<span class='b noimg'>БЕЗ КАРТИНОК</span>")
        srcs = [embed(i) for i in ims[:MAX_IMGS]]
        imhtml = "".join(f"<img src='{u}' loading='lazy'>" for u in srcs if u)
        desc = html.escape(str(r["description"])[:600] if pd.notna(r["description"]) else "")
        parts.append(
            f"<div class=card><div class=imgs>{imhtml}</div><div class=txt>"
            f"<div class=badges>{''.join(badges)}</div>"
            f"<div class=nm>{html.escape(str(r['name']))}</div>"
            f"<div class=ds>{desc}</div></div></div>")

out = ROOT + f"gallery_{SLUG}.html"
open(out, "w", encoding="utf-8").write("\n".join(parts))
n = sum(min(PER, len(g)) for _, g in groups)
print(f"готово: {out}  ({n} карточек)")
print(f"товаров без изображений в категории: "
      f"{sum(1 for p in d['id'] if not imgs(p))} из {len(d)}")
