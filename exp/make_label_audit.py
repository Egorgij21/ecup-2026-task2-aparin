"""Аудит разметки: товары одного типа с РАЗНЫМИ метками, рядом, с картинками.

Зачем. Беглый взгляд на галерею создаёт впечатление шума в разметке: хлопушки
и свечи встречаются и с меткой 1, и с меткой 0. Но проверка по спичкам показала
обратное — там метки различают настоящие спички от сувенирной спичечницы «без
спичек» и от «вечной спички» (кремень, не горит). То есть правило применяется
последовательно, а различие лежит в детали, которой в названии может не быть.

Главная гипотеза, ради которой это собрано: хлопушки бывают ПНЕВМАТИЧЕСКИЕ
(сжатый воздух, пиротехники нет) и ПИРОТЕХНИЧЕСКИЕ. Правила относят к опасным
только вторые, по названию их почти не отличить — **а на упаковке написано**.
Если так, картинки дают ровно ту информацию, которой не хватает тексту.

Показывает по каждому типу товара позитивы и негативы В ДВУХ КОЛОНКАХ,
чтобы разницу было видно глазами. Картинки вшиты в HTML (data-URI): относительные
ссылки не работают при открытии через Jupyter или предпросмотр редактора.

Запуск: python exp/make_label_audit.py [БАД|Легковоспламеняющиеся] [строк на колонку]
"""
import base64
import html
import io
import os
import sys

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/workspace/counter/"
CAT = sys.argv[1] if len(sys.argv) > 1 else "Легковоспламеняющиеся"
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SLUG = "bad" if CAT == "БАД" else "flam"
THUMB, QUALITY, MAX_IMGS = 300, 70, 2

GROUPS_FLAM = {
    "Хлопушки — пневматика против пиротехники?": r"хлопушк",
    "Свечи для торта": r"свеч.{0,14}(для торта|в торт|цифр)|торт.{0,8}свеч",
    "Свечи прочие": r"свеч(?!.{0,14}(для торта|в торт|цифр))",
    "Бенгальские огни": r"бенгальск",
    "Цветной дым и дымовые шашки": r"цветной дым|дым.{0,4} шашк|дымов.{0,4} шашк",
    "Спички": r"спичк",
    "Зажигалки": r"зажигалк",
    "Розжиг и растопка": r"розжиг|растопк",
    "Уголь и брикеты": r"уголь|брикет",
    "Сухое горючее": r"сухое горюч",
    "Горелки": r"горелк",
    "Газ и баллоны": r"\bгаз|баллон",
    "Мангалы и грили": r"мангал|гриль|барбекю",
    "Наборы (подарочные, туристические, выживания)": r"набор",
}
GROUPS_BAD = {
    "Витамины": r"витамин",
    "Омега / рыбий жир": r"омега|рыбий жир|omega",
    "Коллаген": r"коллаген",
    "Магний / цинк / селен": r"магни|цинк|селен",
    "Протеин и гейнер": r"протеин|гейнер|whey",
    "Креатин / BCAA / аминокислоты": r"креатин|bcaa|аминокислот",
    "L-карнитин": r"карнитин",
    "Мелатонин и сон": r"мелатонин|для сна",
    "Пробиотики": r"пробиотик|лактобакт|бифид",
    "Экстракты и травы": r"экстракт|трав|настой",
    "Жиросжигатели": r"жиросжигат|для похудени",
    "Слово «БАД» прямо в названии": r"\bбад\b",
}
GROUPS = GROUPS_FLAM if CAT != "БАД" else GROUPS_BAD


def embed(pid, k):
    p = os.path.join(ROOT, "data/images", str(pid))
    if not os.path.isdir(p):
        return []
    fs = sorted(f for f in os.listdir(p) if f.lower().endswith(".jpg"))[:k]
    out = []
    for f in fs:
        try:
            im = Image.open(os.path.join(p, f)).convert("RGB")
        except Exception:
            continue
        im.thumbnail((THUMB, THUMB), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=QUALITY, optimize=True)
        im.close()
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
oof = pd.read_parquet(ROOT + "exp/text_oof.parquet")
d = df.merge(oof[["id", "text_score", "err"]], on="id")
d = d[d["category"] == CAT].reset_index(drop=True)
d["n"] = d["name"].fillna("").str.lower().str.replace("ё", "е")

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}
header{position:sticky;top:0;background:#fff;padding:14px 20px;border-bottom:1px solid #d2d2d7;z-index:20}
h1{margin:0;font-size:19px}
h2{margin:0;padding:16px 20px 8px;font-size:16px;background:#f5f5f7;position:sticky;top:62px;z-index:10}
.stat{font-size:12px;color:#6e6e73;font-weight:400;margin-left:8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 20px 8px}
.col h3{margin:4px 0 8px;font-size:13px;padding:5px 10px;border-radius:6px}
.yes h3{background:#d1f0d8;color:#0b6b2b} .no h3{background:#ffe0e0;color:#8b1a1a}
.it{background:#fff;border:1px solid #e5e5ea;border-radius:8px;padding:8px;margin-bottom:8px;display:flex;gap:8px}
.it img{height:96px;border-radius:5px;border:1px solid #eee;cursor:zoom-in;background:#fafafa}
.it img:hover{transform:scale(3);transform-origin:top left;position:relative;z-index:30;box-shadow:0 10px 34px rgba(0,0,0,.35)}
.t{font-size:12px;line-height:1.4;min-width:0}
.t b{display:block;margin-bottom:3px;font-size:12.5px}
.t .d{color:#8a8a8e;max-height:52px;overflow:hidden}
.sc{font-size:10.5px;color:#6e6e73;margin-top:3px}
"""

parts = [f"<!doctype html><meta charset=utf-8><title>Аудит разметки {CAT}</title><style>{CSS}</style>",
         f"<header><h1>Аудит разметки — «{html.escape(CAT)}»</h1>"
         "<div style='font-size:12px;color:#6e6e73;margin-top:3px'>Слева товары с меткой ДА, "
         "справа с меткой НЕТ, одного типа. Вопрос: видно ли на упаковке, чем они отличаются? "
         "Наведи на картинку — увеличится.</div></header>"]

for title, rx in GROUPS.items():
    g = d[d["n"].str.contains(rx, regex=True)]
    if len(g) < 6:
        continue
    pos, neg = g[g["label"] == 1], g[g["label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        continue
    parts.append(f"<h2>{html.escape(title)}<span class=stat> — всего {len(g)}, "
                 f"с меткой ДА {len(pos)} ({len(pos)/len(g):.0%}), с меткой НЕТ {len(neg)}</span></h2>")
    parts.append("<div class=cols>")
    for side, sub_, cls in [("метка ДА — легковоспламеняющийся" if CAT != "БАД" else "метка ДА — это БАД",
                             pos, "yes"),
                            ("метка НЕТ", neg, "no")]:
        parts.append(f"<div class='col {cls}'><h3>{html.escape(side)}</h3>")
        for _, r in sub_.head(PER).iterrows():
            ims = "".join(f"<img src='{u}' loading='lazy'>" for u in embed(r["id"], MAX_IMGS))
            parts.append(
                f"<div class=it><div>{ims}</div><div class=t>"
                f"<b>{html.escape(str(r['name'])[:120])}</b>"
                f"<div class=d>{html.escape(str(r['description'])[:200] if pd.notna(r['description']) else '')}</div>"
                f"<div class=sc>скор текста {r['text_score']:.3f} · {r['err']} · id {r['id']}</div>"
                f"</div></div>")
        parts.append("</div>")
    parts.append("</div>")

out = ROOT + f"audit_{SLUG}.html"
open(out, "w", encoding="utf-8").write("\n".join(parts))
print(f"готово: {out}")
