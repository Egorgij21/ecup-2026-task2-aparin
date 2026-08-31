"""E2E-прогон сабмита на фейковом test.csv + строгая валидация формата вывода."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path("/workspace/counter")
SUBMIT = ROOT / "submit4"
PY = str(ROOT / ".venv/bin/python")

full = pd.read_csv(ROOT / "data/data.csv").drop(columns=["Unnamed: 0"])
tmp = Path(tempfile.mkdtemp())

# test.csv БЕЗ колонки label — как у организаторов
test = full.sample(n=min(5400, len(full)), random_state=123).reset_index(drop=True)
truth = test[["id", "category", "label"]].copy()
test.drop(columns=["label"]).to_csv(tmp / "test.csv", index=False)
(tmp / "images").mkdir(exist_ok=True)
print(f"фейковый тест: {len(test)} строк -> {tmp/'test.csv'}")

out_path = tmp / "submit.csv"
r = subprocess.run([PY, "run.py", "--test_data_path", str(tmp / "test.csv"),
                    "--output_path", str(out_path)],
                   cwd=SUBMIT, capture_output=True, text=True)
print("\n--- stdout ---\n" + r.stdout)
if r.returncode != 0:
    print("--- stderr ---\n" + r.stderr)
    sys.exit(1)

# ------------------------------------------------------------------ валидация
sub = pd.read_csv(out_path)
errs = []
if list(sub.columns) != ["id", "result"]:
    errs.append(f"колонки: {list(sub.columns)}, ожидались ['id','result']")
if len(sub) != len(test):
    errs.append(f"строк {len(sub)}, ожидалось {len(test)}")
if set(sub["id"]) != set(test["id"]):
    errs.append("множество id не совпадает с тестовым")
if sub["id"].duplicated().any():
    errs.append("есть дубли id")

PAT = re.compile(r"^<комментарий>(.+)<вердикт>(бан|не бан)$", re.S)
lens, verdicts, bad_fmt = [], [], 0
for v in sub["result"]:
    m = PAT.match(str(v))
    if not m:
        bad_fmt += 1
        if bad_fmt <= 3:
            errs.append(f"не matched формат: {str(v)[:120]!r}")
        continue
    c, verd = m.group(1), m.group(2)
    lens.append(len(c))
    verdicts.append(verd)
    if not (50 <= len(c) <= 300):
        errs.append(f"длина комментария {len(c)}: {c[:80]!r}")
    if "<" in c or ">" in c:
        errs.append(f"теги внутри комментария: {c[:80]!r}")

lens = np.array(lens)
print(f"формат ок: {len(lens)}/{len(sub)}")
print(f"длина комментария: min={lens.min()} p50={np.median(lens):.0f} max={lens.max()}")
print(f"вердикты: {pd.Series(verdicts).value_counts().to_dict()}")

# ------------------------------------------------------------------ метрика
sub["pred"] = [1 if str(v).endswith("не бан") else 0 for v in sub["result"]]
mg = truth.merge(sub[["id", "pred"]], on="id")
print("\n--- метрика на фейковом тесте (данные видел при обучении, цифры завышены) ---")
b, m_ = [], []
for cat, g in mg.groupby("category"):
    fb = f1_score(g["label"], g["pred"])
    fm = f1_score(g["label"], g["pred"], average="macro")
    b.append(fb)
    m_.append(fm)
    print(f"  {cat}: F1bin={fb:.4f} F1macro={fm:.4f} n={len(g)}")
print(f"  mean binary = {np.mean(b):.4f}   mean macro = {np.mean(m_):.4f}")

# ------------------------------------------------------------------ краевые случаи
print("\n--- краевые случаи ---")
edge = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "name": ["", "а", "Спички ГОСТ", "БАД витамин D3"],
    "description": [None, "", "<br/><ul><li>уголь в комплекте</li></ul>", None],
    "category": ["БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся", "БАД"],
})
edge.to_csv(tmp / "edge.csv", index=False)
r2 = subprocess.run([PY, "run.py", "-i", str(tmp / "edge.csv"), "-o", str(tmp / "edge_out.csv")],
                    cwd=SUBMIT, capture_output=True, text=True)
if r2.returncode != 0:
    print("ПАДЕНИЕ на краевых случаях:\n", r2.stderr[-2000:])
    errs.append("падение на краевых случаях")
else:
    eo = pd.read_csv(tmp / "edge_out.csv")
    for _, row in eo.iterrows():
        m = PAT.match(str(row["result"]))
        status = f"len={len(m.group(1))} verdict={m.group(2)}" if m else "ФОРМАТ СЛОМАН"
        print(f"  id={row['id']}: {status}")
        print(f"     {str(row['result'])[:170]}")
        if not m or not (50 <= len(m.group(1)) <= 300):
            errs.append(f"краевой случай id={row['id']} невалиден")

print("\n" + "=" * 80)
if errs:
    print("ОШИБКИ:")
    for e in errs[:20]:
        print("  -", e)
    sys.exit(1)
print("ВСЁ ВАЛИДНО")
