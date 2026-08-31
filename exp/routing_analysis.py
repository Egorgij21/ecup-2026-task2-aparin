"""Кто из компонентов где работает и стоит ли маршрутизировать вместо усреднения.

Мотив. Скоры адаптеров почти двоичные (97% у краёв). Значит вся содержательная
работа происходит там, где они РАСХОДЯТСЯ, — и сейчас мы это разногласие просто
усредняем. Геометрическое среднее оказалось лучше арифметического именно потому,
что трактует расхождение как «нет», а не как «0.5». Логичный следующий вопрос:
а не лучше ли на расхождении СПРАШИВАТЬ ТЕКСТ, вместо того чтобы решать за него?

Что считаем:
  1) precision / recall каждого инструмента по отдельности при вложенном пороге —
     кто точен, кто полон;
  2) разбиение строк по картине согласия двух адаптеров (оба «да», оба «нет»,
     расхождение) и доля позитивов в каждой части;
  3) кто прав в зоне расхождения — адаптеры или текст;
  4) правила маршрутизации против нынешнего усреднения, всё вложенно.

Все пороги подбираются на фолдах != k и применяются к k. Правила фиксируются
заранее, а не выбираются по результату: пятикратный перебор правил на тех же
данных уже стоил нам +0.0076 завышения на весах бленда.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THS = np.linspace(0.005, 0.995, 199)
CAT = "Легковоспламеняющиеся"


def load(t):
    return pd.concat([pd.read_parquet(ROOT + f"exp/lora_oof_fold{k}{t}.parquet")
                      [["id", "lora_score"]] for k in range(5)]).rename(
        columns={"lora_score": t or "q"})


df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
for t in ["", "_gemma"]:
    df = df.merge(load(t), on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean)
df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"] + " " + df["dc"]
df["ti"] = (df["nc"] + " ") * 3 + df["dc"]
sub = df[df["category"] == CAT].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
q, g = sub["q"].values, sub["_gemma"].values


def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > .5) * 1., (s > .8) * 1., (s < .2) * 1.]


R = extract(sub["full"].values, sub["nc"].values, CAT)
L = pf(sub["llm_score"].values)
txt = np.zeros(len(y))
rules_only = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(R[tr]),
                csr_matrix(L[tr])]).tocsr()
    B = hstack([v.transform(sub["ti"].values[te]), csr_matrix(R[te]),
                csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(A, y[tr]).predict_proba(B)[:, 1]
    m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(R[tr], y[tr])
    rules_only[te] = m.predict_proba(R[te])[:, 1]


def nested_pred(o):
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        f = np.array([f1_score(y[tr], (o[tr] >= t).astype(int)) for t in THS])
        pred[te] = (o[te] >= THS[int(f.argmax())]).astype(int)
    return pred


geo = np.sqrt(np.clip(q, 1e-9, 1) * np.clip(g, 1e-9, 1))
comps = {"только правила": rules_only, "текст (TF-IDF+правила+zs)": txt,
         "Qwen LoRA": q, "gemma LoRA": g,
         "бленд арифм. (v10)": 0.5 * txt + 0.5 * (q + g) / 2,
         "бленд геом. (v11)": 0.5 * txt + 0.5 * geo}

print("1) ТОЧНОСТЬ И ПОЛНОТА при вложенном пороге, флам "
      f"(n={len(y)}, позитивов {y.sum()})")
print(f"   {'инструмент':28s} {'precision':>10s} {'recall':>8s} {'F1':>8s} "
      f"{'ложных':>8s} {'пропусков':>10s}")
preds = {}
for tag, o in comps.items():
    p = nested_pred(o)
    preds[tag] = p
    print(f"   {tag:28s} {precision_score(y, p):10.4f} {recall_score(y, p):8.4f} "
          f"{f1_score(y, p):8.4f} {int(((p == 1) & (y == 0)).sum()):8d} "
          f"{int(((p == 0) & (y == 1)).sum()):10d}")

both_yes = (q > 0.5) & (g > 0.5)
both_no = (q <= 0.5) & (g <= 0.5)
disag = ~both_yes & ~both_no
print(f"\n2) КАРТИНА СОГЛАСИЯ ДВУХ АДАПТЕРОВ")
print(f"   {'зона':22s} {'строк':>7s} {'позитивов':>10s} {'доля':>8s}")
for tag, m in [("оба «да»", both_yes), ("РАСХОЖДЕНИЕ", disag), ("оба «нет»", both_no)]:
    print(f"   {tag:22s} {int(m.sum()):7d} {int(y[m].sum()):10d} {y[m].mean():8.1%}")

print(f"\n3) КТО ПРАВ В ЗОНЕ РАСХОЖДЕНИЯ (n={int(disag.sum())}, "
      f"позитивов {int(y[disag].sum())})")
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
for tag, o in [("Qwen", q), ("gemma", g), ("текст", txt), ("только правила", rules_only)]:
    if 0 < y[disag].sum() < disag.sum():
        print(f"   {tag:16s} AUC={roc_auc_score(y[disag], o[disag]):.4f} "
              f"PR={average_precision_score(y[disag], o[disag]):.4f}")

print("\n4) ПРАВИЛА МАРШРУТИЗАЦИИ против усреднения (вложенно, порог на train-фолдах)")


def routed(rule):
    """rule(idx) -> скор; порог подбирается на train-фолдах по этому же правилу."""
    o = rule
    pred = np.zeros(len(y), dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        f = np.array([f1_score(y[tr], (o[tr] >= t).astype(int)) for t in THS])
        pred[te] = (o[te] >= THS[int(f.argmax())]).astype(int)
    return f1_score(y, pred)


# на согласии доверяем адаптерам, на расхождении отдаём решение тексту
route_txt = np.where(disag, txt, geo)
# то же, но на расхождении смешиваем текст с геометрическим
route_mix = np.where(disag, 0.5 * txt + 0.5 * geo, geo)
# на согласии адаптеры, на расхождении правила
route_rules = np.where(disag, rules_only, geo)
for tag, o in [("бленд арифм. (v10, сабмит)", 0.5 * txt + 0.5 * (q + g) / 2),
               ("бленд геом. (v11)", 0.5 * txt + 0.5 * geo),
               ("на расхождении -> текст", route_txt),
               ("на расхождении -> текст+геом", route_mix),
               ("на расхождении -> правила", route_rules)]:
    print(f"   {tag:32s} F1={routed(o):.4f}   метрика={(routed(o) + 0.9378) / 2:.4f}")
print("\n   ориентир: v10 на паблике 0.899479")
