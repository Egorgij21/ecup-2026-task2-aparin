"""Комплементарность флам-промптов: можно ли БЛЕНДОМ разных промптов взять сильные
стороны каждого. Идея — декорреляция по ПРОМПТАМ (как v10 по базам дал +0.009).

Данные: zero-shot скоры каждого промпта по каждому товару (flam_prompt_screen_* и
flam_prompt_surgical_*). base лучший по AUC, но пропускает газовые горелки с пьезо
(0.00); types/piezo их ловят (0.97), но задирают негативы. Проверяем, комбинирует ли
бленд эти силы, ДО траты GPU на обучение.

ВАЖНО: это zero-shot прокси. Сабмит использует ОБУЧЕННЫЕ адаптеры (обучение калибрует,
негативы не так задираются). Но zero-shot комплементарность — дешёвый первый сигнал:
если декорреляция есть уже здесь, обучать бленд осмысленно.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = "/workspace/counter/"


def load(base):
    a = pd.read_parquet(ROOT + f"exp/flam_prompt_screen_{base}.parquet")
    b = pd.read_parquet(ROOT + f"exp/flam_prompt_surgical_{base}.parquet")[
        ["id", "s_piezo", "s_piezo_pneu"]]
    m = a.merge(b, on="id")
    return m.rename(columns={c: f"{base}_{c[2:]}" for c in m.columns if c.startswith("s_")})


g = load("gemma")
q = load("qwen")[["id", "qwen_base", "qwen_types", "qwen_piezo", "qwen_piezo_pneu"]]
df = g.merge(q, on="id")
y = df["label"].values
burner = df["name"].str.contains("горелк|баллон|газ|резак|паяльн", case=False, na=False).values
gas_pos = burner & (y == 1)
print(f"флам {len(df)}  поз {y.sum()}  газовых-поз {int(gas_pos.sum())}\n")

cols = ["gemma_base", "gemma_types", "gemma_piezo", "qwen_base", "qwen_types", "qwen_piezo"]


def auc(s):
    return roc_auc_score(y, s)


def summ(name, s):
    gr = (s[gas_pos] >= 0.5).mean()
    np90 = np.quantile(s[y == 0], 0.90)
    print(f"  {name:34s} AUC={auc(s):.4f} PR={average_precision_score(y,s):.4f} "
          f"газ-recall={gr:.3f} нег-p90={np90:.3f}")


print("=== одиночные ===")
for c in cols:
    summ(c, df[c].values)

print("\n=== КОРРЕЛЯЦИИ скоров (ниже corr -> сильнее декорреляция) ===")
C = df[cols].corr(method="spearman")
print("  текущий бленд baseQ×baseG corr =", f"{C.loc['gemma_base','qwen_base']:.3f}")
print("  gemma_base × gemma_types    corr =", f"{C.loc['gemma_base','gemma_types']:.3f}")
print("  gemma_base × gemma_piezo    corr =", f"{C.loc['gemma_base','gemma_piezo']:.3f}")
print("  gemma_base × qwen_types     corr =", f"{C.loc['gemma_base','qwen_types']:.3f}")

print("\n=== БЛЕНДЫ (grid веса, лучший по AUC) ===")
def best_blend(a, b, label):
    best = None
    for w in np.linspace(0, 1, 21):
        s = w * df[a].values + (1 - w) * df[b].values
        best = max(best or (0, 0), (auc(s), w))
    w = best[1]
    summ(f"{label} w={w:.2f}", w * df[a].values + (1 - w) * df[b].values)

# текущая пара (обе base, разные базы) — эталон декорреляции
best_blend("gemma_base", "qwen_base", "baseG⊕baseQ (текущий)")
# добавляем прицельный промпт
best_blend("gemma_base", "gemma_types", "baseG⊕typesG")
best_blend("gemma_base", "gemma_piezo", "baseG⊕piezoG")
best_blend("gemma_base", "qwen_types", "baseG⊕typesQ")
best_blend("gemma_base", "qwen_piezo", "baseG⊕piezoQ")

print("\n=== ТРОЙНОЙ бленд base_g + base_q + прицельный ===")
for third in ["gemma_types", "gemma_piezo", "qwen_types", "qwen_piezo"]:
    best = None
    for w3 in np.linspace(0, 0.6, 13):
        s = (1 - w3) * 0.5 * (df["gemma_base"].values + df["qwen_base"].values) + w3 * df[third].values
        best = max(best or (0, 0), (auc(s), w3))
    w3 = best[1]
    s = (1 - w3) * 0.5 * (df["gemma_base"].values + df["qwen_base"].values) + w3 * df[third].values
    summ(f"baseG+baseQ + {third} w3={w3:.2f}", s)

print("\n=== 'СПАСЕНИЕ' газовых: base, но где base<0.3 и прицельный>0.8 -> поднять ===")
for tgt in ["gemma_types", "gemma_piezo", "qwen_piezo"]:
    s = df["gemma_base"].values.copy()
    rescue = (s < 0.3) & (df[tgt].values > 0.8)
    added_tp = int((rescue & (y == 1)).sum())
    added_fp = int((rescue & (y == 0)).sum())
    s2 = s.copy()
    s2[rescue] = np.maximum(s2[rescue], df[tgt].values[rescue])
    print(f"  rescue<-{tgt}: поднято {rescue.sum()} строк (+TP {added_tp}, +FP {added_fp}), "
          f"AUC {auc(df['gemma_base'].values):.4f}->{auc(s2):.4f}")
