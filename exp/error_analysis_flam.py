"""На чём именно ошибается текущий бленд на флам-категории?

Нужно перед аугментацией: генерировать данные вслепую — значит с равной
вероятностью усилить то, что и так работает. Смотрим, чего не хватает:
пропусков (модель не узнаёт класс товара) или ложных срабатываний
(модель цепляется за слово-триггер).

Печатается также, сколько ошибок приходится на семьи, целиком отсутствующие
в train-фолде: если пропуск — это НОВАЯ семья, аугментация новыми товарами
осмысленна. Если модель валит семьи, которые видела, — дело не в разнообразии.
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = "/workspace/counter/"
sys.path.insert(0, ROOT + "exp")
from folds import clean, family_labels, make_folds  # noqa: E402
from rule_features import extract  # noqa: E402

THS = np.linspace(0.005, 0.995, 199)
CAT = "Легковоспламеняющиеся"

df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(ROOT + "exp/llm_scores.parquet")[["id", "llm_score"]], on="id")
lora = pd.concat([pd.read_parquet(p)[["id", "lora_score"]]
                  for p in sorted(glob.glob(ROOT + "exp/lora_oof_fold[0-9].parquet"))])
df = df.merge(lora.drop_duplicates(subset="id"), on="id", how="left")
df["fold"] = make_folds(df)
df["name_c"] = df["name"].fillna("").map(clean)
df["desc_c"] = df["description"].fillna("").map(clean)
df["full"] = df["name_c"] + " " + df["desc_c"]
df["txt_in"] = (df["name_c"] + " ") * 3 + df["desc_c"]

sub = df[df["category"] == CAT].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
fam = family_labels(sub["name"].fillna("").values)
lo = sub["lora_score"].values
R = extract(sub["full"].values, sub["name_c"].values, CAT)
L = np.c_[np.clip(sub["llm_score"].values, 1e-4, 1 - 1e-4)]
L = np.c_[L, np.log(L / (1 - L)), (L > .5).astype(float), (L > .8).astype(float),
          (L < .2).astype(float)]

txt = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([v.fit_transform(sub["txt_in"].values[tr]), csr_matrix(R[tr]),
                  csr_matrix(L[tr])]).tocsr()
    Xte = hstack([v.transform(sub["txt_in"].values[te]), csr_matrix(R[te]),
                  csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced")\
        .fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

score = 0.5 * txt + 0.5 * lo
pred = np.zeros(len(y), dtype=int)
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    f = np.array([f1_score(y[tr], (score[tr] >= t).astype(int)) for t in THS])
    pred[te] = (score[te] >= THS[int(f.argmax())]).astype(int)

fn = np.where((y == 1) & (pred == 0))[0]
fp = np.where((y == 0) & (pred == 1))[0]
print(f"F1={f1_score(y, pred):.4f}  позитивов {y.sum()} в {len(set(fam[y == 1]))} семьях")
print(f"пропуски (FN): {len(fn)}   ложные (FP): {len(fp)}")

pos_fams = set(fam[y == 1])
lone = sum(1 for i in fn if (fam[fam == fam[i]].size == 1))
print(f"\nиз {len(fn)} пропусков в семьях-одиночках (1 товар): {lone}")
seen = 0
for i in fn:
    k = fold[i]
    tr = np.where(fold != k)[0]
    if fam[i] in set(fam[tr][y[tr] == 1]):
        seen += 1
print(f"пропусков, чья семья ЕСТЬ среди позитивов train-фолда: {seen}/{len(fn)}")
print("  -> если это число мало, модели не хватает РАЗНООБРАЗИЯ семей,")
print("     и генерация новых товаров осмысленна; если велико — дело не в нём.")

print("\n" + "=" * 96)
print("ПРОПУСКИ (модель сказала «не опасно», разметка — опасно), по возрастанию скора")
for i in fn[np.argsort(score[fn])][:25]:
    print(f"  [{score[i]:.3f} txt={txt[i]:.2f} lora={lo[i]:.2f}] {str(sub['name'][i])[:88]}")

print("\n" + "=" * 96)
print("ЛОЖНЫЕ СРАБАТЫВАНИЯ (разметка — не опасно), по убыванию скора")
for i in fp[np.argsort(-score[fp])][:25]:
    print(f"  [{score[i]:.3f} txt={txt[i]:.2f} lora={lo[i]:.2f}] {str(sub['name'][i])[:88]}")

print("\n" + "=" * 96)
print("ПОГРАНИЧНАЯ ЗОНА: истинные позитивы со скором 0.2..0.6 (модель не уверена)")
band = np.where((y == 1) & (score > 0.2) & (score < 0.6))[0]
for i in band[:15]:
    print(f"  [{score[i]:.3f}] {str(sub['name'][i])[:88]}")
print(f"  всего в зоне: {len(band)}")
