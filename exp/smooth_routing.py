"""Плавная маршрутизация: вес текста как функция расхождения адаптеров.

Жёсткое переключение (на расхождении -> текст) проиграло: 0.9004 против 0.9047
у геометрического среднего. Причина, скорее всего, в том, что переключение рвёт
шкалу — скор перестаёт быть сравнимым между зонами, и единый порог не переносится.

Но факт остаётся: в зоне расхождения (56 строк, 31 позитив из 198) текст даёт
AUC 0.8284, а адаптеры 0.5677 и 0.5503, то есть практически монетку. Значит
доля текста там должна быть выше — но БЕЗ разрыва шкалы.

Формула: w_lora = 0.5 * (1 - alpha * d), где d = |p_qwen - p_gemma|.
alpha=0 воспроизводит текущий v11 в точности, что и служит контролем.

Смотрим не только F1, но и разброс вложенного порога по фолдам: именно он
показывает, цела ли двухкластерность, и именно он у геометрического среднего 0.000.
"""
import sys
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
R = "/workspace/counter/"
sys.path.insert(0, R + "exp")
from folds import clean, make_folds
from rule_features import extract
TH = np.linspace(.005, .995, 199); CAT = "Легковоспламеняющиеся"
def load(t): return pd.concat([pd.read_parquet(R+f"exp/lora_oof_fold{k}{t}.parquet")[["id","lora_score"]] for k in range(5)]).rename(columns={"lora_score": t or "q"})
df = pd.read_csv(R+"data/data.csv").drop(columns=["Unnamed: 0"])
df = df.merge(pd.read_parquet(R+"exp/llm_scores.parquet")[["id","llm_score"]], on="id")
for t in ["", "_gemma"]: df = df.merge(load(t), on="id")
df["fold"] = make_folds(df)
df["nc"] = df["name"].fillna("").map(clean); df["dc"] = df["description"].fillna("").map(clean)
df["full"] = df["nc"]+" "+df["dc"]; df["ti"] = (df["nc"]+" ")*3+df["dc"]
sub = df[df["category"] == CAT].reset_index(drop=True)
y, fold = sub["label"].values, sub["fold"].values
q, g = sub["q"].values, sub["_gemma"].values
def pf(s):
    s = np.clip(np.asarray(s, float), 1e-4, 1-1e-4)
    return np.c_[s, np.log(s/(1-s)), (s>.5)*1., (s>.8)*1., (s<.2)*1.]
Rr = extract(sub["full"].values, sub["nc"].values, CAT); L = pf(sub["llm_score"].values)
txt = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=200_000)
    A = hstack([v.fit_transform(sub["ti"].values[tr]), csr_matrix(Rr[tr]), csr_matrix(L[tr])]).tocsr()
    B = hstack([v.transform(sub["ti"].values[te]), csr_matrix(Rr[te]), csr_matrix(L[te])]).tocsr()
    txt[te] = LogisticRegression(max_iter=4000, C=10., class_weight="balanced").fit(A, y[tr]).predict_proba(B)[:,1]
geo = np.sqrt(np.clip(q,1e-9,1)*np.clip(g,1e-9,1)); d = np.abs(q-g)
def rep(tag, o):
    pred = np.zeros(len(y), int); ths = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        f = np.array([f1_score(y[tr], (o[tr] >= t).astype(int)) for t in TH]); t = TH[int(f.argmax())]
        ths.append(t); pred[te] = (o[te] >= t).astype(int)
    F = f1_score(y, pred)
    print("  %-34s F1=%.4f мет=%.4f | P=%.3f R=%.3f | разброс порога %.3f"
          % (tag, F, (F+0.9378)/2, precision_score(y, pred), recall_score(y, pred), max(ths)-min(ths)))
    return F
print("Плавное снижение веса LoRA при расхождении: w = 0.5*(1 - alpha*|q-g|)")
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w = 0.5*(1 - a*d); rep(f"alpha={a} (0 = текущий v11)", (1-w)*txt + w*geo)
print()
print("Другие способы собрать три источника:")
rep("геом. трёх (текст,qwen,gemma)", np.cbrt(np.clip(txt,1e-9,1)*np.clip(q,1e-9,1)*np.clip(g,1e-9,1)))
rep("текст^0.5 * геом^0.5", np.sqrt(np.clip(txt,1e-9,1)*np.clip(geo,1e-9,1)))
rep("v10 арифм. (сабмит)", .5*txt+.5*(q+g)/2)
rep("v11 геом. (собран)", .5*txt+.5*geo)
