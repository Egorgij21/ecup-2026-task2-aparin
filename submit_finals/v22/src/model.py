"""Линейная модель ансамбля: [TF-IDF | правила | LLM] -> сигмоида.

TF-IDF реализован вручную (см. v1): версия sklearn в докер-образе неизвестна,
а падение на распиковке стоит слота сабмита. Совпадение с sklearn проверено
скриптом exp/verify_vectorizer_v2.py.
"""
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(s) -> str:
    s = str(s) if s is not None else ""
    if s == "nan":
        s = ""
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip().lower()


def build_text(name, desc, name_rep: int = 3) -> str:
    n, d = clean_text(name), clean_text(desc)
    return ((n + " ") * name_rep + d).strip()


def _analyze(text: str, max_n: int) -> List[str]:
    tokens = _TOKEN_RE.findall(text)
    if max_n == 1:
        return tokens
    out = list(tokens)
    n0 = len(tokens)
    for n in range(2, max_n + 1):
        for i in range(n0 - n + 1):
            out.append(" ".join(tokens[i:i + n]))
    return out


def llm_features(scores) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.c_[s, np.log(s / (1 - s)), (s > 0.5).astype(float),
                 (s > 0.8).astype(float), (s < 0.2).astype(float)]


class CategoryModel:
    def __init__(self, terms, idf, entry, npz, max_ngram):
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(terms)}
        self.idf = np.asarray(idf, dtype=np.float64)
        self.max_ngram = max_ngram
        self.coef_tfidf_llm = npz["coef_tfidf_llm"].astype(np.float64)
        self.coef_rules_llm = npz["coef_rules_llm"].astype(np.float64)
        self.coef_llm = npz["coef_llm"].astype(np.float64)
        self.coef_tfidf_nollm = npz["coef_tfidf_nollm"].astype(np.float64)
        self.coef_rules_nollm = npz["coef_rules_nollm"].astype(np.float64)
        self.intercept_llm = float(entry["intercept_llm"])
        self.intercept_nollm = float(entry["intercept_nollm"])
        self.threshold_llm = float(entry["threshold_llm"])
        self.threshold_nollm = float(entry["threshold_nollm"])

    def _tfidf_dot(self, text: str, coef: np.ndarray) -> float:
        counts: Dict[int, int] = {}
        for term in _analyze(text, self.max_ngram):
            j = self.vocab.get(term)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return 0.0
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        vals = (1.0 + np.log(tf)) * self.idf[idx]
        norm = np.sqrt((vals * vals).sum())
        if norm > 0:
            vals /= norm
        return float(vals @ coef[idx])

    def score(self, text: str, rules: np.ndarray, llm_feat, use_llm: bool) -> float:
        if use_llm:
            z = (self._tfidf_dot(text, self.coef_tfidf_llm)
                 + float(rules @ self.coef_rules_llm)
                 + float(np.asarray(llm_feat) @ self.coef_llm)
                 + self.intercept_llm)
        else:
            z = (self._tfidf_dot(text, self.coef_tfidf_nollm)
                 + float(rules @ self.coef_rules_nollm)
                 + self.intercept_nollm)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def threshold(self, use_llm: bool) -> float:
        return self.threshold_llm if use_llm else self.threshold_nollm


def load_models(art_dir: Path):
    meta = json.load(open(art_dir / "meta.json", encoding="utf-8"))
    max_ngram = int(meta["config"]["word_ng"][1])
    name_rep = int(meta["config"]["name_rep"])
    models = {}
    for cat, entry in meta["categories"].items():
        terms = json.load(open(art_dir / f"vocab_{entry['slug']}.json", encoding="utf-8"))
        npz = np.load(art_dir / f"model_{entry['slug']}.npz")
        models[cat] = CategoryModel(terms, npz["idf"], entry, npz, max_ngram)
    return models, name_rep, meta
