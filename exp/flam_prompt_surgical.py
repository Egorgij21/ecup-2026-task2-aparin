"""ХИРУРГИЧЕСКИЙ скрининг флам-промпта: официальные правила + 1-2 уточняющие строки.

Из flam_prompt_screen.py: полная ЗАМЕНА правил прицельным списком типов рушит AUC
(0.912 -> 0.74) — модель начинает говорить «Да» на всё (негативы p90 0.95 -> 1.00),
хотя целевой кластер (газовые горелки с пьезо) чинит (base 0.00 -> 0.99). Значит
правила заменять нельзя, но можно ТОЧЕЧНО снять конкретное противоречие, не подмешивая
общий «список горючего», который смещает к «Да».

Проверяем добавки к дословным официальным правилам:
  * piezo   — газовая горелка/резак флам ДАЖЕ с пьезоподжигом (крупнейший FN-кластер);
  * piezo_pneu — + хлопушки на сжатом воздухе (пневматика) НЕ флам (крупнейший FP);
  * piezo_coal — + уголь и топливные брикеты флам (FN: Weber, берёзовый уголь ~0.07).
Критерий отбора ДО просмотра: вариант годится, только если AUC >= base (0.912 gemma /
0.897 qwen) И средний скор на 37 газовых позитивах вырос, а p90 негативов НЕ вырос.

Запуск: CUDA_VISIBLE_DEVICES=0 python exp/flam_prompt_surgical.py [qwen|gemma]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, "/workspace/counter/exp")
from flam_prompt_screen import (BS, CAT, ROOT, RULES_BASE, SYSTEM,  # noqa: E402
                                build_prompt, clean)

BASE = (sys.argv[1] if len(sys.argv) > 1 else "qwen").lower()
MODEL = "Qwen/Qwen3.5-4B" if BASE == "qwen" else "google/gemma-4-E4B-it"

_PIEZO = ("\n\nУточнение: газовая горелка, газовый резак, паяльная лампа и газовый баллон "
          "являются легковоспламеняющимися ДАЖЕ если снабжены пьезоподжигом или встроенным "
          "поджигом — значение имеет наличие горючего газа, а не способ его поджига.")
_PNEU = ("\n\nУточнение: хлопушки, пневмохлопушки и конфетти-пушки, работающие на сжатом "
         "ВОЗДУХЕ (пневматические, без пиротехнического состава), НЕ являются "
         "легковоспламеняющимися.")
_COAL = ("\n\nУточнение: древесный уголь, угольные и топливные брикеты для мангала, гриля и "
         "барбекю являются легковоспламеняющимися.")

VARIANTS = {
    "base": RULES_BASE,
    "piezo": RULES_BASE + _PIEZO,
    "piezo_pneu": RULES_BASE + _PIEZO + _PNEU,
    "piezo_coal": RULES_BASE + _PIEZO + _COAL,
    "piezo_pneu_coal": RULES_BASE + _PIEZO + _PNEU + _COAL,
}


def main():
    df = pd.read_csv(ROOT + "data/data.csv").drop(columns=["Unnamed: 0"])
    df = df[df["category"] == CAT].reset_index(drop=True)
    y = df["label"].values
    burner = df["name"].str.contains("горелк|баллон|газ|резак|паяльн", case=False, na=False).values
    print(f"[{BASE}] {MODEL}  флам-строк {len(df)}  позитивов {y.sum()}  газовых-поз {int((burner&(y==1)).sum())}",
          flush=True)

    if BASE == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                     local_files_only=True).cuda().eval()
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        tok = proc.tokenizer
        tok.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                            local_files_only=True).cuda().eval()

    import inspect
    _f = inspect.signature(model.forward).parameters
    keep = ({"logits_to_keep": 1} if "logits_to_keep" in _f
            else {"num_logits_to_keep": 1} if "num_logits_to_keep" in _f else {})

    def ids_for(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t:
                    out.add(t[0])
        return sorted(out)
    yes_ids, no_ids = ids_for(["Да", "да"]), ids_for(["Нет", "нет"])

    names, descs = df["name"].fillna("").tolist(), df["description"].fillna("").tolist()
    rows = []
    for vname, rules in VARIANTS.items():
        prompts = [build_prompt(tok, rules, n, d) for n, d in zip(names, descs)]
        order = np.argsort([len(p) for p in prompts])
        scores = np.zeros(len(prompts), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(order), BS):
                ch = order[i:i + BS]
                enc = tok([prompts[j] for j in ch], return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to("cuda")
                lp = torch.log_softmax(model(**enc, **keep).logits[:, -1, :].float(), dim=-1)
                yy = torch.logsumexp(lp[:, yes_ids], dim=-1)
                nn = torch.logsumexp(lp[:, no_ids], dim=-1)
                scores[ch] = torch.sigmoid(yy - nn).cpu().numpy()
        df[f"s_{vname}"] = scores
        ths = np.linspace(0.01, 0.99, 99)
        fb, tb = max((f1_score(y, (scores >= t).astype(int)), t) for t in ths)
        auc, pr = roc_auc_score(y, scores), average_precision_score(y, scores)
        gas_pos = scores[burner & (y == 1)].mean()
        neg_p90 = np.quantile(scores[y == 0], 0.90)
        rows.append((vname, auc, pr, fb, tb, gas_pos, neg_p90))
        print(f"MACHINE\t{BASE}\t{vname}\tAUC={auc:.4f}\tPR={pr:.4f}\tF1bin={fb:.4f}@{tb:.2f}\t"
              f"газ-поз={gas_pos:.3f}\tнег-p90={neg_p90:.3f}\t{time.time()-t0:.0f}s", flush=True)

    df.to_parquet(ROOT + f"exp/flam_prompt_surgical_{BASE}.parquet")
    b = next(r for r in rows if r[0] == "base")
    print(f"\n=== ИТОГ [{BASE}] (base: AUC={b[1]:.4f} газ-поз={b[5]:.3f} нег-p90={b[6]:.3f}) ===")
    for vname, auc, pr, fb, tb, gp, np90 in sorted(rows, key=lambda r: -r[1]):
        ok = "✓" if (auc >= b[1] - 1e-4 and gp > b[5] and np90 <= b[6] + 1e-4) else " "
        print(f"  {ok} {vname:16s} AUC={auc:.4f} (Δ{auc-b[1]:+.4f}) PR={pr:.4f} "
              f"газ-поз={gp:.3f} нег-p90={np90:.3f}")
    print("критерий ✓: AUC не ниже base, газ-позитивы выросли, p90 негативов не вырос")


if __name__ == "__main__":
    main()
