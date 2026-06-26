"""
build_model_cards.py — MODEL CARDS (metadata de cada modelo de produccion), junto a los .pkl.

Para cada (grupo, horizonte) con modelo guardado, consolida en artifacts/models/model_cards.json:
  - fecha_entrenamiento : mtime del bundle .pkl (cuando se entreno por ultima vez).
  - n_pares            : nº de pares de entrenamiento de ese (grupo, horizonte).
  - n_features / features : tamano y lista del set de features usado.
  - commit_git         : commit corto del repo (si git esta disponible).
  - skill_validado     : skill anidado [punto, lo, hi] del TEST INTACTO (nested_metrics.json).
  - pr_auc_alerta      : PR-AUC de alerta del test intacto, si esta.

NO reentrena nada: solo lee los bundles ya guardados + reportes. run_forecast.py lo incluye
en la columna 'modelo_meta' para trazabilidad de cada pronostico emitido.

Uso:  python build_model_cards.py
"""
from __future__ import annotations
import os, json, glob, subprocess, datetime
import joblib
import pandas as pd
import config as C
from train import PAIRS

OUT = os.path.join(C.DIR_MODELS, "model_cards.json")
NESTED = os.path.join(C.DIR_REPORTS, "nested_metrics.json")


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def build():
    commit = _git_commit()
    nested = json.load(open(NESTED, encoding="utf-8")) if os.path.exists(NESTED) else {}
    npairs = {}
    if os.path.exists(PAIRS):
        df = pd.read_csv(PAIRS)
        npairs = df.groupby(["group", "horizon"]).size().to_dict()

    cards = {}
    for pkl in sorted(glob.glob(os.path.join(C.DIR_MODELS, "*_h*.pkl"))):
        if pkl.endswith("_nn.pt"):
            continue
        b = joblib.load(pkl)
        group, h = b.get("group"), b.get("horizon")
        tag = f"{group}_h{h}"
        node = nested.get(group, {}).get(str(h), {})
        cards[tag] = {
            "grupo": group, "horizonte": h,
            "fecha_entrenamiento": datetime.datetime.fromtimestamp(
                os.path.getmtime(pkl)).strftime("%Y-%m-%d"),
            "n_pares": int(npairs.get((group, h), 0)),
            "n_features": len(b.get("feats", [])),
            "features": list(b.get("feats", [])),
            "commit_git": commit,
            "skill_validado": node.get("skill_nested"),
            "pr_auc_alerta": node.get("pr_auc_nested"),
            "tiene_intervalos_cqr": b.get("qlo") is not None and "q_conformal" in b,
        }
    json.dump(cards, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Model cards ({len(cards)}) -> {OUT}")
    return cards


if __name__ == "__main__":
    build()
