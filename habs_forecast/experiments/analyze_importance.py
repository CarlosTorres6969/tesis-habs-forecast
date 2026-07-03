"""
analyze_importance.py — Importancia de features por grupo y horizonte (diagnostico).

Objetivo: entender QUE impulsa el pronostico y detectar fuga oculta. Si una feature espectral
dominara de forma irreal seria sospechoso; lo esperado es que el backbone autorregresivo
(chl reciente) mande en h corto y los drivers ERA5 ganen peso en h largo.

Usa importancia por ganancia de XGBoost + (opcional) permutacion sobre un holdout temporal.
Salida por consola + artifacts/reports/feature_importance.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C
from train import FEATURES, _model, PAIRS

OUT = os.path.join(C.DIR_REPORTS, "feature_importance.csv")


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    feats = [f for f in FEATURES if f in df.columns]
    recs = []
    for group in ("freshwater", "marine"):
        for h in (1, 3, 7):                      # horizontes de pronostico (no h=0 degenerado)
            d = df[(df["group"] == group) & (df["horizon"] == h)].dropna(subset=["log_chl_target"])
            if len(d) < 60:
                continue
            d = d.sort_values("fecha_t0")
            cut = int(len(d) * 0.7)
            tr, te = d.iloc[:cut], d.iloc[cut:]
            m = _model().fit(tr[feats], tr["log_chl_target"])
            imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
            print(f"\n=== {group} | h+{h}d | n={len(d)} | top features (ganancia) ===")
            for f, v in imp.head(8).items():
                print(f"    {f:24s} {v:.3f}")
            for f, v in imp.items():
                recs.append({"group": group, "horizon": h, "feature": f, "importance": v})
    pd.DataFrame(recs).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")

    # agregado: peso por FAMILIA de features
    fam = pd.DataFrame(recs)
    def family(f):
        if f.startswith("chl") or f.startswith("log_chl"):
            return "autorregresivo"
        if any(f.startswith(p) for p in ("temp", "solar", "precip", "wind", "surface")):
            return "ERA5"
        return "espectral_S2"
    fam["familia"] = fam["feature"].map(family)
    print("\n=== peso medio por familia de features y horizonte ===")
    piv = fam.groupby(["horizon", "familia"])["importance"].sum().unstack(fill_value=0)
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
