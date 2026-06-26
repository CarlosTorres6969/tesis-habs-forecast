"""
select_features_per_horizon.py — Selecciona el SET DE FEATURES por horizonte via ablacion.

Motivo: el contexto in-situ ayuda a +7d pero estorba a +1d. Un set unico es un mal
compromiso. Aqui cada horizonte prueba combinaciones de FAMILIAS de features y se queda
con la que maximiza el skill OOS (ventana expansiva por cuerpo, vs persistencia).

Familias:
  AUTOREG  : clorofila reciente (backbone, siempre incluido)
  ERA5     : meteorologia
  SPECTRAL : bandas/indices Sentinel-2
  INSITU   : nutrientes + calidad de agua (fosforo, temp agua, OD, pH, turbidez, ...)

Salida: mejor combinacion por (grupo, horizonte) + tabla de skill. Define FEATURE_SETS
para usar en el entrenamiento final.
"""
from __future__ import annotations
import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import config as C
from train import SPECTRAL, AUTOREG, ERA5, NUTRIENTS, WATERQUAL, _model, PAIRS
from evaluate_robust import N_FOLDS, MIN_TRAIN_FRAC

OUT = os.path.join(C.DIR_REPORTS, "feature_sets.json")
FAMILIES = {"ERA5": ERA5, "SPECTRAL": SPECTRAL, "INSITU": NUTRIENTS + WATERQUAL}
OPTIONAL = ["ERA5", "SPECTRAL", "INSITU"]


def oos_skill(d, feats):
    ys, yh, yp = [], [], []
    for _, g in d.groupby("water_body"):
        g = g.sort_values("fecha_t0").reset_index(drop=True)
        N = len(g); start = int(N * MIN_TRAIN_FRAC)
        if N - start < N_FOLDS * 4:
            continue
        fold = (N - start) // N_FOLDS
        for k in range(N_FOLDS):
            a = start + k * fold
            b = N if k == N_FOLDS - 1 else a + fold
            tr, te = g.iloc[:a], g.iloc[a:b]
            if len(te) < 3 or len(tr) < 20:
                continue
            m = _model().fit(tr[feats], tr["log_chl_target"])
            ys.append(te["log_chl_target"].values); yh.append(m.predict(te[feats]))
            yp.append(te["log_chl_t0"].values)
    if not ys:
        return np.nan
    y, h, p = np.concatenate(ys), np.concatenate(yh), np.concatenate(yp)
    rp = np.sqrt(mean_squared_error(y, p))
    return 1 - np.sqrt(mean_squared_error(y, h)) / rp if rp > 0 else np.nan


def subsets():
    for r in range(len(OPTIONAL) + 1):
        for c in itertools.combinations(OPTIONAL, r):
            yield list(c)


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    avail = set(df.columns)
    best = {}
    for group in ("freshwater", "marine"):
        print(f"\n############  {group}  —  ablacion por horizonte  ############")
        best[group] = {}
        for h in [1, 3, 5, 7]:
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            scored = []
            for combo in subsets():
                feats = list(AUTOREG)
                for fam in combo:
                    feats += FAMILIES[fam]
                feats = [f for f in feats if f in avail]
                sk = oos_skill(d, feats)
                scored.append((sk, combo))
            scored.sort(key=lambda x: -(x[0] if np.isfinite(x[0]) else -9))
            bsk, bcombo = scored[0]
            best[group][h] = {"families": ["AUTOREG"] + bcombo, "skill": float(bsk)}
            base = next(s for s, c in scored if c == [])      # solo AUTOREG
            allf = next(s for s, c in scored if set(c) == set(OPTIONAL))
            print(f"  +{h}d | mejor: AUTOREG+{'+'.join(bcombo) or '(solo)':20s} "
                  f"skill={bsk:+.3f} | (solo AUTOREG={base:+.3f}, todo={allf:+.3f})")
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
