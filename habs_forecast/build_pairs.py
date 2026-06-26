"""
build_pairs.py — Reconstruye PARES CAUSALES limpios para pronostico 0-7 d.

Toma datasets/pares_predictivos.csv (estructura t -> t+gap ya existente) y:
  1. Separa PREDICTORES (estado en t, sufijo _prev) del TARGET (estado en t+gap).
  2. ELIMINA fugas:
       - delta_*  : se calculan como (actual - prev) => usan el futuro. LEAKAGE.
       - chl_actual : es el target de regresion, no puede ser feature.
       - NDVI/NDWI como predictor (solo mascara/QA, ver config).
  3. Asigna cuerpo de agua (water_body) y grupo ecologico (freshwater/marine)
     por bounding-box, para validacion Leave-One-Water-Body-Out por tipo.
  4. Asigna un horizonte nominal (0,1,3,5,7) al gap real via HORIZON_TOLERANCE.
  5. Guarda artifacts/pairs/pairs_clean.csv e imprime el chequeo de realidad
     (cuantos pares utiles por grupo y horizonte).

NO entrena nada. Solo deja los datos en forma honesta.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import config as C

SRC = os.path.join(C.DIR_DATASETS, "pares_predictivos.csv")
OUT = os.path.join(C.DIR_PAIRS, "pairs_clean.csv")

# --------------------------------------------------------------------------------------
# Cuerpos de agua derivados de los clusters reales del dataset (bounding boxes).
#   freshwater: yojoa (HND), okeechobee (USA)
#   marine/estuarino: pensacola_est, ne_florida_est (USA) -- usados como costa vs costa
#   (Cajon/Fonseca/Tampa tienen muy pocos pares aqui; entran si caen en sus cajas.)
# box = (lat_min, lat_max, lon_min, lon_max)
# --------------------------------------------------------------------------------------
WATER_BODIES = [
    ("yojoa",          "freshwater", (14.0, 16.0, -88.4, -86.0)),
    ("okeechobee",     "freshwater", (26.2, 27.5, -81.3, -80.5)),
    ("pensacola_est",  "marine",     (30.0, 31.3, -87.8, -86.4)),
    ("ne_florida_est", "marine",     (29.5, 31.3, -82.6, -80.9)),
]


def assign_water_body(lat: float, lon: float):
    for name, group, (la0, la1, lo0, lo1) in WATER_BODIES:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name, group
    return "other", "other"


def assign_horizon(gap: float):
    """Devuelve el horizonte nominal cuyo rango de tolerancia contiene gap, o None."""
    for h, (lo, hi) in C.HORIZON_TOLERANCE.items():
        if lo <= gap <= hi:
            return h
    return None


def build():
    df = pd.read_csv(SRC)
    n0 = len(df)

    # --- coordenadas ---
    locs = df["loc"].str.split(",", expand=True).astype(float)
    df["lat"], df["lon"] = locs[0], locs[1]
    wb = df.apply(lambda r: assign_water_body(r["lat"], r["lon"]), axis=1)
    df["water_body"] = [w[0] for w in wb]
    df["group"] = [w[1] for w in wb]

    # --- horizonte nominal a partir del gap real ---
    df["horizon"] = df["gap_dias"].apply(assign_horizon)

    # --- features predictoras: estado en t (sufijo _prev), SIN fugas ---
    leak_cols = [c for c in df.columns if c.startswith("delta_")]
    leak_cols += ["chl_actual"]  # target de regresion
    prev_cols = [c for c in df.columns if c.endswith("_prev")]
    # quitar NDVI/NDWI _prev como predictor (solo QA); chl_anterior SI es predictor valido
    drop_prev = [c for c in prev_cols if c.startswith(("NDVI", "NDWI"))]
    feat_cols = [c for c in prev_cols if c not in drop_prev]

    keep = (["water_body", "group", "horizon", "gap_dias", "lat", "lon",
             "fecha_anterior", "fecha_actual"]
            + feat_cols + ["chl_actual", "hab_target"])
    clean = df[keep].copy()

    # target de regresion en log (chl es lognormal)
    clean["log_chl_target"] = np.log1p(clean["chl_actual"].clip(lower=0))
    clean = clean.rename(columns={"chl_actual": "chl_target"})

    os.makedirs(C.DIR_PAIRS, exist_ok=True)
    clean.to_csv(OUT, index=False)

    # ----------------------------- chequeo de realidad -----------------------------
    print(f"Fuente: {SRC}")
    print(f"Pares totales: {n0} -> limpios: {len(clean)}")
    print(f"Columnas de fuga eliminadas: {len(leak_cols)} delta_/chl_actual + "
          f"{len(drop_prev)} NDVI/NDWI_prev")
    print(f"Predictores conservados ({len(feat_cols)}): {feat_cols}")
    print(f"Salida: {OUT}\n")

    print("=== Pares por grupo y horizonte (solo gap dentro de tolerancia 0-8 d) ===")
    short = clean[clean["horizon"].notna()]
    if len(short):
        tab = short.pivot_table(index="group", columns="horizon",
                                values="hab_target", aggfunc="count", fill_value=0)
        print(tab.to_string())
        print()
        print("=== HAB+ (%) por grupo y horizonte ===")
        tab2 = short.pivot_table(index="group", columns="horizon",
                                 values="hab_target", aggfunc="mean").round(2)
        print(tab2.to_string())
    else:
        print("  (ningun par cae en 0-8 dias -> confirma el muro de factibilidad)")

    print("\n=== Cuerpos de agua (todos los gaps) ===")
    print(clean.groupby(["group", "water_body"]).size().to_string())
    return clean


if __name__ == "__main__":
    build()
