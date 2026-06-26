"""
harmonize_landsat.py — Corrige el OFFSET cross-sensor de Landsat hacia la escala de Sentinel-2.

Problema: las reflectancias Landsat (C2 L2 SR) salen sistematicamente mas bajas que las de
Sentinel-2 (p.ej. en Cajon B2 ~0.024 vs 0.084). Mezclar ambos sensores mete un "salto de sensor"
en las features espectrales. Solucion: por cada cuerpo con ambos sensores, emparejar escenas
S2 y Landsat de fechas CERCANAS (<= TOL dias) y ajustar una regresion lineal por banda
S2 ~ a*Landsat + b; aplicar esa correccion a las filas Landsat -> quedan en la escala de S2.

El red-edge (B5/NDCI/CI_red/FAI) sigue NaN en Landsat (no existe la banda). Solo se corrigen las
bandas comunes: B2, B3, B4, B8 y turbidity.

Entrada : artifacts/state_series/scene_state.csv  (con columna 'sensor')
Salida  : artifacts/state_series/scene_state_harmonized.csv  (match_pairs lo prefiere si existe)
Se valida aguas abajo con evaluate_nested (si mejora el skill de Cajon, se queda).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C

SCENE = os.path.join(C.DIR_STATE, "scene_state.csv")
OUT = os.path.join(C.DIR_STATE, "scene_state_harmonized.csv")
BANDS = ["B2", "B3", "B4", "B8", "turbidity"]      # comunes (Landsat no tiene red-edge)
TOL_DAYS = 3
MIN_MATCH = 8


def main():
    sc = pd.read_csv(SCENE, parse_dates=["fecha"])
    if "sensor" not in sc.columns:
        print("scene_state sin columna 'sensor' (no hay Landsat). Nada que harmonizar."); return
    out = sc.copy()
    any_fix = False
    for wb, g in sc.groupby("water_body"):
        s2 = g[g.sensor == "sentinel2"].sort_values("fecha")
        ls = g[g.sensor == "landsat"].sort_values("fecha")
        if len(ls) == 0 or len(s2) == 0:
            continue
        # emparejar Landsat con la escena S2 mas cercana (<= TOL dias)
        m = pd.merge_asof(ls[["fecha"] + BANDS], s2[["fecha"] + BANDS],
                          on="fecha", direction="nearest",
                          tolerance=pd.Timedelta(days=TOL_DAYS), suffixes=("_ls", "_s2")).dropna()
        print(f"\n{wb}: {len(ls)} Landsat, {len(s2)} S2 -> {len(m)} matchups (<= {TOL_DAYS} d)")
        if len(m) < MIN_MATCH:
            print(f"  matchups insuficientes (<{MIN_MATCH}) -> Landsat sin corregir (se mantiene crudo)")
            continue
        # Emparejado de MOMENTOS (media+desv): LS->S2 = a*LS + b con a=std_S2/std_LS,
        # b=mean_S2 - a*mean_LS. Alinea escala/offset sin exigir correlacion pareada (la regresion
        # OLS daba R2~0). Conserva el orden relativo de las escenas Landsat.
        mask = out.water_body.eq(wb) & out.sensor.eq("landsat")
        for b in BANDS:
            x = m[f"{b}_ls"].values; y = m[f"{b}_s2"].values
            sx = x.std()
            if sx <= 1e-9:
                continue
            a = y.std() / sx; bb = y.mean() - a * x.mean()
            out.loc[mask, b] = a * out.loc[mask, b].values + bb
            print(f"  {b:10s}: a={a:+.3f} b={bb:+.4f} | LS media {x.mean():.3f}/sd {x.std():.3f}"
                  f" -> S2 media {y.mean():.3f}/sd {y.std():.3f}")
        any_fix = True
    if not any_fix:
        print("\nSin cuerpos harmonizables. No se escribe salida."); return
    out.to_csv(OUT, index=False)
    print(f"\nEstado harmonizado -> {OUT}")
    print("match_pairs lo usara automaticamente. Re-corre: match_pairs.py && evaluate_nested.py")


if __name__ == "__main__":
    main()
