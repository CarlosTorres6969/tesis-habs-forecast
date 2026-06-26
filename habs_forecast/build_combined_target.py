"""
build_combined_target.py — Target OPTIMO por cuerpo (densifica la costa con OLCI).

Decision por validacion contra in-situ:
  - okeechobee : VIIRS corregido (bias-correct a in-situ; OLCI fallo en lago somero, corr=0)
  - yojoa, cajon : VIIRS (sin in-situ; OLCI no fiable en lagos por analogia con Okeechobee)
  - tampa_bay, fonseca : OLCI (COSTA: OLCI concuerda con VIIRS y es ~2x mas denso ->
                          mas pares marinos -> mas potencia estadistica y mas datos para la red)

Salida: artifacts/targets/combined_target.csv  (water_body, fecha, chl_ugl, group)
"""
from __future__ import annotations
import os
import pandas as pd
import config as C

T = os.path.join(C.DIR_OUT, "targets")
CORR = os.path.join(T, "satellite_chl_corrected.csv")     # 5 cuerpos (okeechobee corregido)
OLCI = os.path.join(T, "olci_chl_daily.csv")
OUT = os.path.join(T, "combined_target.csv")

OLCI_BODIES = ["tampa_bay", "fonseca"]                     # costa: usar OLCI


def build():
    base = pd.read_csv(CORR)[["water_body", "fecha", "chl_ugl", "group"]].copy()
    base["fecha"] = pd.to_datetime(base["fecha"], utc=True).dt.tz_localize(None).dt.normalize()

    olci = pd.read_csv(OLCI)[["water_body", "fecha", "chl_ugl", "group"]].copy()
    olci["fecha"] = pd.to_datetime(olci["fecha"], utc=True).dt.tz_localize(None).dt.normalize()

    keep = base[~base["water_body"].isin(OLCI_BODIES)]     # okeechobee, yojoa, cajon (VIIRS)
    add = olci[olci["water_body"].isin(OLCI_BODIES)]       # tampa, fonseca (OLCI)
    out = pd.concat([keep, add], ignore_index=True).sort_values(["water_body", "fecha"])
    out.to_csv(OUT, index=False)

    print(f"Target combinado -> {OUT}")
    print(out.groupby(["group", "water_body"]).agg(
        dias=("fecha", "size"), fuente=("chl_ugl", lambda s: "")).drop(columns="fuente").to_string())
    src = {"okeechobee": "VIIRS-corregido", "yojoa": "VIIRS", "cajon": "VIIRS",
           "tampa_bay": "OLCI", "fonseca": "OLCI"}
    print("\nfuente por cuerpo:", src)


if __name__ == "__main__":
    build()
