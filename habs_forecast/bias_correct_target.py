"""
bias_correct_target.py — Corrige la ESCALA del target satelital (VIIRS) anclandolo al
in-situ por QUANTILE MAPPING (CDF matching), donde exista verdad de campo.

Problema: VIIRS subestima la clorofila en lagos someros (Okeechobee mediana satelital 1.9
vs in-situ real 13.2). El quantile mapping reescala la serie satelital para que su
distribucion coincida con la in-situ, preservando el orden temporal (monotono):
    corregido = interp(valor_sat, percentiles_sat, percentiles_insitu)

Alcance: solo cuerpos con in-situ matcheado (Okeechobee). El resto pasa sin corregir
(se marca corrected=0). Para los corregidos, los umbrales OMS (10/24 ug/L) ya son fisicos.

Salida: artifacts/targets/satellite_chl_corrected.csv (mismo schema + columna corrected).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C

TGT = os.path.join(C.DIR_OUT, "targets", "satellite_chl_daily.csv")
INS = os.path.join(C.DIR_OUT, "targets", "insitu_chl.csv")
OUT = os.path.join(C.DIR_OUT, "targets", "satellite_chl_corrected.csv")
MIN_INSITU = 30          # minimo de puntos in-situ para calibrar un cuerpo
PCTS = np.arange(1, 100)  # percentiles para construir el mapeo


def build():
    sat = pd.read_csv(TGT)
    ins = pd.read_csv(INS)
    out = sat.copy()
    out["corrected"] = 0

    info = []
    for wb in sat["water_body"].unique():
        s = sat.loc[sat.water_body == wb, "chl_ugl"].dropna().values
        i = ins.loc[ins.water_body == wb, "chl_ugl"].dropna().values
        if len(i) < MIN_INSITU or len(s) < MIN_INSITU:
            continue
        sat_q = np.percentile(s, PCTS)
        ins_q = np.percentile(i, PCTS)
        # monotonizar sat_q (necesario para np.interp) por si hay percentiles repetidos
        sat_q = np.maximum.accumulate(sat_q)
        mask = sat.water_body == wb
        corr = np.interp(sat.loc[mask, "chl_ugl"].values, sat_q, ins_q)
        out.loc[mask, "chl_ugl"] = corr
        out.loc[mask, "corrected"] = 1
        info.append((wb, len(i), float(np.median(s)), float(np.median(i)), float(np.median(corr))))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Target corregido -> {OUT}")
    print(f"{'cuerpo':12s} {'n_insitu':>8} {'sat_med':>8} {'insitu_med':>10} {'corr_med':>9}")
    for wb, n, sm, im, cm in info:
        print(f"{wb:12s} {n:>8} {sm:>8.2f} {im:>10.2f} {cm:>9.2f}")
    if not info:
        print("Ningun cuerpo con in-situ suficiente para calibrar.")
    else:
        # tasa de evento OMS (>=10 ug/L) antes/despues en los corregidos
        for wb, *_ in info:
            o = out[out.water_body == wb]
            print(f"  {wb}: eventos >=10 ug/L (OMS) tras correccion = "
                  f"{(o['chl_ugl']>=10).mean()*100:.0f}% de dias")


if __name__ == "__main__":
    build()
