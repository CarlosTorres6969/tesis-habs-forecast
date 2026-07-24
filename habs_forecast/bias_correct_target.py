"""
bias_correct_target.py — Corrige la ESCALA del target satelital (VIIRS) anclandolo al
in-situ por QUANTILE MAPPING (CDF matching), donde exista verdad de campo.

Problema: VIIRS subestima la clorofila en lagos someros (Okeechobee mediana satelital 1.9
vs in-situ real 13.2). El quantile mapping reescala la serie satelital para que su
distribucion coincida con la in-situ, preservando el orden temporal (monotono):
    corregido = interp(valor_sat, percentiles_sat, percentiles_insitu)

Alcance: solo cuerpos con in-situ matcheado (Okeechobee). El resto pasa sin corregir
(se marca corrected=0). Para los corregidos, los umbrales OMS (10/24 ug/L) ya son fisicos.

La calibracion queda congelada al final de 2023. Ninguna observacion posterior
(incluido el TEST final) interviene en los percentiles del mapeo.

Salida: artifacts/targets/satellite_chl_corrected.csv + metadata de trazabilidad.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import config as C

TGT = os.path.join(C.DIR_OUT, "targets", "satellite_chl_daily.csv")
INS = os.path.join(C.DIR_OUT, "targets", "insitu_chl.csv")
OUT = os.path.join(C.DIR_OUT, "targets", "satellite_chl_corrected.csv")
META = os.path.join(C.DIR_OUT, "targets", "satellite_chl_correction_meta.json")
MIN_INSITU = 30          # minimo de puntos in-situ para calibrar un cuerpo
PCTS = np.arange(1, 100)  # percentiles para construir el mapeo


def build():
    sat = pd.read_csv(TGT)
    ins = pd.read_csv(INS)
    sat["fecha"] = pd.to_datetime(sat["fecha"], utc=True, errors="coerce").dt.tz_localize(None)
    ins["fecha"] = pd.to_datetime(ins["fecha"], utc=True, errors="coerce").dt.tz_localize(None)
    calibration_end = pd.Timestamp(C.TARGET_CALIBRATION_END)
    out = sat.copy()
    out["corrected"] = 0
    out["correction_calibration_end"] = C.TARGET_CALIBRATION_END

    info = []
    metadata = {
        "method": "quantile_mapping_fixed_preperiod",
        "calibration_end": C.TARGET_CALIBRATION_END,
        "bodies": {},
    }
    for wb in sat["water_body"].unique():
        sat_cal = sat[(sat.water_body == wb) & (sat.fecha <= calibration_end)]
        ins_cal = ins[(ins.water_body == wb) & (ins.fecha <= calibration_end)]
        s = sat_cal["chl_ugl"].dropna().values
        i = ins_cal["chl_ugl"].dropna().values
        if len(i) < MIN_INSITU or len(s) < MIN_INSITU:
            metadata["bodies"][wb] = {
                "corrected": False,
                "n_sat_calibration": int(len(s)),
                "n_insitu_calibration": int(len(i)),
            }
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
        metadata["bodies"][wb] = {
            "corrected": True,
            "n_sat_calibration": int(len(s)),
            "n_insitu_calibration": int(len(i)),
            "satellite_percentiles": sat_q.tolist(),
            "insitu_percentiles": ins_q.tolist(),
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    with open(META, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Target corregido -> {OUT}")
    print(f"Calibracion congelada hasta {C.TARGET_CALIBRATION_END} -> {META}")
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
