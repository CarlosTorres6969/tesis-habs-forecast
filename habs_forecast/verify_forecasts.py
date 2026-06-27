"""
verify_forecasts.py — VERIFICACION OPERATIVA POSTERIOR de los pronosticos ya emitidos.

Recorre la bitacora (artifacts/forecasts/forecast_log.csv) y, para cada pronostico cuya
fecha objetivo (t0 + horizonte) YA tiene target real disponible en combined_target.csv,
calcula el desempeno REALIZADO:
  - error  = chl_pred - chl_real  (y |error|),
  - in_band: si el valor real cayo dentro de la banda P10-P90 emitida,
  - alert_hit: si la bandera de RIESGO acerto el evento real (chl_real >= umbral del cuerpo).

Escribe artifacts/reports/forecast_verification.csv (detalle) y un resumen por
(grupo, horizonte): MAE, cobertura empirica de la banda y hit-rate de la alerta.

NO entrena ni ajusta nada: es validacion operativa de lo ya pronosticado (cierra el lazo).
El nucleo (verify) es PURO y testeable: recibe DataFrames y devuelve (detalle, resumen).

Uso:  python verify_forecasts.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
import config as C

LOG = os.path.join(C.DIR_FORECASTS, "forecast_log.csv")
TARGET = os.path.join(C.DIR_OUT, "targets", "combined_target.csv")
OUT_DETAIL = os.path.join(C.DIR_REPORTS, "forecast_verification.csv")
OUT_SUMMARY = os.path.join(C.DIR_REPORTS, "forecast_verification_summary.csv")


def _match_real(target_wb, t0, h):
    """Busca el target real para (t0, horizonte h) dentro de la tolerancia del horizonte,
    eligiendo el mas cercano a t0+h. Devuelve (fecha_target, chl_real) o (None, None) si
    aun no hay dato (pronostico no madurado)."""
    lo, hi = C.HORIZON_TOLERANCE[h]
    win = target_wb[(target_wb["fecha"] >= t0 + pd.Timedelta(days=lo)) &
                    (target_wb["fecha"] <= t0 + pd.Timedelta(days=hi))]
    if win.empty:
        return None, None
    win = win.assign(dist=(win["fecha"] - (t0 + pd.Timedelta(days=h))).abs())
    best = win.sort_values("dist").iloc[0]
    return best["fecha"], float(best["chl_ugl"])


def verify(log_df, target_df, thr_body):
    """Nucleo PURO. Cruza pronosticos emitidos con el target real y devuelve (detalle, resumen).
      log_df    : filas de la bitacora (run_forecast SCHEMA).
      target_df : combined_target (water_body, fecha, chl_ugl).
      thr_body  : dict {cuerpo: umbral de alerta} para definir el evento real.
    Solo evalua pronosticos MADURADOS (con target real disponible)."""
    log_df = log_df.copy()
    log_df["t0"] = pd.to_datetime(log_df["t0"]).dt.normalize()
    target_df = target_df.copy()
    target_df["fecha"] = pd.to_datetime(target_df["fecha"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    tgt_by_body = {wb: g.sort_values("fecha") for wb, g in target_df.groupby("water_body")}

    rows = []
    for _, r in log_df.iterrows():
        wb, h = r["water_body"], int(r["horizon"])
        if wb not in tgt_by_body or h not in C.HORIZON_TOLERANCE:
            continue
        fecha_real, chl_real = _match_real(tgt_by_body[wb], r["t0"], h)
        if chl_real is None:
            continue                                  # aun no madura -> no verificable
        thr = C.alert_threshold_ugl(thr_body.get(wb, C.THRESHOLDS["moderate"]))  # mismo umbral que la alerta
        p10, p90 = r.get("p10"), r.get("p90")
        in_band = (pd.notna(p10) and pd.notna(p90) and float(p10) <= chl_real <= float(p90))
        event_real = bool(chl_real >= thr)
        alerta_pred = bool(r["riesgo"])
        rows.append({
            "run_ts": r.get("run_ts"), "water_body": wb, "group": r.get("group"),
            "t0": r["t0"].date().isoformat(), "horizon": h,
            "fecha_target": pd.Timestamp(fecha_real).date().isoformat(),
            "chl_pred": float(r["chl_pred"]), "chl_real": chl_real,
            "error": float(r["chl_pred"]) - chl_real,
            "abs_error": abs(float(r["chl_pred"]) - chl_real),
            "p10": None if pd.isna(p10) else float(p10),
            "p90": None if pd.isna(p90) else float(p90),
            "in_band": bool(in_band),
            "riesgo_pred": alerta_pred, "event_real": event_real,
            "alert_hit": bool(alerta_pred == event_real),
            "confianza": r.get("confianza"),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summary = (detail.groupby(["group", "horizon"])
               .agg(n=("abs_error", "size"),
                    MAE=("abs_error", "mean"),
                    cobertura_banda=("in_band", "mean"),
                    hit_rate_alerta=("alert_hit", "mean"),
                    eventos_reales=("event_real", "sum"))
               .reset_index())
    return detail, summary


def main():
    if not os.path.exists(LOG):
        print(f"Sin bitacora ({LOG}); corre run_forecast.py primero."); return
    log_df = pd.read_csv(LOG)
    if not os.path.exists(TARGET):
        print(f"Sin target ({TARGET})."); return
    target_df = pd.read_csv(TARGET)
    thr_body = joblib.load(os.path.join(C.DIR_MODELS, "thr_body.pkl")) \
        if os.path.exists(os.path.join(C.DIR_MODELS, "thr_body.pkl")) else {}

    detail, summary = verify(log_df, target_df, thr_body)
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    n_log = len(log_df)
    if detail.empty:
        print(f"Bitacora: {n_log} pronosticos. Ninguno MADURADO aun "
              f"(target real t0+h todavia no disponible). Vuelve a correr cuando haya datos.")
        return
    detail.to_csv(OUT_DETAIL, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Bitacora: {n_log} pronosticos | verificados (madurados): {len(detail)}\n")
    print("=== DESEMPENO REALIZADO por (grupo, horizonte) ===")
    print(summary.to_string(index=False))
    print(f"\nDetalle -> {OUT_DETAIL}")
    print(f"Resumen -> {OUT_SUMMARY}")
    print("\nNota: MAE en ug/L; cobertura_banda objetivo ~0.80 (CQR); "
          "hit_rate_alerta = acierto de la bandera de riesgo vs evento real.")


if __name__ == "__main__":
    main()
