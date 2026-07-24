"""
verify_forecasts.py — VERIFICACION OPERATIVA POSTERIOR de los pronosticos ya emitidos.

Recorre la bitacora (artifacts/forecasts/forecast_log.csv) y, para cada pronostico cuya
fecha objetivo (t0 + horizonte) YA tiene target real disponible en combined_target.csv,
calcula el desempeno REALIZADO:
  - error  = chl_pred - chl_real  (y |error|),
  - in_band: si el valor real cayo dentro de la banda P10-P90 emitida,
  - alert_hit: si la bandera de RIESGO acerto el evento real (chl_real >= umbral del cuerpo).

Escribe artifacts/reports/forecast_verification.csv (detalle) y un resumen por
(grupo, horizonte): MAE, cobertura empirica de la banda y desempeno de la ALERTA con las
metricas estandar de pronostico de eventos (POD, FAR, precision, F1; mas honestas que la
exactitud cuando el evento es raro).

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
    Solo evalua pronosticos MADURADOS (con target real disponible).
    La bitacora se acumula por apend: un mismo pronostico (cuerpo, horizonte, t0) puede repetirse
    en varias corridas (run_ts). Se DEDUPLICA quedandose con la corrida MAS RECIENTE, para no
    contar el mismo pronostico varias veces (inflaria n y sesgaria MAE/POD/FAR) y para que la
    verificacion sea idempotente frente a re-ejecuciones o backfill."""
    log_df = log_df.copy()
    if "evaluation_mode" not in log_df.columns:
        log_df["evaluation_mode"] = "legacy_unknown"
    # Un backfill con el modelo final vio etiquetas posteriores a su t0; las filas
    # heredadas sin procedencia tampoco pueden presumirse prospectivas/OOS.
    log_df = log_df[log_df["evaluation_mode"].fillna("legacy_unknown") == "operational"]
    log_df["t0"] = pd.to_datetime(log_df["t0"]).dt.normalize()
    if "run_ts" in log_df.columns:
        log_df = (log_df.sort_values("run_ts")
                  .drop_duplicates(["water_body", "horizon", "t0"], keep="last"))
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
        stored_threshold = r.get("event_threshold_ugl")
        thr = (float(stored_threshold) if pd.notna(stored_threshold)
               else float(thr_body.get(wb, C.THRESHOLDS["moderate"])))
        p10, p90 = r.get("p10"), r.get("p90")
        in_band = (pd.notna(p10) and pd.notna(p90) and float(p10) <= chl_real <= float(p90))
        event_real = bool(chl_real >= thr)
        # Se evalua exactamente la alerta que fue emitida, con el umbral de evento
        # versionado en esa misma fila. No se reescribe retrospectivamente la politica.
        stored_alert = r.get("alerta_anomalia")
        if pd.isna(stored_alert):
            stored_alert = r.get("riesgo", False)
        alerta_pred = bool(stored_alert) if pd.notna(stored_alert) else False
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
               .apply(_group_metrics, include_groups=False)
               .reset_index())
    return detail, summary


def _group_metrics(g):
    """Metricas de desempeno por (grupo, horizonte). Ademas del error de intensidad y la
    cobertura de la banda, desglosa la ALERTA con las metricas ESTANDAR de pronostico de
    eventos (mas honestas que la exactitud cuando el evento es raro):
      - POD (probability of detection = recall = TP/(TP+FN)): que fraccion de eventos reales
        se alerto. NaN si no hubo eventos en la ventana.
      - FAR (false alarm ratio = FP/(TP+FP)): que fraccion de las alertas fue falsa. NaN si no
        se emitio ninguna alerta.
      - precision (TP/(TP+FP)) y F1 (media armonica de precision y POD).
    Se conserva hit_rate_alerta (exactitud: aciertos incl. no-eventos) para continuidad, pero
    con eventos raros infla el numero; POD/FAR/F1 son la lectura defendible."""
    pred = g["riesgo_pred"].astype(bool)
    real = g["event_real"].astype(bool)
    tp = int((pred & real).sum())
    fp = int((pred & ~real).sum())
    fn = int((~pred & real).sum())
    pod = tp / (tp + fn) if (tp + fn) else float("nan")          # recall / deteccion
    far = fp / (tp + fp) if (tp + fp) else float("nan")          # razon de falsas alarmas
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * pod / (prec + pod)
          if (tp + fp) and (tp + fn) and (prec + pod) > 0 else float("nan"))
    return pd.Series({
        "n": int(len(g)),
        "MAE": float(g["abs_error"].mean()),
        "cobertura_banda": float(g["in_band"].mean()),
        "eventos_reales": int(real.sum()),
        "alertas_emitidas": int(pred.sum()),
        "POD": pod,
        "FAR": far,
        "precision": prec,
        "F1": f1,
        "hit_rate_alerta": float(g["alert_hit"].mean()),
    })


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
        n_operational = (int(log_df["evaluation_mode"].eq("operational").sum())
                         if "evaluation_mode" in log_df else 0)
        if n_operational == 0:
            print(f"Bitacora: {n_log} filas, pero ninguna tiene procedencia operacional "
                  "verificable (las legacy/backfill se excluyen).")
        else:
            print(f"Bitacora: {n_log} pronosticos ({n_operational} operacionales). "
                  "Ninguno ha madurado aun con target real t0+h.")
        return
    detail.to_csv(OUT_DETAIL, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Bitacora: {n_log} pronosticos | verificados (madurados): {len(detail)}\n")
    print("=== DESEMPENO REALIZADO por (grupo, horizonte) ===")
    print(summary.to_string(index=False))
    print(f"\nDetalle -> {OUT_DETAIL}")
    print(f"Resumen -> {OUT_SUMMARY}")
    print("\nNota: MAE en ug/L; cobertura_banda objetivo ~0.80 (CQR). Alerta (eventos raros): "
          "POD = fraccion de eventos detectados (recall), FAR = fraccion de alertas falsas, "
          "F1 = balance precision/POD. hit_rate_alerta (exactitud) se conserva pero infla con "
          "eventos raros -> leer POD/FAR/F1.")


if __name__ == "__main__":
    main()
