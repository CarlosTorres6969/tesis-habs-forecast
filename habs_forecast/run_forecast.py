"""
run_forecast.py — BUCLE de pronóstico OPERATIVO de alerta temprana de HABs (0-7 días).

Para CADA cuerpo (config.REGIONS) y CADA horizonte (1,3,5,7):
  - toma la ÚLTIMA escena disponible como t0 (causal: solo datos <= t0),
  - reusa predict.forecast_body (misma construcción de features y modelos que predict.py),
  - emite clorofila-a esperada + banda P10-P90 (CQR) + probabilidad y bandera de RIESGO
    (ensamble Red+XGBoost), con una etiqueta de CONFIANZA (guards.py: frescura/cobertura/estado).

Salidas (con timestamp del run):
  artifacts/forecasts/forecast_<YYYYMMDD_HHMMSS>.csv  y  .json   -> snapshot del run
  artifacts/forecasts/forecast_log.csv                          -> BITÁCORA acumulada
    (se apenda una fila por cuerpo-horizonte-run; base de verify_forecasts.py)

Robustez operativa: usa logging (no print suelto) y try/except POR cuerpo: si uno falla,
loguea el motivo y continúa con los demás. NO entrena ni modifica modelos.

Uso:  python run_forecast.py
"""
from __future__ import annotations
import os, sys, json, logging
import pandas as pd
import config as C
import guards
import build_model_cards
# NB: predict (forecast_body, _load, SCENE) se importa PEREZOSAMENTE dentro de run()/backfill():
# arrastra torch, y así run_forecast (y su núcleo puro build_rows) se importa sin torch -> testeable
# en CI con dependencias mínimas.

LOG = os.path.join(C.DIR_FORECASTS, "forecast_log.csv")
CARDS = os.path.join(C.DIR_MODELS, "model_cards.json")

# esquema ESTRUCTURADO de salida (orden de columnas estable, contrato del pronóstico)
SCHEMA = ["run_ts", "water_body", "group", "t0", "horizon", "chl_pred", "p10", "p90",
          "prob_riesgo", "event_threshold_ugl", "alerta_anomalia", "riesgo",
          "nivel", "floracion_magnitud", "confianza", "data_age_days", "n_water_px",
          "evaluation_mode", "modelo_meta"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_forecast")


def _load_cards():
    """Carga las model cards; si no existen, las genera (sin reentrenar)."""
    if os.path.exists(CARDS):
        return json.load(open(CARDS, encoding="utf-8"))
    try:
        return build_model_cards.build()
    except Exception as e:
        log.warning("No se pudieron generar model cards: %s", e)
        return {}


def build_rows(fc, run_ts, cards, run_ts_for_age=None, evaluation_mode="operational"):
    """Convierte el pronóstico estructurado de un cuerpo (forecast_body) en filas con el
    SCHEMA operativo, anotando confianza (guards) y metadata de modelo. Función pura y
    testeable (no toca disco): el test de esquema la alimenta con un fc sintético."""
    confianza, flags, age = guards.evaluate_guards(
        fc["water_body"], fc["t0"], fc["n_water_px"], run_ts_for_age or run_ts,
        feature_ages=fc.get("feature_ages"),
        missing_context=fc.get("missing_context"))
    rows = []
    for h in fc["horizons"]:
        card = cards.get(f"{fc['group']}_h{h['horizon']}", {})
        meta = {"commit_git": card.get("commit_git"),
                "fecha_entrenamiento": card.get("fecha_entrenamiento"),
                "n_pares": card.get("n_pares"),
                "skill_validado": card.get("skill_validado")}
        rows.append({
            "run_ts": run_ts,
            "water_body": fc["water_body"],
            "group": fc["group"],
            "t0": pd.Timestamp(fc["t0"]).date().isoformat(),
            "horizon": int(h["horizon"]),
            "chl_pred": round(float(h["chl_pred"]), 3),
            "p10": None if h["p10"] is None else round(float(h["p10"]), 3),
            "p90": None if h["p90"] is None else round(float(h["p90"]), 3),
            "prob_riesgo": round(float(h["prob_riesgo"]), 4),
            "event_threshold_ugl": float(h.get("event_threshold_ugl", fc.get("thr_body", 10.0))),
            "alerta_anomalia": bool(h.get("alerta_anomalia", h["riesgo"])),
            "riesgo": bool(h.get("alerta_anomalia", h["riesgo"])),
            "nivel": h.get("nivel"),
            "floracion_magnitud": bool(h.get("floracion_magnitud", False)),
            "confianza": confianza,
            "data_age_days": age,
            "n_water_px": fc["n_water_px"],
            "evaluation_mode": evaluation_mode,
            "modelo_meta": json.dumps(meta, ensure_ascii=False),
        })
    return rows


def _append_log(df):
    """Apend compatible con cambios de esquema; reescribe atomicamente el CSV acumulado."""
    if os.path.exists(LOG):
        previous = pd.read_csv(LOG)
        if "evaluation_mode" not in previous:
            previous["evaluation_mode"] = "legacy_unknown"
        else:
            previous["evaluation_mode"] = previous["evaluation_mode"].fillna("legacy_unknown")
        combined = pd.concat([previous, df], ignore_index=True, sort=False)
    else:
        combined = df.copy()
    for column in SCHEMA:
        if column not in combined:
            combined[column] = None
    extra = [column for column in combined.columns if column not in SCHEMA]
    tmp = LOG + ".tmp"
    combined[SCHEMA + extra].to_csv(tmp, index=False)
    os.replace(tmp, LOG)


def run(run_ts=None):
    """Ejecuta el bucle operativo sobre todos los cuerpos y devuelve el DataFrame del run."""
    from predict import forecast_body
    run_dt = pd.Timestamp.now() if run_ts is None else pd.Timestamp(run_ts)
    run_iso = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    stamp = run_dt.strftime("%Y%m%d_%H%M%S")
    cards = _load_cards()
    bodies = [m["key"] for m in C.REGIONS.values()]
    log.info("Pronóstico operativo: %d cuerpos x horizontes [1,3,5,7] | run=%s",
             len(bodies), run_iso)

    rows = []
    for wb in bodies:
        try:
            fc = forecast_body(wb)                       # última escena = t0
            if fc is None:
                log.warning("%s: sin escenas/datos suficientes -> se omite", wb); continue
            scene_age = guards.data_age_days(fc["t0"], run_dt)
            if scene_age > C.MAX_DATA_AGE_DAYS:
                log.warning("%s: escena de %d dias (max=%d) -> no se emite como operacional",
                            wb, scene_age, C.MAX_DATA_AGE_DAYS)
                continue
            br = build_rows(fc, run_iso, cards, run_ts_for_age=run_dt)
            rows.extend(br)
            conf = br[0]["confianza"] if br else "?"
            n_alert = sum(r["riesgo"] for r in br)
            log.info("%-12s t0=%s confianza=%-12s riesgo en %d/%d horizontes",
                     wb, br[0]["t0"] if br else "?", conf, n_alert, len(br))
        except Exception as e:                            # un cuerpo no debe tumbar el run
            log.exception("%s: fallo el pronóstico (%s) -> continuo con los demás", wb, e)

    if not rows:
        log.error("Ningún pronóstico generado."); return None
    df = pd.DataFrame(rows, columns=SCHEMA)

    os.makedirs(C.DIR_FORECASTS, exist_ok=True)
    snap_csv = os.path.join(C.DIR_FORECASTS, f"forecast_{stamp}.csv")
    snap_json = os.path.join(C.DIR_FORECASTS, f"forecast_{stamp}.json")
    df.to_csv(snap_csv, index=False)
    df.to_json(snap_json, orient="records", indent=2, force_ascii=False)
    # bitacora acumulada (apend; crea cabecera solo la primera vez)
    _append_log(df)

    log.info("Snapshot -> %s", snap_csv)
    log.info("Bitácora (apend) -> %s", LOG)
    print("\n=== RESUMEN DEL RUN ===")
    print(df[["water_body", "horizon", "chl_pred", "p10", "p90",
              "prob_riesgo", "riesgo", "confianza"]].to_string(index=False))
    return df


def backfill(per_body=12):
    """Siembra la bitácora con pronósticos HISTÓRICOS (escenas pasadas ya madurables) para
    demostración retrospectiva. El modelo final pudo entrenarse con datos posteriores a esos
    t0, por lo que estas filas se marcan ``retrospective_in_sample`` y quedan excluidas de la
    verificación operativa/OOS. No es el modo por defecto."""
    from predict import forecast_body, _load, SCENE
    run_iso = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    cards = _load_cards()
    bodies = [m["key"] for m in C.REGIONS.values()]
    maxh = max(h for h in C.HORIZONS if h != 0)
    # máximo del target por cuerpo: un t0 solo madura si t0+maxh <= último target disponible
    tgt = pd.read_csv(os.path.join(C.DIR_OUT, "targets", "combined_target.csv"))
    tgt["fecha"] = pd.to_datetime(tgt["fecha"], utc=True, errors="coerce").dt.tz_localize(None)
    tmax = tgt.groupby("water_body")["fecha"].max().to_dict()
    rows = []
    for wb in bodies:
        try:
            sc = _load(SCENE, wb).sort_values("fecha")
            if sc.empty or wb not in tmax:
                continue
            # escenas cuyo t0+maxh cae dentro del rango con target disponible (por cuerpo)
            limite = min(sc["fecha"].max(), tmax[wb]) - pd.Timedelta(days=maxh + 2)
            cand = sc[sc["fecha"] <= limite]["fecha"].tolist()[-per_body:]
            for t0 in cand:
                fc = forecast_body(wb, t0)
                if fc is None:
                    continue
                # antigüedad relativa al propio t0 (como si se hubiera corrido ese día)
                rows.extend(build_rows(
                    fc, run_iso, cards, run_ts_for_age=pd.Timestamp(t0),
                    evaluation_mode="retrospective_in_sample"))
            log.info("%-12s backfill: %d escenas históricas", wb, len(cand))
        except Exception as e:
            log.exception("%s: fallo backfill (%s)", wb, e)
    if not rows:
        log.error("Backfill sin filas."); return None
    df = pd.DataFrame(rows, columns=SCHEMA)
    os.makedirs(C.DIR_FORECASTS, exist_ok=True)
    df.to_csv(os.path.join(C.DIR_FORECASTS, f"forecast_backfill_{stamp}.csv"), index=False)
    _append_log(df)
    log.info("Backfill: %d pronósticos históricos apendados a %s", len(df), LOG)
    return df


if __name__ == "__main__":
    if "--backfill" in sys.argv:
        i = sys.argv.index("--backfill")
        k = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 12
        backfill(per_body=k)
    else:
        run()
