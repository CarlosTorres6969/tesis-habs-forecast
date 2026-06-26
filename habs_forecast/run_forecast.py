"""
run_forecast.py — BUCLE de pronostico OPERATIVO de alerta temprana de HABs (0-7 dias).

Para CADA cuerpo (config.REGIONS) y CADA horizonte (1,3,5,7):
  - toma la ULTIMA escena disponible como t0 (causal: solo datos <= t0),
  - reusa predict.forecast_body (misma construccion de features y modelos que predict.py),
  - emite clorofila-a esperada + banda P10-P90 (CQR) + probabilidad y bandera de RIESGO
    (ensamble Red+XGBoost), con una etiqueta de CONFIANZA (guards.py: frescura/cobertura/estado).

Salidas (con timestamp del run):
  artifacts/forecasts/forecast_<YYYYMMDD_HHMMSS>.csv  y  .json   -> snapshot del run
  artifacts/forecasts/forecast_log.csv                          -> BITACORA acumulada
    (se apenda una fila por cuerpo-horizonte-run; base de verify_forecasts.py)

Robustez operativa: usa logging (no print suelto) y try/except POR cuerpo: si uno falla,
loguea el motivo y continua con los demas. NO entrena ni modifica modelos.

Uso:  python run_forecast.py
"""
from __future__ import annotations
import os, sys, json, logging
import pandas as pd
import config as C
import guards
import build_model_cards
# NB: predict (forecast_body, _load, SCENE) se importa PEREZOSAMENTE dentro de run()/backfill():
# arrastra torch, y asi run_forecast (y su nucleo puro build_rows) se importa sin torch -> testeable
# en CI con dependencias minimas.

LOG = os.path.join(C.DIR_FORECASTS, "forecast_log.csv")
CARDS = os.path.join(C.DIR_MODELS, "model_cards.json")

# esquema ESTRUCTURADO de salida (orden de columnas estable, contrato del pronostico)
SCHEMA = ["run_ts", "water_body", "group", "t0", "horizon", "chl_pred", "p10", "p90",
          "prob_riesgo", "riesgo", "confianza", "data_age_days", "n_water_px", "modelo_meta"]

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


def build_rows(fc, run_ts, cards, run_ts_for_age=None):
    """Convierte el pronostico estructurado de un cuerpo (forecast_body) en filas con el
    SCHEMA operativo, anotando confianza (guards) y metadata de modelo. Funcion pura y
    testeable (no toca disco): el test de esquema la alimenta con un fc sintetico."""
    confianza, flags, age = guards.evaluate_guards(
        fc["water_body"], fc["t0"], fc["n_water_px"], run_ts_for_age or run_ts)
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
            "riesgo": bool(h["riesgo"]),
            "confianza": confianza,
            "data_age_days": age,
            "n_water_px": fc["n_water_px"],
            "modelo_meta": json.dumps(meta, ensure_ascii=False),
        })
    return rows


def run(run_ts=None):
    """Ejecuta el bucle operativo sobre todos los cuerpos y devuelve el DataFrame del run."""
    from predict import forecast_body
    run_dt = pd.Timestamp.now() if run_ts is None else pd.Timestamp(run_ts)
    run_iso = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    stamp = run_dt.strftime("%Y%m%d_%H%M%S")
    cards = _load_cards()
    bodies = [m["key"] for m in C.REGIONS.values()]
    log.info("Pronostico operativo: %d cuerpos x horizontes [1,3,5,7] | run=%s",
             len(bodies), run_iso)

    rows = []
    for wb in bodies:
        try:
            fc = forecast_body(wb)                       # ultima escena = t0
            if fc is None:
                log.warning("%s: sin escenas/datos suficientes -> se omite", wb); continue
            br = build_rows(fc, run_iso, cards, run_ts_for_age=run_dt)
            rows.extend(br)
            conf = br[0]["confianza"] if br else "?"
            n_alert = sum(r["riesgo"] for r in br)
            log.info("%-12s t0=%s confianza=%-12s riesgo en %d/%d horizontes",
                     wb, br[0]["t0"] if br else "?", conf, n_alert, len(br))
        except Exception as e:                            # un cuerpo no debe tumbar el run
            log.exception("%s: fallo el pronostico (%s) -> continuo con los demas", wb, e)

    if not rows:
        log.error("Ningun pronostico generado."); return None
    df = pd.DataFrame(rows, columns=SCHEMA)

    os.makedirs(C.DIR_FORECASTS, exist_ok=True)
    snap_csv = os.path.join(C.DIR_FORECASTS, f"forecast_{stamp}.csv")
    snap_json = os.path.join(C.DIR_FORECASTS, f"forecast_{stamp}.json")
    df.to_csv(snap_csv, index=False)
    df.to_json(snap_json, orient="records", indent=2, force_ascii=False)
    # bitacora acumulada (apend; crea cabecera solo la primera vez)
    df.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)

    log.info("Snapshot -> %s", snap_csv)
    log.info("Bitacora (apend) -> %s", LOG)
    print("\n=== RESUMEN DEL RUN ===")
    print(df[["water_body", "horizon", "chl_pred", "p10", "p90",
              "prob_riesgo", "riesgo", "confianza"]].to_string(index=False))
    return df


def backfill(per_body=12):
    """Siembra la bitacora con pronosticos HISTORICOS (escenas pasadas ya madurables) para
    arrancar la verificacion operativa con datos reales. Toma, por cuerpo, las ultimas
    `per_body` escenas cuyo target t0+h ya existe, y emite el pronostico en cada t0.
    Causal intacto: forecast_body(wb, t0) solo usa datos <= t0. No es el modo por defecto."""
    from predict import forecast_body, _load, SCENE
    run_iso = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    cards = _load_cards()
    bodies = [m["key"] for m in C.REGIONS.values()]
    maxh = max(h for h in C.HORIZONS if h != 0)
    # maximo del target por cuerpo: un t0 solo madura si t0+maxh <= ultimo target disponible
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
                # antiguedad relativa al propio t0 (como si se hubiera corrido ese dia)
                rows.extend(build_rows(fc, run_iso, cards, run_ts_for_age=pd.Timestamp(t0)))
            log.info("%-12s backfill: %d escenas historicas", wb, len(cand))
        except Exception as e:
            log.exception("%s: fallo backfill (%s)", wb, e)
    if not rows:
        log.error("Backfill sin filas."); return None
    df = pd.DataFrame(rows, columns=SCHEMA)
    os.makedirs(C.DIR_FORECASTS, exist_ok=True)
    df.to_csv(os.path.join(C.DIR_FORECASTS, f"forecast_backfill_{stamp}.csv"), index=False)
    df.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    log.info("Backfill: %d pronosticos historicos apendados a %s", len(df), LOG)
    return df


if __name__ == "__main__":
    if "--backfill" in sys.argv:
        i = sys.argv.index("--backfill")
        k = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 12
        backfill(per_body=k)
    else:
        run()
