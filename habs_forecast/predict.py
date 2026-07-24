"""
predict.py — PREDICTOR DESPLEGABLE. Dado un cuerpo de agua y una fecha t0 (por defecto la
última escena disponible), produce el pronóstico 0-7 días:
    - INTENSIDAD: clorofila-a esperada (ug/L) por horizonte = proxy de BIOMASA algal
    - RIESGO: probabilidad de biomasa algal elevada / clorofila-a anómala (ensamble Red+XGBoost)
      y decisión (sí/no). NB: clorofila-a alta indica más biomasa, NO confirma floración NOCIVA
      (toxicidad); la alerta señala condiciones de RIESGO que ameritan verificación de campo
      (identificación de cianobacterias, toxinas). Sentinel-2 no distingue cianobacterias.

Construye el vector de features en t0 desde los artefactos (igual que match_pairs, pero sin
el target futuro), carga los modelos de producción (train_final.py) y predice.

Uso:  python predict.py                 -> todos los cuerpos, última escena
      python predict.py okeechobee      -> un cuerpo
      python predict.py okeechobee 2025-08-01
"""
from __future__ import annotations
import os, sys, joblib
import numpy as np
import pandas as pd
import torch
import config as C
from train import SPECTRAL, AUTOREG, ERA5 as ERA5F
from train_nn import HABNet
from match_pairs import autoregressive_features
import guards

T = os.path.join(C.DIR_OUT, "targets")
_SCENE_RAW = os.path.join(C.DIR_STATE, "scene_state.csv")
_SCENE_HARM = os.path.join(C.DIR_STATE, "scene_state_harmonized.csv")
# Debe coincidir con match_pairs.SCENE_FILE para evitar train-serving skew en Landsat.
SCENE = _SCENE_HARM if os.path.exists(_SCENE_HARM) else _SCENE_RAW
TARGET = os.path.join(T, "combined_target.csv")
ERA5D = os.path.join(C.DIR_STATE, "era5_daily.csv")
NUTR = os.path.join(T, "nutrients_daily.csv")
WQ = os.path.join(T, "waterquality_daily.csv")
MODELS = C.DIR_MODELS

GROUP = {"okeechobee": "freshwater", "yojoa": "freshwater", "cajon": "freshwater",
         "tampa_bay": "marine", "fonseca": "marine"}
SPEC = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
ERA5_BASE = ["temp_air_2m", "solar_radiation", "precipitation", "wind_speed_10m", "surface_pressure"]
WQ_VARS = ["water_temp", "do_mgl", "ph", "turbidity_insitu", "spec_cond", "secchi", "ammonia"]


def _load(path, wb):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["water_body", "fecha"])
    d = pd.read_csv(path)
    if "fecha" in d.columns:
        d["fecha"] = pd.to_datetime(d["fecha"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    return d[d["water_body"] == wb] if "water_body" in d.columns else d


def _age_days(t0, date):
    return int((pd.Timestamp(t0).normalize() - pd.Timestamp(date).normalize()).days)


def build_features(wb, t0, include_meta=False):
    row = {}
    # --- espectral (escena S2 en/antes de t0) ---
    sc = _load(SCENE, wb)
    sc = sc[sc["fecha"] <= t0].sort_values("fecha")
    if sc.empty:
        return None
    s = sc.iloc[-1]
    for f in SPEC:
        row[f] = s[f]
    # --- autorregresivo (clorofila reciente del target) ---
    tg = _load(TARGET, wb).sort_values("fecha")
    past = tg[tg["fecha"] <= t0]
    if past.empty:
        return None
    ar = autoregressive_features(tg, t0)
    if ar is None:
        return None
    target_date = ar.pop("target_date_t0")
    chl0 = ar["chl_t0"]
    row.update(ar)
    ages = {"target": _age_days(t0, target_date)}
    missing_context = []
    # --- ERA5 (en/antes de t0 + media móvil 7d) ---
    er = _load(ERA5D, wb).sort_values("fecha")
    ep = er[er["fecha"] <= t0]
    if len(ep):
        last = ep.iloc[-1]
        era_age = _age_days(t0, last["fecha"])
        ages["era5"] = era_age
        if era_age <= C.MAX_ERA5_AGE_DAYS:
            last_date = pd.Timestamp(last["fecha"])
            w7 = ep[(ep["fecha"] > last_date - pd.Timedelta(days=7)) &
                    (ep["fecha"] <= last_date)]
            for v in ERA5_BASE:
                if v in ep.columns:
                    row[v] = last[v]
                    row[f"{v}_roll7"] = w7[v].mean()
        else:
            missing_context.append("ERA5")
    else:
        ages["era5"] = None
        missing_context.append("ERA5")
    # --- nutrientes + calidad de agua (contexto <= t0) ---
    nu = _load(NUTR, wb)
    ages["nutrients"] = None
    if "tp_mgl" in nu.columns and len(nu):
        p = nu[nu["fecha"] <= t0].sort_values("fecha")
        if len(p):
            ages["nutrients"] = _age_days(t0, p.iloc[-1]["fecha"])
        if len(p) and ages["nutrients"] <= C.MAX_NUTRIENT_AGE_DAYS:
            row["tp_context"] = p.iloc[-1]["tp_mgl"]
        else:
            missing_context.append("NUTRIENTS")
    wq = _load(WQ, wb)
    ages["waterquality"] = None
    if len(wq):
        p = wq[wq["fecha"] <= t0].sort_values("fecha")
        if len(p):
            ages["waterquality"] = _age_days(t0, p.iloc[-1]["fecha"])
        if len(p) and ages["waterquality"] <= C.MAX_WATERQUALITY_AGE_DAYS:
            for v in WQ_VARS:
                if v in wq.columns:
                    row[v] = p.iloc[-1][v]
        else:
            missing_context.append("WATERQUALITY")
    result = (pd.DataFrame([row]), float(chl0), t0)
    if include_meta:
        return (*result, {"feature_ages": ages, "missing_context": missing_context})
    return result


def forecast_body(wb, t0=None, spec_override=None, res=None):
    """Pronóstico ESTRUCTURADO 0-7 d para un cuerpo (sin imprimir). Fuente única de verdad
    de la inferencia: la reusa tanto predict_body (CLI) como run_forecast (bucle operativo).
    Causal: solo datos <= t0 (build_features). Devuelve un dict con metadatos + lista por
    horizonte (chl_pred, banda, probabilidad y alerta de anomalia), o None si faltan datos.

    spec_override (opcional): dict {banda/índice espectral: valor} con la mediana espectral de
    una escena EXTERNA (p.ej. un GeoTIFF subido en la app). Si se pasa, reemplaza las columnas
    espectrales del vector de features; el contexto NO espectral (autorreg/ERA5/in-situ) sigue
    siendo el del cuerpo en t0. No altera el comportamiento por defecto (spec_override=None)."""
    group = GROUP[wb]
    sc = _load(SCENE, wb).sort_values("fecha")
    if sc.empty:
        return None
    t0 = sc["fecha"].max() if t0 is None else pd.Timestamp(t0).normalize()
    scpast = sc[sc["fecha"] <= t0]
    if scpast.empty:
        return None
    n_water_px = int(scpast.iloc[-1]["n_water_px"]) if "n_water_px" in scpast.columns else None
    built = build_features(wb, t0, include_meta=True)
    if built is None:
        return None
    X, chl0, t0, feature_meta = built
    if spec_override:                          # escena externa: usa SU espectral (app: GeoTIFF subido)
        for f, v in spec_override.items():
            if f in X.columns:
                X[f] = v
    # recursos: precargados (res, p.ej. cache de Streamlit) o leídos de disco (comportamiento normal)
    thr_map = res["thr_body"] if res else joblib.load(os.path.join(MODELS, "thr_body.pkl"))
    thr_body = thr_map.get(wb, 10.0)
    thr_floracion = C.alert_threshold_ugl(thr_body)   # p85 acotado al nivel biológico (<=24 ug/L)
    thr_elevada = C.elevated_threshold_ugl(thr_floracion)  # banda de aviso (< floracion, ordenado)
    if res:
        calib = res["calib"].get(group)
    else:
        calib_f = os.path.join(MODELS, f"alert_calib_{group}.pkl")
        calib = joblib.load(calib_f) if os.path.exists(calib_f) else None
    pthr = calib["threshold"] if calib is not None else 0.5

    horizons = []
    for h in [1, 3, 5, 7]:
        if res:
            b = res["bundles"].get((group, h))
            if b is None:
                continue
        else:
            f = os.path.join(MODELS, f"{group}_h{h}.pkl")
            if not os.path.exists(f):
                continue
            b = joblib.load(f)
        feats = b["feats"]
        Xh = X.reindex(columns=feats)              # asegura columnas/orden del modelo
        chl = float(np.expm1(b["reg"].predict(Xh)[0]))
        # banda de incertidumbre (CQR P10-P90 + offset conformal), en ug/L
        p10 = p90 = None
        if b.get("qlo") is not None:
            Qc = b.get("q_conformal", 0.0)
            a, c = float(b["qlo"].predict(Xh)[0]), float(b["qhi"].predict(Xh)[0])
            p10 = max(float(np.expm1(min(a, c) - Qc)), 0.0)
            p90 = max(float(np.expm1(max(a, c) + Qc)), 0.0)
            p10 = min(p10, chl)
            p90 = max(p90, chl)
        # alerta: ensamble XGB_clf + Red
        probs = []
        if b["clf"] is not None:
            probs.append(float(b["clf"].predict_proba(Xh)[0, 1]))
        Xs = b["sc"].transform(b["imp"].transform(Xh))
        if res:
            net = res["nn"][(group, h)]
        else:
            net = HABNet(b["n_in"]); net.load_state_dict(torch.load(os.path.join(MODELS, f"{group}_h{h}_nn.pt")))
            net.eval()
        with torch.no_grad():
            _, logit = net(torch.tensor(Xs, dtype=torch.float32))
            probs.append(float(torch.sigmoid(logit)[0]))
        p = float(np.mean(probs))
        if calib is not None:                       # calibrar a probabilidad operativa
            p = float(calib["iso"].predict([p])[0])
        event_threshold = float(b.get("event_thresholds", {}).get(wb, thr_body))
        alerta_anomalia = bool(p >= pthr)
        # NIVEL = magnitud biologica; ALERTA = probabilidad calibrada de exceder el P85
        # validado. Se conservan como dos salidas distintas, sin mezclar sus umbrales.
        nivel = C.biomass_level(chl, thr_floracion, thr_elevada)
        horizons.append({"horizon": h, "chl_pred": chl, "p10": p10, "p90": p90,
                         "prob_riesgo": p, "nivel": nivel,
                         "event_threshold_ugl": event_threshold,
                         "alerta_anomalia": alerta_anomalia,
                         "riesgo": alerta_anomalia,
                         "floracion_magnitud": bool(nivel == "floracion"),
                         "elevada": bool(nivel in ("elevada", "floracion"))})
    return {"water_body": wb, "group": group, "t0": t0, "chl0": float(chl0),
            "thr_body": float(thr_body), "thr_floracion": float(thr_floracion),
            "thr_elevada": float(thr_elevada), "n_water_px": n_water_px,
            "alert_threshold": float(pthr), "horizons": horizons,
            "feature_ages": feature_meta["feature_ages"],
            "missing_context": feature_meta["missing_context"]}


def predict_body(wb, t0=None):
    fc = forecast_body(wb, t0)
    if fc is None:
        print(f"{wb}: sin escenas / datos suficientes."); return
    t0 = fc["t0"]
    confianza, flags, age = guards.evaluate_guards(
        wb, t0, fc["n_water_px"], feature_ages=fc.get("feature_ages"),
        missing_context=fc.get("missing_context"))
    nota_conf = f" | confianza={confianza}" + (f" ({', '.join(flags)})" if flags else "")
    print(f"\n=== {wb.upper()} ({fc['group']}) | t0={t0.date()} (hace {age}d) | "
          f"chl-a actual={fc['chl0']:.1f} ug/L | floración (chl-a)>={fc['thr_floracion']:.1f} ug/L | "
          f"biomasa elevada>={fc['thr_elevada']:.1f} ug/L{nota_conf} ===")
    print(f"  {'horizonte':10s} {'chl-a_pred(ug/L)':>16s} {'banda P10-P90':>16s} {'prob_anomalia':>13s} {'NIVEL':>16s}")
    for hh in fc["horizons"]:
        banda = f"{hh['p10']:.1f}-{hh['p90']:.1f}" if hh["p10"] is not None else ""
        print(f"  +{hh['horizon']}d{'':6s} {hh['chl_pred']:>16.1f} {banda:>16s} "
              f"{hh['prob_riesgo']:>13.2f} {C.LEVEL_ES[hh['nivel']]:>16s}")
    print("  Nota: prob_anomalia/alerta = excedencia P85 calibrada; NIVEL = magnitud de clorofila. "
          "Ninguna confirma toxicidad; requieren verificación de campo.")
    print("  Banda P10-P90 = intervalo de incertidumbre calibrado (CQR, cobertura ~80%).")


def main():
    args = sys.argv[1:]
    bodies = [args[0]] if args else list(GROUP)
    t0 = args[1] if len(args) > 1 else None
    for wb in bodies:
        predict_body(wb, t0)


if __name__ == "__main__":
    main()
