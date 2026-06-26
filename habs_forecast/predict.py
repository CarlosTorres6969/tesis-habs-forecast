"""
predict.py — PREDICTOR DESPLEGABLE. Dado un cuerpo de agua y una fecha t0 (por defecto la
ultima escena disponible), produce el pronostico 0-7 dias:
    - INTENSIDAD: clorofila-a esperada (ug/L) por horizonte = proxy de BIOMASA algal
    - RIESGO: probabilidad de biomasa algal elevada / clorofila-a anomala (ensamble Red+XGBoost)
      y decision (si/no). NB: clorofila-a alta indica mas biomasa, NO confirma floracion NOCIVA
      (toxicidad); la alerta senala condiciones de RIESGO que ameritan verificacion de campo
      (identificacion de cianobacterias, toxinas). Sentinel-2 no distingue cianobacterias.

Construye el vector de features en t0 desde los artefactos (igual que match_pairs, pero sin
el target futuro), carga los modelos de produccion (train_final.py) y predice.

Uso:  python predict.py                 -> todos los cuerpos, ultima escena
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

T = os.path.join(C.DIR_OUT, "targets")
SCENE = os.path.join(C.DIR_STATE, "scene_state.csv")
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
    d = pd.read_csv(path)
    if "fecha" in d.columns:
        d["fecha"] = pd.to_datetime(d["fecha"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    return d[d["water_body"] == wb] if "water_body" in d.columns else d


def build_features(wb, t0):
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
    chl0 = past.iloc[-1]["chl_ugl"]
    def near(days):
        w = past[(past["fecha"] >= t0 - pd.Timedelta(days=days + 2)) &
                 (past["fecha"] <= t0 - pd.Timedelta(days=days - 2))]
        return w["chl_ugl"].mean() if len(w) else chl0
    roll7 = past[past["fecha"] >= t0 - pd.Timedelta(days=7)]["chl_ugl"].mean()
    l3, l7 = near(3), near(7)
    row.update({"chl_t0": chl0, "log_chl_t0": np.log1p(max(chl0, 0)), "chl_lag3": l3,
                "chl_lag7": l7, "chl_roll7": roll7, "chl_trend7": chl0 - l7})
    # --- ERA5 (en/antes de t0 + media movil 7d) ---
    er = _load(ERA5D, wb).sort_values("fecha")
    ep = er[er["fecha"] <= t0]
    if len(ep):
        last = ep.iloc[-1]
        w7 = ep[ep["fecha"] >= t0 - pd.Timedelta(days=7)]
        for v in ERA5_BASE:
            if v in ep.columns:
                row[v] = last[v]
                row[f"{v}_roll7"] = w7[v].mean()
    # --- nutrientes + calidad de agua (contexto <= t0) ---
    nu = _load(NUTR, wb)
    if "tp_mgl" in nu.columns and len(nu):
        p = nu[nu["fecha"] <= t0].sort_values("fecha")
        if len(p):
            row["tp_context"] = p.iloc[-1]["tp_mgl"]
    wq = _load(WQ, wb)
    if len(wq):
        p = wq[wq["fecha"] <= t0].sort_values("fecha")
        if len(p):
            for v in WQ_VARS:
                if v in wq.columns:
                    row[v] = p.iloc[-1][v]
    return pd.DataFrame([row]), float(chl0), t0


def predict_body(wb, t0=None):
    group = GROUP[wb]
    sc = _load(SCENE, wb).sort_values("fecha")
    if sc.empty:
        print(f"{wb}: sin escenas."); return
    t0 = sc["fecha"].max() if t0 is None else pd.Timestamp(t0).normalize()
    built = build_features(wb, t0)
    if built is None:
        print(f"{wb}: sin datos suficientes en {t0.date()}"); return
    X, chl0, t0 = built
    thr_body = joblib.load(os.path.join(MODELS, "thr_body.pkl")).get(wb, 10.0)
    # calibrador + umbral operativo de alerta (calibrate_alert.py)
    calib_f = os.path.join(MODELS, f"alert_calib_{group}.pkl")
    calib = joblib.load(calib_f) if os.path.exists(calib_f) else None

    pthr = calib["threshold"] if calib is not None else 0.5
    print(f"\n=== {wb.upper()} ({group}) | t0={t0.date()} | chl-a actual={chl0:.1f} ug/L | "
          f"biomasa alta (chl-a)>={thr_body:.1f} ug/L | dispara si prob>={pthr:.2f} ===")
    print(f"  {'horizonte':10s} {'chl-a_pred(ug/L)':>16s} {'banda P10-P90':>16s} {'prob_riesgo':>12s} {'RIESGO':>8s}")
    for h in [1, 3, 5, 7]:
        f = os.path.join(MODELS, f"{group}_h{h}.pkl")
        if not os.path.exists(f):
            continue
        b = joblib.load(f)
        feats = b["feats"]
        Xh = X.reindex(columns=feats)              # asegura columnas/orden del modelo
        chl = float(np.expm1(b["reg"].predict(Xh)[0]))
        # banda de incertidumbre (CQR P10-P90 + offset conformal), en ug/L
        banda = ""
        if b.get("qlo") is not None:
            Qc = b.get("q_conformal", 0.0)
            a, c = float(b["qlo"].predict(Xh)[0]), float(b["qhi"].predict(Xh)[0])
            lo = max(float(np.expm1(min(a, c) - Qc)), 0.0)
            hi = float(np.expm1(max(a, c) + Qc))
            banda = f"{lo:.1f}-{hi:.1f}"
        # alerta: ensamble XGB_clf + Red
        probs = []
        if b["clf"] is not None:
            probs.append(float(b["clf"].predict_proba(Xh)[0, 1]))
        Xs = b["sc"].transform(b["imp"].transform(Xh))
        net = HABNet(b["n_in"]); net.load_state_dict(torch.load(os.path.join(MODELS, f"{group}_h{h}_nn.pt")))
        net.eval()
        with torch.no_grad():
            _, logit = net(torch.tensor(Xs, dtype=torch.float32))
            probs.append(float(torch.sigmoid(logit)[0]))
        p = float(np.mean(probs))
        if calib is not None:                       # calibrar + umbral operativo
            p = float(calib["iso"].predict([p])[0])
            alerta = "SI" if p >= calib["threshold"] else "no"
        else:
            alerta = "SI" if p >= 0.5 else "no"
        print(f"  +{h}d{'':6s} {chl:>16.1f} {banda:>16s} {p:>12.2f} {alerta:>8s}")
    print("  Nota: RIESGO = biomasa algal elevada (clorofila-a anomala), NO confirma toxicidad; "
          "requiere verificacion de campo.")
    print("  Banda P10-P90 = intervalo de incertidumbre calibrado (CQR, cobertura ~80%).")


def main():
    args = sys.argv[1:]
    bodies = [args[0]] if args else list(GROUP)
    t0 = args[1] if len(args) > 1 else None
    for wb in bodies:
        predict_body(wb, t0)


if __name__ == "__main__":
    main()
