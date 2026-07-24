"""
validate_articles_forecast_historical.py — Corre el MODELO REAL (las 23 variables del pipeline
habs_forecast) sobre las fechas de los artículos de Yojoa (2020-2022), reconstruyendo el stack
histórico completo:
  - 9 ESPECTRALES  : de las escenas Sentinel-2 históricas (build_scene_state._scene_features).
  - 5 AUTORREGRESIVAS: del target VIIRS histórico 2018-2022 (fetch_historical_target.py).
  - 9 ERA5         : de ERA5-Land histórico (fetch_era5_historical_yojoa.py).
Replica EXACTAMENTE la lógica de predict.build_features + forecast_body (misma inferencia:
regresión XGBoost + alerta ensamble XGB_clf+Red calibrada). NO reentrena nada.

Es la respuesta a "con las mismas variables del modelo?": SÍ, las 23, en las fechas de los artículos.
Salida: artifacts/validation_articles/forecast_historical_yojoa.csv
"""
from __future__ import annotations
import os, glob, re
import numpy as np
import pandas as pd
import joblib, torch
import config as C
from build_scene_state import _scene_features
from train_nn import HABNet

MODELS = C.DIR_MODELS
SCENES = os.path.join(C.DIR_OUT, "validation_articles", "s2_historico", "yojoa")
HIST_TGT = os.path.join(C.DIR_OUT, "targets", "historical_target_honduras_2018_2022.csv")
HIST_ERA5 = os.path.join(C.DIR_STATE, "era5_daily_hist_yojoa.csv")
OUT = os.path.join(C.DIR_OUT, "validation_articles", "forecast_historical_yojoa.csv")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

SPEC = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
ERA5_BASE = ["temp_air_2m", "solar_radiation", "precipitation", "wind_speed_10m", "surface_pressure"]
GROUP = "freshwater"


def _hist_scene_state():
    rows = []
    for t in sorted(glob.glob(os.path.join(SCENES, "*.tif"))):
        m = DATE_RE.search(os.path.basename(t))
        if not m:
            continue
        try:
            f = _scene_features(t)
        except Exception:
            f = None
        if f is None:
            continue
        f["fecha"] = pd.Timestamp(m.group(1))
        rows.append(f)
    return pd.DataFrame(rows).sort_values("fecha")


def _build_features(t0, scene_row, tgt, era5):
    """Replica predict.build_features con fuentes históricas. tgt/era5 filtrados a Yojoa."""
    row = {f: scene_row[f] for f in SPEC}
    # autorregresivo
    past = tgt[tgt["fecha"] <= t0]
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
    # ERA5
    ep = era5[era5["fecha"] <= t0]
    if len(ep):
        last = ep.iloc[-1]; w7 = ep[ep["fecha"] >= t0 - pd.Timedelta(days=7)]
        for v in ERA5_BASE:
            if v in ep.columns:
                row[v] = last[v]; row[f"{v}_roll7"] = w7[v].mean()
    return pd.DataFrame([row]), float(chl0)


def main():
    ss = _hist_scene_state()
    if ss.empty:
        print("Sin scene_state historico."); return
    tgt = pd.read_csv(HIST_TGT, parse_dates=["fecha"])
    tgt["fecha"] = tgt["fecha"].dt.tz_localize(None)
    tgt = tgt[tgt["water_body"] == "yojoa"].sort_values("fecha")
    era5 = pd.read_csv(HIST_ERA5, parse_dates=["fecha"]).sort_values("fecha")

    thr_map = joblib.load(os.path.join(MODELS, "thr_body.pkl"))
    thr_flor = C.alert_threshold_ugl(thr_map.get("yojoa", 10.0))
    thr_elev = C.elevated_threshold_ugl(thr_flor)
    calib_f = os.path.join(MODELS, f"alert_calib_{GROUP}.pkl")
    calib = joblib.load(calib_f) if os.path.exists(calib_f) else None

    bundles = {}
    for h in (1, 3, 5, 7):
        p = os.path.join(MODELS, f"{GROUP}_h{h}.pkl")
        if os.path.exists(p):
            bundles[h] = joblib.load(p)

    rows = []
    for _, sc in ss.iterrows():
        t0 = sc["fecha"]
        built = _build_features(t0, sc, tgt, era5)
        if built is None:
            continue
        X, chl0 = built
        rec = {"fecha": t0.date(), "chl0_autoreg_ugL": round(chl0, 1)}
        for h, b in bundles.items():
            feats = b["feats"]
            Xh = X.reindex(columns=feats)
            chl = float(np.expm1(b["reg"].predict(Xh)[0]))
            probs = []
            if b["clf"] is not None:
                probs.append(float(b["clf"].predict_proba(Xh)[0, 1]))
            Xs = b["sc"].transform(b["imp"].transform(Xh))
            net = HABNet(b["n_in"]); net.load_state_dict(
                torch.load(os.path.join(MODELS, f"{GROUP}_h{h}_nn.pt"))); net.eval()
            with torch.no_grad():
                _, logit = net(torch.tensor(Xs, dtype=torch.float32))
                probs.append(float(torch.sigmoid(logit)[0]))
            p = float(np.mean(probs))
            if calib is not None:
                p = float(calib["iso"].predict([p])[0])
            nivel = C.biomass_level(chl, thr_flor, thr_elev)
            rec[f"chl_pred_h{h}"] = round(chl, 1)
            rec[f"prob_alerta_h{h}"] = round(p, 2)
            rec[f"nivel_h{h}"] = nivel
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("fecha")
    df.to_csv(OUT, index=False)
    # resumen
    print(f"MODELO REAL (23 variables) en fechas de artículos de Yojoa — {len(df)} escenas\n")
    cols = ["fecha", "chl0_autoreg_ugL", "chl_pred_h3", "prob_alerta_h3", "nivel_h3",
            "chl_pred_h7", "prob_alerta_h7", "nivel_h7"]
    print(df[cols].to_string(index=False))
    det = (df["nivel_h3"].isin(["floracion", "elevada"])).sum()
    print(f"\nDetección h3 (floracion/elevada): {det}/{len(df)} escenas")
    print(f"prob_alerta_h3 media: {df['prob_alerta_h3'].mean():.2f} | h7: {df['prob_alerta_h7'].mean():.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
