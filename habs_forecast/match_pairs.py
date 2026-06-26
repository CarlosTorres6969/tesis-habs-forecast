"""
match_pairs.py — Empareja PREDICTORES en t0 con TARGET en t0+h para pronostico 0-7 d.

  Predictor  : estado por escena Sentinel-2 (artifacts/state_series/scene_state.csv)  [t0]
  Target     : clorofila satelital diaria VIIRS (artifacts/targets/satellite_chl_daily.csv) [t0+h]
               (fuente INDEPENDIENTE de S2 -> rompe la circularidad del pipeline anterior)

Para cada escena S2 en t0 y cada horizonte h en {0,1,3,5,7}, busca el target en t0+h dentro
de la tolerancia (config.HORIZON_TOLERANCE) y crea un par. Agnostico a la fuente del target:
basta cambiar TARGET_FILE (in-situ, OLCI, MODIS).

Sin fuga: el target en t0+h NUNCA entra como predictor; solo se usa estado <= t0.
Salida: artifacts/pairs/pairs_forecast.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C

_SCENE_RAW  = os.path.join(C.DIR_STATE, "scene_state.csv")
_SCENE_HARM = os.path.join(C.DIR_STATE, "scene_state_harmonized.csv")   # Landsat ajustado a S2
SCENE_FILE  = _SCENE_HARM if os.path.exists(_SCENE_HARM) else _SCENE_RAW
# target optimo por cuerpo: combinado (OLCI costa + VIIRS-corregido lagos) > corregido > crudo
_COMB = os.path.join(C.DIR_OUT, "targets", "combined_target.csv")
_CORR = os.path.join(C.DIR_OUT, "targets", "satellite_chl_corrected.csv")
_RAW = os.path.join(C.DIR_OUT, "targets", "satellite_chl_daily.csv")
TARGET_FILE = _COMB if os.path.exists(_COMB) else (_CORR if os.path.exists(_CORR) else _RAW)
ERA5_FILE   = os.path.join(C.DIR_STATE, "era5_daily.csv")
OUT = os.path.join(C.DIR_PAIRS, "pairs_forecast.csv")

SPECTRAL_FEATURES = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
ERA5_BASE = ["temp_air_2m", "solar_radiation", "precipitation",
             "wind_speed_10m", "surface_pressure"]


def enrich_era5(df):
    """Anade drivers ERA5 en t0 + medias moviles causales (3,7d). Merge-asof hacia atras
    (solo datos <= t0). Sin fuga temporal."""
    if not os.path.exists(ERA5_FILE):
        print("  (sin ERA5: se omite enriquecimiento)"); return df
    e = pd.read_csv(ERA5_FILE, parse_dates=["fecha"])
    feats = []
    for wb, g in e.groupby("water_body"):
        g = g.sort_values("fecha").set_index("fecha")
        out = pd.DataFrame(index=g.index)
        for v in ERA5_BASE:
            if v in g:
                out[v] = g[v]
                out[f"{v}_roll7"] = g[v].rolling("7D").mean()
        out = out.reset_index(); out["water_body"] = wb
        feats.append(out)
    ef = pd.concat(feats, ignore_index=True).sort_values("fecha")
    df = df.sort_values("fecha_t0")
    merged = pd.merge_asof(df, ef, left_on="fecha_t0", right_on="fecha",
                           by="water_body", direction="backward",
                           tolerance=pd.Timedelta(days=10))
    return merged.drop(columns=["fecha"], errors="ignore")


NUTRIENT_FILE = os.path.join(C.DIR_OUT, "targets", "nutrients_daily.csv")


def enrich_nutrients(df):
    """Anade fosforo in-situ como CONTEXTO de baja frecuencia (carry-forward <= t0,
    tolerancia amplia: el fosforo cambia lento). Solo Florida (Okeechobee/Tampa);
    Honduras queda NaN -> XGBoost lo maneja nativo. Sin fuga (solo datos <= t0)."""
    if not os.path.exists(NUTRIENT_FILE):
        return df
    n = pd.read_csv(NUTRIENT_FILE, parse_dates=["fecha"]).sort_values("fecha")
    n = n[["water_body", "fecha", "tp_mgl"]].rename(columns={"tp_mgl": "tp_context"})
    df = df.sort_values("fecha_t0")
    merged = pd.merge_asof(df, n, left_on="fecha_t0", right_on="fecha",
                           by="water_body", direction="backward",
                           tolerance=pd.Timedelta(days=45))
    return merged.drop(columns=["fecha"], errors="ignore")


WQ_FILE = os.path.join(C.DIR_OUT, "targets", "waterquality_daily.csv")
WQ_VARS = ["water_temp", "do_mgl", "ph", "turbidity_insitu", "spec_cond", "secchi", "ammonia"]


def enrich_waterquality(df):
    """Anade calidad de agua in-situ como CONTEXTO en t0 (carry-forward <= t0, tol 14d).
    Solo Florida; Honduras NaN (XGBoost nativo). Medidas en t0 -> sin fuga del futuro."""
    if not os.path.exists(WQ_FILE):
        return df
    w = pd.read_csv(WQ_FILE, parse_dates=["fecha"]).sort_values("fecha")
    keep = ["water_body", "fecha"] + [c for c in WQ_VARS if c in w.columns]
    df = df.sort_values("fecha_t0")
    merged = pd.merge_asof(df, w[keep], left_on="fecha_t0", right_on="fecha",
                           by="water_body", direction="backward",
                           tolerance=pd.Timedelta(days=14))
    return merged.drop(columns=["fecha"], errors="ignore")


def build():
    scene = pd.read_csv(SCENE_FILE, parse_dates=["fecha"])
    tgt = pd.read_csv(TARGET_FILE, parse_dates=["fecha"])
    # normalizar a fecha naive (el target VIIRS viene tz-aware UTC)
    for d in (scene, tgt):
        if getattr(d["fecha"].dt, "tz", None) is not None:
            d["fecha"] = d["fecha"].dt.tz_localize(None)
        d["fecha"] = d["fecha"].dt.normalize()
    tgt = tgt.rename(columns={"chl_ugl": "chl_target"})[["water_body", "fecha", "chl_target"]]

    # indexar target por cuerpo -> serie ordenada para busqueda por ventana
    tgt_by_body = {wb: g.sort_values("fecha").reset_index(drop=True)
                   for wb, g in tgt.groupby("water_body")}

    def ar_features(tb, t0):
        """Trayectoria reciente de clorofila (predictores autorregresivos, causales <= t0).
        Sin fuga: solo usa observaciones del target en o antes de t0.
        Incluye dinamica temporal (tasa/aceleracion/volatilidad) y estacionalidad CAUSAL
        (climatologia construida solo con el pasado <= t0)."""
        past = tb[tb["fecha"] <= t0]
        if past.empty:
            return None
        chl_t0 = past.iloc[-1]["chl_target"]                      # ultimo dato <= t0
        last_date = past.iloc[-1]["fecha"]
        def near(days):
            w = past[(past["fecha"] >= t0 - pd.Timedelta(days=days + 2)) &
                     (past["fecha"] <= t0 - pd.Timedelta(days=days - 2))]
            return w["chl_target"].mean() if len(w) else chl_t0
        def window(days):
            return past[past["fecha"] >= t0 - pd.Timedelta(days=days)]["chl_target"]
        roll7 = window(7).mean()
        chl_l3, chl_l7, chl_l14 = near(3), near(7), near(14)
        w14, w30 = window(14), window(30)
        # --- dinamica temporal (causal) ---
        rate3 = (chl_t0 - chl_l3) / 3.0                           # tasa de cambio /dia
        accel = chl_t0 - 2.0 * chl_l7 + chl_l14                   # 2da diferencia (~7d) = aceleracion
        days_since = (t0 - last_date).days                        # frescura del dato (incertidumbre)
        roll14 = w14.mean(); roll30 = w30.mean()
        std14 = float(w14.std()) if len(w14) > 1 else 0.0         # volatilidad reciente
        max14 = float(w14.max()) if len(w14) else chl_t0          # pico reciente
        # --- estacionalidad / climatologia CAUSAL (solo pasado <= t0) ---
        doy = t0.dayofyear
        doy_sin = np.sin(2 * np.pi * doy / 365.25)
        doy_cos = np.cos(2 * np.pi * doy / 365.25)
        pdoy = past["fecha"].dt.dayofyear                         # climatologia: +-25 dias-del-ano
        dd = (pdoy - doy).abs(); dd = np.minimum(dd, 365 - dd)
        clim_w = past[dd <= 25]["chl_target"]
        chl_clim = float(clim_w.mean()) if len(clim_w) >= 3 else np.nan
        chl_anom = (chl_t0 - chl_clim) if pd.notna(chl_clim) else np.nan
        return {
            "chl_t0": chl_t0, "log_chl_t0": np.log1p(max(chl_t0, 0)),
            "chl_lag3": chl_l3, "chl_lag7": chl_l7,
            "chl_roll7": roll7, "chl_trend7": chl_t0 - chl_l7,
            "chl_rate3": rate3, "chl_accel": accel, "days_since_obs": days_since,
            "chl_roll14": roll14, "chl_roll30": roll30, "chl_std14": std14, "chl_max14": max14,
            "doy_sin": doy_sin, "doy_cos": doy_cos, "chl_clim": chl_clim, "chl_anom": chl_anom,
        }

    rows = []
    for _, s in scene.iterrows():
        wb = s["water_body"]
        if wb not in tgt_by_body:
            continue
        tb = tgt_by_body[wb]
        t0 = s["fecha"]
        ar = ar_features(tb, t0)
        if ar is None:
            continue
        for h in C.HORIZONS:
            lo, hi = C.HORIZON_TOLERANCE[h]
            win = tb[(tb["fecha"] >= t0 + pd.Timedelta(days=lo)) &
                     (tb["fecha"] <= t0 + pd.Timedelta(days=hi))]
            if win.empty:
                continue
            # el target mas cercano al horizonte nominal dentro de la ventana
            win = win.assign(dist=(win["fecha"] - (t0 + pd.Timedelta(days=h))).abs())
            best = win.sort_values("dist").iloc[0]
            row = {
                "water_body": wb, "group": s["group"], "horizon": h,
                "fecha_t0": t0, "fecha_target": best["fecha"],
                "gap_real": (best["fecha"] - t0).days,
                "n_water_px": s["n_water_px"],
                "chl_target": best["chl_target"],
                "log_chl_target": np.log1p(max(best["chl_target"], 0)),
                "hab_target": int(best["chl_target"] >= C.THRESHOLDS["moderate"]),
            }
            for f in SPECTRAL_FEATURES:
                row[f] = s[f]
            row.update(ar)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = enrich_era5(df)
    df = enrich_nutrients(df)
    df = enrich_waterquality(df)

    # umbral de alerta RELATIVO por cuerpo (percentil de la climatologia del target del cuerpo)
    clim = pd.read_csv(TARGET_FILE)
    thr_body = (clim.groupby("water_body")["chl_ugl"]
                .quantile(C.RELATIVE_PERCENTILE / 100.0).to_dict())
    df["thr_body"] = df["water_body"].map(thr_body)
    if C.USE_RELATIVE_THRESHOLD:
        df["hab_target"] = (df["chl_target"] >= df["thr_body"]).astype(int)

    # quitar duplicados EXACTOS (misma escena procesada dos veces -> fila identica). Las escenas
    # distintas del mismo dia (tiles/pasadas con espectro diferente) NO son duplicados y se conservan.
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n0:
        print(f"  (quitados {n0 - len(df)} pares duplicados exactos)")

    # ORDEN CANONICO determinista: XGBoost con subsample/colsample y seed fijo elige filas segun
    # su POSICION; si el orden del DataFrame cambia (p.ej. al reconstruir con otro cuerpo), el mismo
    # seed muestrea filas distintas -> modelo distinto -> skill que "baila" entre corridas. Ordenar
    # de forma canonica (cuerpo, horizonte, fecha) hace el pipeline REPRODUCIBLE corrida a corrida.
    df = df.sort_values(["water_body", "horizon", "fecha_t0", "fecha_target"]).reset_index(drop=True)

    os.makedirs(C.DIR_PAIRS, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n=== umbral relativo por cuerpo (ug/L, P{}) ===".format(C.RELATIVE_PERCENTILE))
    for wb, t in sorted(thr_body.items()):
        print(f"  {wb:12s}: {t:.1f}")

    print(f"Escenas: {len(scene)} | target dias-cuerpo: {len(tgt)}")
    print(f"Pares creados: {len(df)} -> {OUT}\n")
    if len(df):
        print("=== pares por grupo y horizonte ===")
        print(df.pivot_table(index="group", columns="horizon",
                             values="hab_target", aggfunc="count", fill_value=0).to_string())
        print("\n=== HAB+ (%) por grupo y horizonte ===")
        print(df.pivot_table(index="group", columns="horizon",
                             values="hab_target", aggfunc="mean").round(2).to_string())
        print("\n=== pares por cuerpo ===")
        print(df.groupby(["group", "water_body"]).size().to_string())
    return df


if __name__ == "__main__":
    build()
