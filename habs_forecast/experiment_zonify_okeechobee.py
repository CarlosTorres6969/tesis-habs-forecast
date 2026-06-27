"""
experiment_zonify_okeechobee.py — EXPERIMENTO (contenido): ¿zonificar Okeechobee sube RMSE/R2?

Hipotesis: predecir un solo promedio de un lago grande y heterogeneo (Okeechobee, ~20 ug/L de
dispersion intra-dia) limita el skill. Dividirlo en zonas deberia mejorar la predictibilidad.

Diseno (NO toca produccion; todo va a artifacts/experiments/zonify/):
  1. Zonas = 4 cuadrantes del bbox de Okeechobee (NW/NE/SW/SE).
  2. TARGET por zona: VIIRS via ERDDAP por sub-bbox (reusa fetch_satellite_chl._fetch_body).
  3. PREDICTOR por zona: re-agrega los rasters S2 existentes por zona (pixel UTM -> lat/lon -> zona).
  4. Pares causales (espectral + autorregresivo) IGUALES para zonas y para el cuerpo entero
     (mismo codigo, solo cambia la granularidad espacial) -> comparacion limpia.
  5. Metricas por horizonte (split temporal walk-forward):
       (a) CUERPO: predecir chl del cuerpo desde predictor del cuerpo.
       (b) ZONAS: predecir chl de zona desde predictor de zona (pooled).
       (c) ZONAS->CUERPO (prueba decisiva): promediar las 4 predicciones de zona y comparar
           contra el MISMO target del cuerpo. Si (c) mejora a (a) -> zonificar ayuda de verdad.

Uso:  python experiment_zonify_okeechobee.py
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import rasterio
from pyproj import Transformer
from sklearn.metrics import mean_squared_error, r2_score
import config as C
from train import _model
from fetch_satellite_chl import _fetch_body

OUTDIR = os.path.join(C.DIR_OUT, "experiments", "zonify")
os.makedirs(OUTDIR, exist_ok=True)
Z_TARGET = os.path.join(OUTDIR, "zoned_target.csv")
Z_SCENE = os.path.join(OUTDIR, "zoned_scene_state.csv")
HORIZONS = [1, 3, 5, 7]
SPEC = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
AR = ["log_chl_t0", "chl_lag3", "chl_lag7", "chl_roll7", "chl_trend7"]
FEATS = SPEC + AR

# bbox Okeechobee (lat_lo, lat_hi, lon_lo, lon_hi) y punto medio para 4 cuadrantes
LA_LO, LA_HI, LO_LO, LO_HI = 26.70, 27.20, -81.10, -80.60
MLA, MLO = (LA_LO + LA_HI) / 2, (LO_LO + LO_HI) / 2
ZONES = {  # zona -> (lat_lo, lat_hi, lon_lo, lon_hi)
    "okee_SW": (LA_LO, MLA, LO_LO, MLO), "okee_SE": (LA_LO, MLA, MLO, LO_HI),
    "okee_NW": (MLA, LA_HI, LO_LO, MLO), "okee_NE": (MLA, LA_HI, MLO, LO_HI),
}


def fetch_zoned_target():
    if os.path.exists(Z_TARGET):
        print(f"  target zonificado ya existe -> {Z_TARGET}"); return pd.read_csv(Z_TARGET, parse_dates=["fecha"])
    frames = []
    for z, (la_lo, la_hi, lo_lo, lo_hi) in ZONES.items():
        print(f"  VIIRS zona {z} ...")
        d = _fetch_body(z, la_lo, la_hi, lo_lo, lo_hi)
        if len(d):
            frames.append(d[["water_body", "fecha", "chl_ugl"]])
    out = pd.concat(frames, ignore_index=True)
    out["fecha"] = pd.to_datetime(out["fecha"]).dt.tz_localize(None).dt.normalize()
    out.to_csv(Z_TARGET, index=False)
    print(f"  -> {Z_TARGET} ({len(out)} dias-zona)")
    return out


def build_zoned_scene():
    if os.path.exists(Z_SCENE):
        print(f"  scene_state zonificado ya existe -> {Z_SCENE}"); return pd.read_csv(Z_SCENE, parse_dates=["fecha"])
    tifs = sorted(glob.glob(os.path.join(C.DIR_IMAGENES, "Okeechobee", "**", "*.tif"), recursive=True))
    tr = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)
    rows = []
    import re
    for path in tifs:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
        if not m:
            continue
        fecha = pd.Timestamp(m.group(1))
        with rasterio.open(path) as ds:
            oh, ow = ds.height // 4, ds.width // 4
            arr = ds.read(out_shape=(ds.count, oh, ow)).astype("float32")
            T = ds.transform * ds.transform.scale(ds.width / ow, ds.height / oh)
        if arr.shape[0] < 5:
            continue
        B2, B3, B4, B5, B8 = arr
        if np.nanmax(arr) > C.BAND_SCALE_THRESHOLD:
            B2, B3, B4, B5, B8 = (b / 10000.0 for b in (B2, B3, B4, B5, B8))
        eps = 1e-10
        ndwi = (B3 - B8) / (B3 + B8 + eps); ndvi = (B8 - B4) / (B8 + B4 + eps)
        water = (B2 > 0) & (B3 > 0) & (B4 > 0) & (B5 > 0) & (B8 > 0) & (ndwi > C.NDWI_MIN) & (ndvi < C.NDVI_MAX)
        if water.sum() < 50:
            continue
        # coords lat/lon de cada pixel (centro)
        jj, ii = np.meshgrid(np.arange(ow), np.arange(oh))
        xs, ys = rasterio.transform.xy(T, ii, jj)
        lon, lat = tr.transform(np.asarray(xs), np.asarray(ys))
        lon = np.asarray(lon).reshape(oh, ow); lat = np.asarray(lat).reshape(oh, ow)
        idx = C.spectral_indices(B2, B3, B4, B5, B8)
        feat2d = {"B2": B2, "B3": B3, "B4": B4, "B5": B5, "B8": B8, "NDCI": idx["NDCI"],
                  "CI_red": idx["CI_red"], "FAI": idx["FAI"], "turbidity": idx["turbidity"]}
        for z, (la_lo, la_hi, lo_lo, lo_hi) in ZONES.items():
            zmask = water & (lat >= la_lo) & (lat < la_hi) & (lon >= lo_lo) & (lon < lo_hi)
            n = int(zmask.sum())
            if n < 20:
                continue
            row = {"water_body": z, "fecha": fecha, "n_water_px": n}
            for f in SPEC:
                row[f] = float(np.median(feat2d[f][zmask]))
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["water_body", "fecha"]).reset_index(drop=True)
    df.to_csv(Z_SCENE, index=False)
    print(f"  -> {Z_SCENE} ({len(df)} filas; {df.water_body.nunique()} zonas)")
    return df


def ar_features(tb, t0):
    """Autorregresivo causal (<= t0) del target de la zona/cuerpo."""
    past = tb[tb["fecha"] <= t0]
    if past.empty:
        return None
    chl0 = past.iloc[-1]["chl_ugl"]
    def near(days):
        w = past[(past["fecha"] >= t0 - pd.Timedelta(days=days + 2)) & (past["fecha"] <= t0 - pd.Timedelta(days=days - 2))]
        return w["chl_ugl"].mean() if len(w) else chl0
    roll7 = past[past["fecha"] >= t0 - pd.Timedelta(days=7)]["chl_ugl"].mean()
    l3, l7 = near(3), near(7)
    return {"log_chl_t0": np.log1p(max(chl0, 0)), "chl_lag3": l3, "chl_lag7": l7,
            "chl_roll7": roll7, "chl_trend7": chl0 - l7}


def build_pairs(scene, target):
    """Empareja escena(t0) con target(t0+h) + autorregresivo, por cuerpo/zona."""
    tgt_by = {wb: g.sort_values("fecha").reset_index(drop=True) for wb, g in target.groupby("water_body")}
    rows = []
    for _, s in scene.iterrows():
        wb, t0 = s["water_body"], s["fecha"]
        if wb not in tgt_by:
            continue
        tb = tgt_by[wb]
        ar = ar_features(tb, t0)
        if ar is None:
            continue
        for h in HORIZONS:
            lo, hi = C.HORIZON_TOLERANCE[h]
            win = tb[(tb["fecha"] >= t0 + pd.Timedelta(days=lo)) & (tb["fecha"] <= t0 + pd.Timedelta(days=hi))]
            if win.empty:
                continue
            best = win.iloc[(win["fecha"] - (t0 + pd.Timedelta(days=h))).abs().argmin()]
            row = {"water_body": wb, "horizon": h, "fecha_t0": t0, "fecha_target": best["fecha"],
                   "chl_target": best["chl_ugl"], "log_chl_target": np.log1p(max(best["chl_ugl"], 0))}
            for f in SPEC:
                row[f] = s[f]
            row.update(ar)
            rows.append(row)
    return pd.DataFrame(rows)


def _split_eval(d):
    """Walk-forward temporal por cuerpo/zona; devuelve y, yhat, persist (log) concatenados."""
    ys, yh, yp, keys = [], [], [], []
    for wb, g in d.groupby("water_body"):
        g = g.sort_values("fecha_t0")
        if len(g) < 40:
            continue
        cut = g["fecha_t0"].quantile(0.7)
        tr, te = g[g.fecha_t0 <= cut], g[g.fecha_t0 > cut]
        if len(te) < 8 or len(tr) < 25:
            continue
        m = _model().fit(tr[FEATS], tr["log_chl_target"])
        ys.append(te["log_chl_target"].values); yh.append(m.predict(te[FEATS]))
        yp.append(te["log_chl_t0"].values)
        keys.append(te.assign(yhat=m.predict(te[FEATS]))[["water_body", "fecha_target", "yhat", "log_chl_target"]])
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(yp), pd.concat(keys, ignore_index=True)


def _metrics(y, yhat, yper):
    rmse = np.sqrt(mean_squared_error(y, yhat)); rp = np.sqrt(mean_squared_error(y, yper))
    r2 = r2_score(y, yhat) if np.var(y) > 0 else np.nan
    return rmse, r2, 1 - rmse / rp if rp > 0 else np.nan


def main():
    print("EXPERIMENTO: zonificacion de Okeechobee (4 cuadrantes) vs cuerpo entero\n")
    print("[1/4] Target VIIRS por zona (ERDDAP)..."); ztgt = fetch_zoned_target()
    print("[2/4] Predictor espectral por zona (rasters)..."); zscene = build_zoned_scene()

    # baseline cuerpo entero (mismo feature set, mismo codigo)
    body_scene = pd.read_csv(os.path.join(C.DIR_STATE, "scene_state.csv"), parse_dates=["fecha"])
    body_scene = body_scene[body_scene.water_body == "okeechobee"]
    body_tgt = pd.read_csv(os.path.join(C.DIR_OUT, "targets", "combined_target.csv"), parse_dates=["fecha"])
    body_tgt = body_tgt[body_tgt.water_body == "okeechobee"][["water_body", "fecha", "chl_ugl"]].copy()
    body_tgt["fecha"] = pd.to_datetime(body_tgt["fecha"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()

    print("[3/4] Construyendo pares (cuerpo y zonas)...")
    pb = build_pairs(body_scene, body_tgt)
    pz = build_pairs(zscene, ztgt)

    print("[4/4] Evaluando por horizonte (walk-forward)\n")
    print(f"{'h':>3} | {'CUERPO RMSE/R2/skill':>26} | {'ZONAS RMSE/R2/skill':>26} | {'ZONAS->CUERPO RMSE/R2/skill':>28}")
    print("-" * 95)
    summary = []
    for h in HORIZONS:
        rb = _split_eval(pb[pb.horizon == h])
        rz = _split_eval(pz[pz.horizon == h])
        line = f"{h:>3} |"
        rec = {"horizon": h}
        if rb:
            y, yh, yp, _ = rb; m = _metrics(y, yh, yp)
            line += f" {m[0]:.3f}/{m[1]:+.3f}/{m[2]:+.3f} |"
            rec.update(body_rmse=round(m[0], 3), body_r2=round(m[1], 3), body_skill=round(m[2], 3))
        else:
            line += f" {'sin datos':>26} |"
        if rz:
            y, yh, yp, kz = rz; m = _metrics(y, yh, yp)
            line += f" {m[0]:.3f}/{m[1]:+.3f}/{m[2]:+.3f} |"
            rec.update(zone_rmse=round(m[0], 3), zone_r2=round(m[1], 3), zone_skill=round(m[2], 3))
            # ZONAS->CUERPO: promediar predicciones de zona por fecha_target y comparar al target del cuerpo
            if rb:
                agg = kz.copy()
                agg["chl_pred"] = np.expm1(agg["yhat"]); agg["chl_real_zone"] = np.expm1(agg["log_chl_target"])
                byd = agg.groupby("fecha_target").agg(chl_pred=("chl_pred", "mean")).reset_index()
                bt = body_tgt.rename(columns={"fecha": "fecha_target", "chl_ugl": "chl_body"})
                mm = byd.merge(bt[["fecha_target", "chl_body"]], on="fecha_target", how="inner").dropna()
                if len(mm) > 5:
                    yb = np.log1p(mm["chl_body"]); ybh = np.log1p(mm["chl_pred"].clip(lower=0))
                    # persistencia del cuerpo en esas fechas objetivo (target previo) — proxy: media
                    rmse = np.sqrt(mean_squared_error(yb, ybh)); r2 = r2_score(yb, ybh) if np.var(yb) > 0 else np.nan
                    line += f" {rmse:.3f}/{r2:+.3f}/ (n={len(mm)})"
                    rec.update(z2body_rmse=round(rmse, 3), z2body_r2=round(r2, 3), z2body_n=len(mm))
        else:
            line += f" {'sin datos':>26} |"
        print(line)
        summary.append(rec)
    out = os.path.join(OUTDIR, "zonify_results.csv")
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\n-> {out}")
    print("\nLectura: si ZONAS mejora R2/skill vs CUERPO, la zonificacion ayuda a predecir mejor cada")
    print("region. Si ZONAS->CUERPO (re-agregado) bate a CUERPO contra el MISMO target -> ademas")
    print("mejora la prediccion a nivel de lago. Si no mejora -> el cuello no es la granularidad.")


if __name__ == "__main__":
    main()
