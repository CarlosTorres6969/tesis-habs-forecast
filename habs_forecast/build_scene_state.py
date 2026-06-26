"""
build_scene_state.py — MODULO A: estado del bloom por ESCENA desde los rasters Sentinel-2.

Lee cada GeoTIFF (5 bandas B2,B3,B4,B5,B8), enmascara agua (NDWI/NDVI), y agrega los
pixeles de agua validos a una MEDIANA por escena -> una fila de PREDICTORES por (region, fecha).
Esta es la fuente real de predictores X(t0); reemplaza el reuso de tablas viejas.

Salida: artifacts/state_series/scene_state.csv
    region, water_body, group, fecha, n_water_px, B2..B8, NDCI, CI_red, FAI, turbidity
"""
from __future__ import annotations
import os, glob, re
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

import config as C

OUT = os.path.join(C.DIR_STATE, "scene_state.csv")
# carpeta en imagenes/ -> (water_body key, group). Reusa config.REGIONS.
FOLDER2BODY = {folder: (meta["key"], meta["group"]) for folder, meta in C.REGIONS.items()}
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
DOWNSCALE = 4          # leer a 1/4 de resolucion (suficiente para mediana de escena)


def _scene_features(path: str):
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        oh, ow = max(1, h // DOWNSCALE), max(1, w // DOWNSCALE)
        arr = ds.read(out_shape=(ds.count, oh, ow),
                      resampling=Resampling.bilinear).astype("float32")
    if arr.shape[0] < 5:
        return None
    B2, B3, B4, B5, B8 = arr[0], arr[1], arr[2], arr[3], arr[4]
    # normalizar a reflectancia 0-1 si vienen escaladas (uint16 * 10000)
    if np.nanmax(arr) > C.BAND_SCALE_THRESHOLD:
        B2, B3, B4, B5, B8 = (b / 10000.0 for b in (B2, B3, B4, B5, B8))

    eps = 1e-10
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    valid = (B2 > 0) & (B3 > 0) & (B4 > 0) & (B5 > 0) & (B8 > 0)
    water = valid & (ndwi > C.NDWI_MIN) & (ndvi < C.NDVI_MAX)
    n = int(water.sum())
    if n < C.MIN_WATER_PIXELS:
        return None

    idx = C.spectral_indices(B2, B3, B4, B5, B8)
    feats = {
        "n_water_px": n,
        "B2": np.median(B2[water]), "B3": np.median(B3[water]),
        "B4": np.median(B4[water]), "B5": np.median(B5[water]),
        "B8": np.median(B8[water]),
        "NDCI": np.median(idx["NDCI"][water]),
        "CI_red": np.median(idx["CI_red"][water]),
        "FAI": np.median(idx["FAI"][water]),
        "turbidity": np.median(idx["turbidity"][water]),
    }
    return feats


def build():
    rows = []
    for folder, (body, group) in FOLDER2BODY.items():
        tifs = sorted(glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"),
                                recursive=True))
        ok = 0
        for t in tifs:
            m = DATE_RE.search(os.path.basename(t))
            if not m:
                continue
            try:
                f = _scene_features(t)
            except Exception:
                continue
            if f is None:
                continue
            f.update({"region": folder, "water_body": body, "group": group,
                      "fecha": pd.Timestamp(m.group(1))})
            rows.append(f)
            ok += 1
        print(f"  {folder:16s}: {ok}/{len(tifs)} escenas con agua valida")

    df = pd.DataFrame(rows).sort_values(["water_body", "fecha"]).reset_index(drop=True)
    os.makedirs(C.DIR_STATE, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nEstado por escena -> {OUT} ({len(df)} escenas)")
    if len(df):
        print("\n=== escenas por cuerpo y rango temporal ===")
        g = df.groupby(["group", "water_body"]).agg(
            escenas=("fecha", "size"), desde=("fecha", "min"), hasta=("fecha", "max"))
        print(g.to_string())
    return df


if __name__ == "__main__":
    build()
