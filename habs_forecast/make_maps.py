"""
make_maps.py — MAPAS ESPACIALES de clorofila-a prevista / riesgo de biomasa algal (el "donde").
NB: clorofila-a = proxy de biomasa, NO confirma floracion NOCIVA (requiere verificacion de campo).

Aplica el modelo de alerta PIXEL A PIXEL sobre una escena Sentinel-2: las features
espectrales (B2..B8, NDCI, FAI, ...) varian por pixel; las dinamicas del cuerpo
(clorofila reciente, ERA5, in-situ) se mantienen constantes (broadcast). Resultado: un
mapa de riesgo de HAB dentro del cuerpo de agua, calibrado.

Uso:  python make_maps.py okeechobee          (horizonte +7d, ultima escena)
      python make_maps.py cajon 3
Salida: artifacts/reports/mapa_{cuerpo}_h{h}.png
"""
from __future__ import annotations
import os, sys, glob, joblib
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from predict import build_features, GROUP, SPEC
from train_nn import HABNet
import torch

MODELS = C.DIR_MODELS
REPORTS = C.DIR_REPORTS
KEY2FOLDER = {meta["key"]: folder for folder, meta in C.REGIONS.items()}


def _scene_pixels(path):
    """Lee la escena y devuelve features espectrales por pixel de agua + mascara 2D."""
    with rasterio.open(path) as ds:
        arr = ds.read().astype("float32")          # (5, H, W)
    if arr.shape[0] < 5:
        return None
    B2, B3, B4, B5, B8 = arr[0], arr[1], arr[2], arr[3], arr[4]
    if np.nanmax(arr) > C.BAND_SCALE_THRESHOLD:
        B2, B3, B4, B5, B8 = (b / 10000.0 for b in (B2, B3, B4, B5, B8))
    eps = 1e-10
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    valid = (B2 > 0) & (B3 > 0) & (B4 > 0) & (B5 > 0) & (B8 > 0)
    water = valid & (ndwi > C.NDWI_MIN) & (ndvi < C.NDVI_MAX)
    idx = C.spectral_indices(B2, B3, B4, B5, B8)
    feats2d = {"B2": B2, "B3": B3, "B4": B4, "B5": B5, "B8": B8,
               "NDCI": idx["NDCI"], "CI_red": idx["CI_red"],
               "FAI": idx["FAI"], "turbidity": idx["turbidity"]}
    return feats2d, water


def make_map(wb, h=7):
    group = GROUP[wb]
    folder = KEY2FOLDER[wb]
    tifs = sorted(glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True))
    if not tifs:
        print(f"{wb}: sin rasters."); return
    path = tifs[-1]                                  # ultima escena
    sp = _scene_pixels(path)
    if sp is None:
        print(f"{wb}: escena invalida."); return
    feats2d, water = sp
    H, W = water.shape
    nwater = int(water.sum())
    if nwater < 50:
        print(f"{wb}: pocos pixeles de agua."); return

    # features no-espectrales del cuerpo en t0 (broadcast a todos los pixeles)
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    t0 = pd.Timestamp(m.group(1)) if m else None
    built = build_features(wb, t0)
    body_row = built[0].iloc[0] if built is not None else pd.Series(dtype=float)

    bundle = joblib.load(os.path.join(MODELS, f"{group}_h{h}.pkl"))
    feats = bundle["feats"]
    # matriz pixel x feature
    X = pd.DataFrame(index=np.arange(nwater), columns=feats, dtype="float32")
    for f in feats:
        if f in SPEC:
            X[f] = feats2d[f][water].astype("float32")
        else:
            X[f] = float(body_row.get(f, np.nan))

    # INTENSIDAD de clorofila por pixel (cabeza de regresion -> varia espacialmente
    # con la senal espectral). Mapea la distribucion espacial del bloom.
    chl = np.expm1(bundle["reg"].predict(X))
    chl = np.clip(chl, 0, None)
    grid = np.full((H, W), np.nan, dtype="float32")
    grid[water] = chl

    thr = joblib.load(os.path.join(MODELS, "thr_body.pkl")).get(wb, 10.0)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, f"mapa_{wb}_h{h}.png")
    vmax = float(np.nanpercentile(grid, 98)) or 1.0
    plt.figure(figsize=(9, 7))
    plt.imshow(grid, cmap="turbo", vmin=0, vmax=max(vmax, thr))
    plt.colorbar(label="Clorofila-a prevista (ug/L) ~ biomasa algal")
    pct_alert = float(np.nanmean(grid >= thr) * 100)
    plt.title(f"Pronostico +{h}d — {wb} | escena {t0.date() if t0 is not None else '?'}\n"
              f"clorofila-a media={np.nanmean(grid):.1f} ug/L | "
              f"area de riesgo / biomasa alta (>= {thr:.0f})={pct_alert:.0f}%")
    plt.axis("off"); plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"  {wb} +{h}d -> {out} | chl-a media={np.nanmean(grid):.1f} ug/L "
          f"| area de riesgo/biomasa alta={pct_alert:.0f}%")


def main():
    # h=1 por defecto: a corto plazo el modelo usa la senal espectral por pixel -> mapa
    # espacialmente informativo. A +7d el modelo es body-level (sin espectral) -> mapa plano.
    args = sys.argv[1:]
    wb = args[0] if args else "okeechobee"
    h = int(args[1]) if len(args) > 1 else 1
    make_map(wb, h)


if __name__ == "__main__":
    main()
