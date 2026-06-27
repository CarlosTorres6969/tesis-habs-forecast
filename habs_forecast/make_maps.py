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

# --- Mascara de agua ESTRICTA solo para los MAPAS (no toca config ni el modelo) ---
# El umbral del pipeline (NDWI>-0.5) es muy permisivo y en escenas con neblina pinta
# tierra como "agua". Para VISUALIZAR exigimos agua inequivoca (NDWI claramente
# positivo) y descartamos pixeles brillantes (nube/neblina). Luego se limpian los
# blobs dispersos por componentes conectados (conserva el cuerpo y sus brazos).
NDWI_WATER = 0.0          # agua abierta tiene NDWI > 0 (vs -0.5 del enmascarado del modelo)
NDVI_LAND  = 0.20         # excluye vegetacion terrestre con mas margen
BRIGHT_MAX = 0.25         # reflectancia visible media: por encima ~ nube/neblina/nata


def _strict_water(B2, B3, B4, B5, B8, eps=1e-10):
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    bright = (B2 + B3 + B4) / 3.0
    valid = (B2 > 0) & (B3 > 0) & (B4 > 0) & (B5 > 0) & (B8 > 0)
    return valid & (ndwi > NDWI_WATER) & (ndvi < NDVI_LAND) & (bright < BRIGHT_MAX)


def _clean_mask(water, min_frac=0.02, min_abs=20):
    """Quita blobs dispersos: conserva componentes >= min_frac del mayor (y >= min_abs).
    Para Cajon (embalse ramificado) esto mantiene los brazos grandes y borra el ruido."""
    from scipy import ndimage
    lab, n = ndimage.label(water)
    if n == 0:
        return water
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0                                   # fondo
    thr = max(min_abs, int(min_frac * sizes.max()))
    keep = np.where(sizes >= thr)[0]
    return np.isin(lab, keep)


def _scene_pixels(path):
    """Lee la escena y devuelve features espectrales por pixel de agua + mascara 2D."""
    with rasterio.open(path) as ds:
        arr = ds.read().astype("float32")          # (5, H, W)
    if arr.shape[0] < 5:
        return None
    B2, B3, B4, B5, B8 = arr[0], arr[1], arr[2], arr[3], arr[4]
    if np.nanmax(arr) > C.BAND_SCALE_THRESHOLD:
        B2, B3, B4, B5, B8 = (b / 10000.0 for b in (B2, B3, B4, B5, B8))
    water = _clean_mask(_strict_water(B2, B3, B4, B5, B8))
    idx = C.spectral_indices(B2, B3, B4, B5, B8)
    feats2d = {"B2": B2, "B3": B3, "B4": B4, "B5": B5, "B8": B8,
               "NDCI": idx["NDCI"], "CI_red": idx["CI_red"],
               "FAI": idx["FAI"], "turbidity": idx["turbidity"]}
    return feats2d, water


def _clear_water_score(path, D=12):
    """Puntua una escena por COBERTURA de agua LIMPIA (lectura decimada, rapida).
    Penaliza escenas recortadas (pocos pixeles validos) y nubes (agua brillante).
    Devuelve el nº de pixeles de agua oscura/limpia escalado a resolucion nativa,
    de modo que a igual area fisica gana la de mayor resolucion (S2 > Landsat)."""
    try:
        with rasterio.open(path) as ds:
            h0, w0 = ds.height, ds.width
            oh, ow = max(1, h0 // D), max(1, w0 // D)
            arr = ds.read(out_shape=(ds.count, oh, ow)).astype("float32")
    except Exception:
        return -1.0
    if arr.shape[0] < 5:
        return -1.0
    B2, B3, B4, B5, B8 = arr[0], arr[1], arr[2], arr[3], arr[4]
    if np.nanmax(arr) > C.BAND_SCALE_THRESHOLD:
        B2, B3, B4, B5, B8 = (b / 10000.0 for b in (B2, B3, B4, B5, B8))
    clear = _clean_mask(_strict_water(B2, B3, B4, B5, B8), min_abs=3)  # agua limpia, sin ruido
    scale = (h0 * w0) / float(oh * ow)           # a "pixeles full equivalentes"
    return float(clear.sum()) * scale


def _best_scene(tifs):
    """Elige la mejor escena: maxima cobertura de agua limpia (no la mas reciente)."""
    best, best_score = None, -1.0
    for p in tifs:
        s = _clear_water_score(p)
        if s > best_score:
            best, best_score = p, s
    return best


def build_map_figure(wb, h, path, t0, res=None):
    """Construye la figura de 2 paneles (1: satelital real; 2: biomasa algal prevista a +h d)
    para una escena Sentinel-2 dada. REUTILIZADA por make_map (CLI -> PNG) y por app.py
    (Streamlit -> st.pyplot); NO guarda ni cierra la figura (decide el llamador).
      path : raster Sentinel-2 de 5 bandas (B2,B3,B4,B5,B8).
      t0   : fecha de contexto para las features NO espectrales (broadcast); puede ser None.
      res  : recursos precargados (cache de Streamlit) opcionales; si None, lee de disco.
    Devuelve (fig, stats). Lanza ValueError con mensaje claro si la escena no sirve."""
    group = GROUP[wb]
    sp = _scene_pixels(path)
    if sp is None:
        raise ValueError("La escena no tiene 5 bandas validas (se requieren B2,B3,B4,B5,B8).")
    feats2d, water = sp
    H, W = water.shape
    nwater = int(water.sum())
    if nwater < 50:
        raise ValueError("La escena tiene muy pocos pixeles de agua validos para analizar.")

    # features no-espectrales del cuerpo en t0 (broadcast a todos los pixeles)
    built = build_features(wb, t0) if t0 is not None else None
    body_row = built[0].iloc[0] if built is not None else pd.Series(dtype=float)

    bundle = res["bundles"][(group, h)] if res else joblib.load(os.path.join(MODELS, f"{group}_h{h}.pkl"))
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

    thr = (res["thr_body"] if res else joblib.load(os.path.join(MODELS, "thr_body.pkl"))).get(wb, 10.0)

    # --- fondo satelital color verdadero (RGB = B4,B3,B2) con realce por percentiles ---
    rgb = np.dstack([feats2d["B4"], feats2d["B3"], feats2d["B2"]]).astype("float32")
    finite = np.isfinite(rgb).all(axis=2) & (rgb.sum(axis=2) > 0)
    rgbn = np.zeros_like(rgb)
    for k in range(3):
        ch = rgb[:, :, k]
        lo, hi = np.nanpercentile(ch[finite], 2), np.nanpercentile(ch[finite], 98)
        rgbn[:, :, k] = np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8   # gamma
    rgbn[~finite] = 0.0

    # --- recorte: enfoca el cuerpo de agua y elimina los bordes negros sin dato ---
    rows, cols = np.any(water, axis=1), np.any(water, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        mr = max(int(0.08 * (r1 - r0)), 8)        # margen ~8% para dar contexto de orilla
        mc = max(int(0.08 * (c1 - c0)), 8)
        r0, r1 = max(r0 - mr, 0), min(r1 + mr + 1, H)
        c0, c1 = max(c0 - mc, 0), min(c1 + mc + 1, W)
        rgbn = rgbn[r0:r1, c0:c1]
        grid = grid[r0:r1, c0:c1]
        water = water[r0:r1, c0:c1]

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    chl_ma = np.ma.masked_invalid(grid)                    # clorofila solo en agua
    # color anclado al p98 de la escena -> resalta la variacion espacial DENTRO del cuerpo
    # (el "donde hay mas biomasa"); el riesgo absoluto se marca con el contorno rojo (>= thr).
    vmax = float(np.nanpercentile(grid, 98)) or 1.0
    pct_alert = float(np.nanmean(grid >= thr) * 100)
    chlmean = float(np.nanmean(grid))
    waterf = water.astype("float32")
    riskf = np.where(np.isfinite(grid) & (grid >= thr), 1.0, 0.0)
    # tierra en GRIS (luminancia) para separar claramente agua (color) de terreno
    gray = 0.299 * rgbn[:, :, 0] + 0.587 * rgbn[:, :, 1] + 0.114 * rgbn[:, :, 2]
    base_gray = np.dstack([gray, gray, gray])

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
    # Panel 1: contexto satelital real + contorno del cuerpo de agua
    ax[0].imshow(rgbn)
    ax[0].contour(waterf, levels=[0.5], colors="cyan", linewidths=1.0)
    ax[0].set_title("1) Imagen satelital real\nlinea cian = borde del cuerpo de agua analizado", fontsize=11)

    # Panel 2: tierra en gris, agua coloreada por biomasa, contorno rojo = zona de riesgo
    ax[1].imshow(base_gray)
    im = ax[1].imshow(chl_ma, cmap="turbo", vmin=0, vmax=vmax)
    if riskf.sum() > 0:
        ax[1].contour(riskf, levels=[0.5], colors="red", linewidths=1.6)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label("Clorofila-a prevista (ug/L) — biomasa algal", fontsize=9)
    ax[1].set_title(f"2) Donde se espera mas biomasa algal (a +{h} dias)\n"
                    f"tierra = gris  ·  agua = color (azul bajo -> rojo alto)", fontsize=11)
    leg = [Patch(facecolor="0.6", label="Tierra (gris, fuera del analisis)"),
           Patch(facecolor="#2b3ff5", label="Agua: biomasa BAJA"),
           Patch(facecolor="#d62718", label="Agua: biomasa ALTA (posible floracion)"),
           Line2D([0], [0], color="red", lw=2, label=f"Zona de RIESGO (>= {thr:.0f} ug/L)")]
    ax[1].legend(handles=leg, loc="lower left", fontsize=8, framealpha=0.92)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{wb.upper()} ({'lago' if group=='freshwater' else 'costa'}) — pronostico de riesgo de "
                 f"biomasa algal a +{h} dias  |  escena {t0.date() if t0 is not None else '?'}\n"
                 f"clorofila-a media = {chlmean:.1f} ug/L   ·   area en riesgo = {pct_alert:.0f}%   "
                 f"(herramienta de alerta; NO confirma toxicidad, requiere verificacion de campo)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stats = {"chl_mean": float(chlmean), "pct_alert": float(pct_alert), "thr": float(thr),
             "t0": t0, "n_water_px": int(nwater), "h": int(h), "group": group}
    return fig, stats


def make_map(wb, h=7, scene=None):
    """CLI: elige la escena (mejor o por fecha), construye la figura y la guarda como PNG."""
    folder = KEY2FOLDER[wb]
    tifs = sorted(glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True))
    if not tifs:
        print(f"{wb}: sin rasters."); return
    if scene:                                        # fecha concreta pedida (YYYY-MM-DD)
        match = [p for p in tifs if scene in os.path.basename(p)]
        path = match[-1] if match else _best_scene(tifs)
    else:
        path = _best_scene(tifs)                     # MEJOR escena (cobertura de agua limpia)
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    t0 = pd.Timestamp(m.group(1)) if m else None
    try:
        fig, stats = build_map_figure(wb, h, path, t0)
    except ValueError as e:
        print(f"{wb}: {e}"); return
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, f"mapa_{wb}_h{h}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {wb} +{h}d -> {out} | chl-a media={stats['chl_mean']:.1f} ug/L "
          f"| area de riesgo/biomasa alta={stats['pct_alert']:.0f}%")


def main():
    # h=1 por defecto: a corto plazo el modelo usa la senal espectral por pixel -> mapa
    # espacialmente informativo. A +7d el modelo es body-level (sin espectral) -> mapa plano.
    args = sys.argv[1:]
    wb = args[0] if args else "okeechobee"
    h = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1
    # 3er argumento opcional: fecha de escena YYYY-MM-DD (si no, se elige la mejor)
    scene = next((a for a in args[1:] if "-" in a and len(a) == 10), None)
    make_map(wb, h, scene=scene)


if __name__ == "__main__":
    main()
