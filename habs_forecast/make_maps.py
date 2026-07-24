"""
make_maps.py — MAPAS ESPACIALES de clorofila-a prevista / riesgo de biomasa algal (el "donde").
NB: clorofila-a = proxy de biomasa, NO confirma floración NOCIVA (requiere verificación de campo).

El pronóstico validado se calcula con medianas de escena para el cuerpo de agua completo.
Cuando hay señal espectral, el mapa reparte ese nivel mediante un patrón espacial heurístico
de media uno. El detalle por píxel es exploratorio y no una predicción espacial validada.

Uso:  python make_maps.py okeechobee          (horizonte +7d, última escena)
      python make_maps.py cajon 3
Salida: artifacts/reports/mapa_{cuerpo}_h{h}.png
"""
from __future__ import annotations
import os, sys, glob, joblib
from functools import lru_cache
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
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

# --- Máscara de agua ESTRICTA solo para los MAPAS (no toca config ni el modelo) ---
# El umbral del pipeline (NDWI>-0.5) es muy permisivo y en escenas con neblina pinta
# tierra como "agua". Para VISUALIZAR exigimos agua inequívoca (NDWI claramente
# positivo) y descartamos pixeles brillantes (nube/neblina). Luego se limpian los
# blobs dispersos por componentes conectados (conserva el cuerpo y sus brazos).
NDWI_WATER = 0.0          # agua abierta tiene NDWI > 0 (vs -0.5 del enmascarado del modelo)
NDVI_LAND  = 0.20         # excluye vegetación terrestre con más margen
BRIGHT_MAX = 0.25         # reflectancia visible media: por encima ~ nube/neblina/nata
# Rechazo de nube/bruma PROTEGIDO POR VERDOR (solo-mapa): la nube fina que sobrevive al gate
# del pipeline se pinta como biomasa (roja). La nube/bruma es BRILLANTE y espectralmente PLANA
# (green_ratio = B3/max(B2,B4) ~ 1); la nata algal es VERDE (green_ratio > 1.1). Se excluye solo
# lo brillante-y-NO-verde -> quita nube sin borrar floraciones reales. Verificado en escenas
# nubosas: quita 0.4-4.5% (la nube) y conserva 57-92% de agua verde.
HAZE_BRIGHT = 0.13        # brillo visible medio por encima del agua limpia (p99 agua <= 0.10-0.12)
GREEN_MIN   = 1.10        # green_ratio >= esto = agua verde (nata algal) -> se PROTEGE de la máscara

# Tope de resolución de trabajo SOLO para el mapa/predicción por píxel: las escenas costeras
# enormes (Golfo de Fonseca ~24 Mpx = ~13 M px de agua) hacían la inferencia píxel-a-píxel muy
# lenta (~25-30 s/mapa; x5 en la animación -> >2 min, parecía colgada). La figura se muestra a
# 200 dpi (~1500 px por panel), así que predecir a resolución nativa era desperdicio. Se lee la
# escena DECIMADA (promediando) cuando su lado mayor supera este tope: mismo mapa visible, ~10x
# más rápido. Los cuerpos chicos (Cajón, Yojoa, Tampa en escenas pequeñas) no se tocan (factor 1).
MAP_MAX_DIM = 2400       # suficiente detalle para mapas nítidos sin inferencia a resolución nativa extrema


def _strict_water(B2, B3, B4, B5, B8, eps=1e-10):
    # Base: la MISMA máscara del pipeline (config.water_mask) -> mismo rechazo de nube/bruma
    # que las features del modelo (fuente única). Encima, estrictez extra SOLO-mapa: agua
    # inequívoca (NDWI>0), menos vegetación (NDVI<0.20), tope de brillo visible y rechazo de
    # nube fina protegido por verdor (no toca la nata algal verde).
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    bright = (B2 + B3 + B4) / 3.0
    green_ratio = B3 / (np.maximum(B2, B4) + eps)
    hazy = (bright > HAZE_BRIGHT) & (green_ratio < GREEN_MIN)   # brillante y NO verde = nube/bruma
    strict = (ndwi > NDWI_WATER) & (ndvi < NDVI_LAND) & (bright < BRIGHT_MAX) & ~hazy
    return C.water_mask(B2, B3, B4, B5, B8) & strict


def _clean_mask(water, min_frac=0.02, min_abs=20):
    """Quita blobs dispersos y reconstruye el cuerpo continuo: 1) cierre morfológico para
    unir los brazos finos del embalse separados por neblina/pixeles dudosos; 2) conserva
    componentes >= min_frac del mayor (y >= min_abs); 3) rellena huecos internos.
    Para Cajón (embalse ramificado) esto recupera el 'trayecto' del cuerpo de agua."""
    from scipy import ndimage
    st = ndimage.generate_binary_structure(2, 2)   # 8-conectividad
    water = ndimage.binary_closing(water, structure=st, iterations=2)
    lab, n = ndimage.label(water)
    if n == 0:
        return water
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0                                   # fondo
    thr = max(min_abs, int(min_frac * sizes.max()))
    keep = np.where(sizes >= thr)[0]
    mask = np.isin(lab, keep)
    return ndimage.binary_fill_holes(mask)         # cobertura continua del cuerpo


@lru_cache(maxsize=48)
def _scene_pixels_cached(path, _mtime):
    return _scene_pixels_impl(path)


_PATTERN_CACHE = {}


def _spatial_pattern_memo(group, path, t0, feats2d, water, body_row, res):
    """Cache del patrón espacial por (grupo, escena, t0): es IDÉNTICO para todos los horizontes de
    una misma escena, así la animación 1->7d no lo recomputa 4 veces (~1.4s c/u). Devuelve el mismo
    array (o None) que _spatial_pattern."""
    key = (group, path, str(t0))
    pat = _PATTERN_CACHE.get(key, False)                 # False = aún no calculado (None es válido)
    if pat is False:
        pat = _spatial_pattern(group, feats2d, water, body_row, res)
        if len(_PATTERN_CACHE) > 64:
            _PATTERN_CACHE.clear()
        _PATTERN_CACHE[key] = pat
    return pat


def _scene_pixels(path):
    """Lectura de escena CACHEADA por (ruta, mtime): la misma escena se lee UNA sola vez aunque
    build_map_figure se invoque para varios horizontes (p.ej. la animación 1->7d, que antes leía
    el raster 4 veces). Los resultados son de solo-lectura; no mutar feats2d/water."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = 0.0
    return _scene_pixels_cached(path, mt)


def _scene_pixels_impl(path):
    """Lee la escena y devuelve (features espectrales por pixel de agua, máscara 2D, factor de
    decimación). Si el lado mayor del raster supera MAP_MAX_DIM, se lee a resolución reducida
    (promediando bloques) para acelerar la inferencia por pixel sin cambiar el mapa visible ni
    los stats (que son ratios sobre el agua). 'factor' = cuántos pixeles nativos representa cada
    pixel leído (1 = sin decimar), para reescalar conteos a resolución completa."""
    with rasterio.open(path) as ds:
        if ds.count < 5:
            return None
        H0, W0 = ds.height, ds.width
        factor = max(1, int(np.ceil(max(H0, W0) / MAP_MAX_DIM)))   # decimación entera solo si es grande
        if factor > 1:
            oh, ow = max(1, H0 // factor), max(1, W0 // factor)
            arr = ds.read(out_shape=(ds.count, oh, ow),
                          resampling=Resampling.bilinear).astype("float32")
        else:
            arr = ds.read().astype("float32")      # (5, H, W)
    B2, B3, B4, B5, B8 = arr[0], arr[1], arr[2], arr[3], arr[4]
    if np.nanmax(arr) > C.BAND_SCALE_THRESHOLD:
        B2, B3, B4, B5, B8 = (b / 10000.0 for b in (B2, B3, B4, B5, B8))
    water = _clean_mask(_strict_water(B2, B3, B4, B5, B8))
    idx = C.spectral_indices(B2, B3, B4, B5, B8)
    feats2d = {"B2": B2, "B3": B3, "B4": B4, "B5": B5, "B8": B8,
               "NDCI": idx["NDCI"], "CI_red": idx["CI_red"],
               "FAI": idx["FAI"], "turbidity": idx["turbidity"]}
    return feats2d, water, factor


def _clear_water_score(path, D=12):
    """Puntúa una escena por el CUERPO DE AGUA COHERENTE más grande (lectura decimada).
    Usa el COMPONENTE CONEXO mayor (no el total de pixeles dispersos): así una escena con
    neblina —que fragmenta el agua en parches— pierde frente a una despejada donde el
    embalse forma un solo cuerpo continuo. Escala a resolución nativa, de modo que a igual
    cuerpo físico gana la de mayor resolución (S2 > Landsat)."""
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
    from scipy import ndimage
    clear = _strict_water(B2, B3, B4, B5, B8)
    clear = ndimage.binary_closing(clear, structure=ndimage.generate_binary_structure(2, 2))
    lab, n = ndimage.label(clear)
    if n == 0:
        return 0.0
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    big, tot = float(sizes.max()), float(sizes.sum())
    coh = big / tot if tot else 0.0              # coherencia: que el agua forme UN cuerpo, no parches
    scale = (h0 * w0) / float(oh * ow)           # a "pixeles full equivalentes"
    score = big * scale
    if coh < 0.40:                               # agua dispersa (neblina/nubes): penaliza fuerte
        score *= 0.05
    return score


def _best_scene(tifs):
    """Elige la mejor escena: máxima cobertura de agua limpia (no la más reciente)."""
    best, best_score = None, -1.0
    for p in tifs:
        s = _clear_water_score(p)
        if s > best_score:
            best, best_score = p, s
    return best


def _spatial_pattern(group, feats2d, water, body_row, res):
    """Patrón espacial RELATIVO (media ~1) de clorofila a partir de la señal espectral
    ACTUAL, usando la respuesta relativa de un modelo con features espectrales. Sirve para
    desagregar visualmente un pronóstico body-level, que de por sí es uniforme:
    el NIVEL lo pone el horizonte pedido y el DÓNDE lo pone la imagen de hoy. Devuelve un
    vector por pixel de agua (media ~1) o None si no hay un modelo espectral disponible."""
    def as_visible_gradient(signal):
        """Convierte una señal relativa en un gradiente robusto y controlado.

        Recorta extremos al P5-P95 y limita el multiplicador aproximadamente a
        0.6-1.4 antes de renormalizarlo. Así las nubes residuales no dominan el mapa
        y el valor medio sigue siendo exactamente el pronóstico body-level.
        """
        values = np.asarray(signal, dtype="float32")
        finite = values[np.isfinite(values)]
        if len(finite) < 50:
            return None
        lo, hi = np.nanpercentile(finite, [5, 95])
        if not np.isfinite(lo + hi) or hi - lo <= 1e-8:
            return None
        scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        pattern = 0.6 + 0.8 * scaled
        mean = float(np.nanmean(pattern))
        return (pattern / mean).astype("float32") if mean > 0 else None

    n = int(water.sum())
    # La selección de familias cambia al reentrenar. Se busca cualquier horizonte
    # que conserve señal espectral, en vez de asumir que siempre será +3/+5 días.
    for hp in (3, 5, 7, 1):
        if res:
            bundle = res["bundles"].get((group, hp))
        else:
            p = os.path.join(MODELS, f"{group}_h{hp}.pkl")
            bundle = joblib.load(p) if os.path.exists(p) else None
        if bundle is None or not any(f in SPEC for f in bundle["feats"]):
            continue
        feats = bundle["feats"]
        X = pd.DataFrame(index=np.arange(n), columns=feats, dtype="float32")
        for f in feats:
            X[f] = feats2d[f][water].astype("float32") if f in SPEC else float(body_row.get(f, np.nan))
        pat = as_visible_gradient(np.clip(np.expm1(bundle["reg"].predict(X)), 0, None))
        if pat is not None:
            return pat

    # Fallback visual: combinación robusta de índices ópticos de biomasa/turbidez.
    # Sigue siendo HEURÍSTICO (así se etiqueta), pero evita mapas planos cuando la
    # selección temporal valida solo variables autorregresivas para un horizonte.
    optical = []
    for feature in ("NDCI", "FAI", "CI_red", "turbidity"):
        if feature not in feats2d:
            continue
        values = np.asarray(feats2d[feature][water], dtype="float32")
        finite = values[np.isfinite(values)]
        if len(finite) < 50:
            continue
        lo, hi = np.nanpercentile(finite, [5, 95])
        if np.isfinite(lo + hi) and hi - lo > 1e-8:
            optical.append(np.clip((values - lo) / (hi - lo), 0.0, 1.0))
    if optical:
        return as_visible_gradient(np.nanmean(np.vstack(optical), axis=0))
    return None


def build_map_figure(wb, h, path, t0, res=None, gradient_focus=False, focus_water=False,
                     nowcast_level=None, hq=False, color_limits=None):
    """Construye la figura de 2 paneles (1: satelital real; 2: biomasa algal prevista a +h d)
    para una escena Sentinel-2 dada. REUTILIZADA por make_map (CLI -> PNG) y por app.py
    (Streamlit -> st.pyplot); NO guarda ni cierra la figura (decide el llamador).
      path : raster Sentinel-2 de 5 bandas (B2,B3,B4,B5,B8).
      t0   : fecha de contexto para las features NO espectrales (broadcast); puede ser None.
      res  : recursos precargados (cache de Streamlit) opcionales; si None, lee de disco.
      color_limits : (min, max) opcional compartido entre horizontes. Evita que cada
        cuadro autoajuste sus colores y parezca idéntico aunque cambie la intensidad.
      gradient_focus : si True, prioriza la LEGIBILIDAD del gradiente espacial de clorofila
        (suaviza el campo para quitar ruido sal-y-pimienta y aligera los contornos de umbral
        para que no tapen el color). Solo afecta la VISUALIZACIÓN, no los stats ni el modelo.
        Pensado para figuras de validación en cuerpos casi totalmente en floración (Cajón,
        Fonseca), donde los contornos densos pintaban todo de rojo. Default False => sin cambios.
    Devuelve (fig, stats). Lanza ValueError con mensaje claro si la escena no sirve."""
    group = GROUP[wb]
    sp = _scene_pixels(path)
    if sp is None:
        raise ValueError("La escena no tiene 5 bandas válidas (se requieren B2,B3,B4,B5,B8).")
    feats2d, water, decim = sp
    H, W = water.shape
    nwater = int(water.sum())
    if nwater < 50:
        raise ValueError("La escena tiene muy pocos píxeles de agua válidos para analizar.")
    n_water_full = int(nwater * decim * decim)     # conteo equivalente a resolución nativa (guardas)

    # features no-espectrales del cuerpo en t0 (broadcast a todos los pixeles)
    built = build_features(wb, t0) if t0 is not None else None
    body_row = built[0].iloc[0] if built is not None else pd.Series(dtype=float)

    bundle = res["bundles"][(group, h)] if res else joblib.load(os.path.join(MODELS, f"{group}_h{h}.pkl"))
    feats = bundle["feats"]
    # El modelo se validó con medianas de escena a nivel del cuerpo de agua.
    # Se ejecuta una sola vez con esos agregados, nunca como regresor por píxel.
    X = pd.DataFrame(index=[0], columns=feats, dtype="float32")
    for f in feats:
        if f in SPEC:
            X.loc[0, f] = float(np.nanmedian(feats2d[f][water]))
        else:
            X.loc[0, f] = float(body_row.get(f, np.nan))
    body_level = float(np.clip(np.expm1(bundle["reg"].predict(X))[0], 0, None))
    chl = np.full(nwater, body_level, dtype="float32")

    # La textura es una desagregación heurística del nivel body-level. Sirve para
    # visualización exploratoria y no se presenta como pronóstico validado por píxel.
    spatial_mode = "uniform"
    pattern = _spatial_pattern_memo(group, path, t0, feats2d, water, body_row, res)
    if pattern is not None:
        chl = body_level * pattern
        spatial_mode = "heuristic"
    # Cuadro "HOY" (observado): en vez de la predicción del modelo, se pinta el nivel de clorofila
    # OBSERVADO del cuerpo (fc['chl0'], de la serie satelital -> NO es la óptica S2 con fuga)
    # repartido por el patrón espacial actual. Solo se activa si el llamador pasa nowcast_level.
    if nowcast_level is not None:
        pat = _spatial_pattern_memo(group, path, t0, feats2d, water, body_row, res)
        chl = float(nowcast_level) * (pat if pat is not None
                                      else np.ones(nwater, dtype="float32"))
        spatial_mode = "nowcast"
    grid = np.full((H, W), np.nan, dtype="float32")
    grid[water] = chl
    grid_full = grid                     # referencia SIN recortar: los stats se calculan sobre TODA
                                         # el agua del cuerpo (el recorte de abajo solo afecta el DISPLAY,
                                         # no los valores de retorno de build_map_figure).

    thr_rel = (res["thr_body"] if res else joblib.load(os.path.join(MODELS, "thr_body.pkl"))).get(wb, 10.0)
    thr = C.alert_threshold_ugl(thr_rel)               # FLORACIÓN: p85 acotado al nivel biológico (<=24)
    thr_elev = C.elevated_threshold_ugl(thr)           # BIOMASA ELEVADA: banda de aviso (< floración)

    # --- fondo satelital color verdadero (RGB = B4,B3,B2) con realce por percentiles ---
    # Se calculan DOS estiramientos distintos (a resolución completa, ANTES del recorte):
    #  * rgb_water: percentiles p2-p98 SOLO sobre pixeles de agua -> el panel 1 deja de comprimir
    #    el agua a negro (la tierra brillante ya no domina el estiramiento). Fallback a toda la
    #    escena si hay <50 px de agua.
    #  * rgb_scene: percentiles sobre TODA la escena -> del que se deriva el GRIS de la tierra del
    #    panel 2 (si se usara rgb_water, la tierra se saturaría a blanco).
    rgb = np.dstack([feats2d["B4"], feats2d["B3"], feats2d["B2"]]).astype("float32")
    finite = np.isfinite(rgb).all(axis=2) & (rgb.sum(axis=2) > 0)
    rgb_water = np.zeros_like(rgb)                 # panel 1 (color verdadero, contraste en agua)
    rgb_scene = np.zeros_like(rgb)                 # panel 2 (gris de tierra, contraste de escena)
    for k in range(3):
        ch = rgb[:, :, k]
        ref = ch[water & finite]                   # estiramiento del panel 1 anclado al agua
        if ref.size < 50:
            ref = ch[finite]
        lo, hi = np.nanpercentile(ref, 2), np.nanpercentile(ref, 98)
        rgb_water[:, :, k] = np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8   # gamma
        slo, shi = np.nanpercentile(ch[finite], 2), np.nanpercentile(ch[finite], 98)
        rgb_scene[:, :, k] = np.clip((ch - slo) / (shi - slo + 1e-9), 0, 1) ** 0.8
    # Sin cobertura/nodata en gris muy claro: evita grandes bloques negros que hacían
    # que la lámina pareciera dañada y mantiene el foco visual sobre el agua.
    rgb_water[~finite] = 0.93
    rgb_scene[~finite] = 0.93

    # --- ENCUADRE: centra el recorte en el CUERPO DE AGUA COHERENTE MÁS GRANDE ---
    # El bbox de TODA el agua detectada se corre por parches espurios (nubes/sombras mal
    # clasificadas) y deja el cuerpo real fuera de cuadro. En su lugar se etiquetan los
    # componentes conexos y se encuadra el MAYOR (el cuerpo físico). Los demás pixeles de agua
    # que caigan dentro del cuadro se conservan visibles; solo el ENCUADRE lo fija el mayor.
    # focus_water: solo agranda el margen (más contexto de orilla) en cuerpos angostos/ramificados.
    from scipy import ndimage
    body_note = None
    lab, n = ndimage.label(water)
    frame = None
    if n > 0:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        biggest = int(sizes.argmax())
        if sizes[biggest] >= 50:                   # cuerpo coherente mínimo
            frame = (lab == biggest)
    if frame is not None:
        rows, cols = np.any(frame, axis=1), np.any(frame, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        fm = 0.12 if focus_water else 0.08         # margen mayor al enfocar para dar contexto
        mr = max(int(fm * (r1 - r0)), 8)           # margen para contexto de orilla
        mc = max(int(fm * (c1 - c0)), 8)
        r0, r1 = r0 - mr, r1 + mr + 1
        c0, c1 = c0 - mc, c1 + mc + 1
        # mínimo de encuadre: un cuerpo chico frente a la escena queda con poco contexto; se
        # expande el cuadro (centrado en el cuerpo) hasta un tamaño mínimo para ver la orilla.
        min_h = int(min(H, max(72, 0.18 * H)))
        min_w = int(min(W, max(72, 0.18 * W)))
        if (r1 - r0) < min_h:
            cr = (r0 + r1) // 2; r0, r1 = cr - min_h // 2, cr + min_h // 2
        if (c1 - c0) < min_w:
            cc = (c0 + c1) // 2; c0, c1 = cc - min_w // 2, cc + min_w // 2
        r0, r1 = max(r0, 0), min(r1, H)
        c0, c1 = max(c0, 0), min(c1, W)
        rgb_water = rgb_water[r0:r1, c0:c1]
        rgb_scene = rgb_scene[r0:r1, c0:c1]
        grid = grid[r0:r1, c0:c1]
        water = water[r0:r1, c0:c1]
    else:
        # fallback robusto: sin cuerpo de agua coherente NO se recorta (evita un cuadro roto);
        # se muestra la escena completa y se avisa en el subtítulo del panel 1.
        body_note = "cuerpo de agua no detectado con claridad en esta escena"

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    wv = grid_full[np.isfinite(grid_full)]                  # valores SOLO en agua (TODO el cuerpo, sin recorte)
    pct_alert = float((wv >= thr).mean() * 100) if wv.size else 0.0       # % AGUA en FLORACIÓN (>= thr)
    pct_elev  = float((wv >= thr_elev).mean() * 100) if wv.size else 0.0  # % AGUA con biomasa elevada
    chlmean = float(np.nanmean(grid_full))
    nivel_body = C.biomass_level(chlmean, thr, thr_elev)  # nivel global del cuerpo (según la media)
    # GRID DE DISPLAY: en gradient_focus se suaviza (nan-aware) para quitar el ruido sal-y-pimienta
    # y que el gradiente espacial se lea limpio. NO altera los stats (calculados sobre 'grid' crudo).
    grid_disp = grid
    if gradient_focus or hq:
        from scipy import ndimage
        sig = 1.15 if hq else 0.9                # limpia ruido sin borrar estructuras espaciales
        m = np.isfinite(grid).astype("float32")
        filled = np.where(m > 0, grid, 0.0).astype("float32")
        num = ndimage.gaussian_filter(filled, sigma=sig)
        den = ndimage.gaussian_filter(m, sigma=sig)
        sm = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)
        if hq:
            # Enfoque multiescala: recupera bordes amplios del gradiente después del
            # antirruido, sin reintroducir el moteado fino del raster original.
            smooth_filled = np.where(m > 0, sm, 0.0).astype("float32")
            broad_num = ndimage.gaussian_filter(smooth_filled, sigma=2.4)
            broad_den = ndimage.gaussian_filter(m, sigma=2.4)
            broad = np.divide(broad_num, broad_den, out=np.full_like(broad_num, np.nan),
                              where=broad_den > 1e-6)
            sharpened = sm + 0.38 * (sm - broad)
            raw = grid[np.isfinite(grid)]
            if raw.size:
                low, high = np.nanpercentile(raw, [0.5, 99.5])
                sm = np.clip(sharpened, low, high)
        grid_disp = np.where(np.isfinite(grid), sm, np.nan).astype("float32")
    chl_ma = np.ma.masked_invalid(grid_disp)               # clorofila (display) solo en agua
    # COLOR RELATIVO a la escena (p2-p98): resalta el GRADIENTE espacial de clorofila dentro
    # del cuerpo (azul=menos -> rojo=más), como las imágenes h3. La barra de color muestra los
    # VALORES reales en ug/L (los mismos que se mapean en la página); el RIESGO absoluto se marca
    # con los contornos (biomasa elevada / floración) y con el % del título.
    dv = grid_disp[np.isfinite(grid_disp)]
    if color_limits is not None:
        vmin, vmax = map(float, color_limits)
    else:
        vmin = float(np.nanpercentile(dv, 2)) if dv.size else 0.0
        vmax = float(np.nanpercentile(dv, 98)) if dv.size else 1.0
    if not (vmax > vmin):                                   # escena casi plana: evita escala degenerada
        vmax = vmin + 1.0
    waterf = water.astype("float32")
    # contornos sobre el grid de display (suaves en gradient_focus -> no fragmentan)
    gcont = grid_disp if gradient_focus else grid
    riskf = np.where(np.isfinite(gcont) & (gcont >= thr), 1.0, 0.0)
    elevf = np.where(np.isfinite(gcont) & (gcont >= thr_elev), 1.0, 0.0)
    # tierra en GRIS (luminancia) para separar claramente agua (color) de terreno.
    # OJO: se deriva de rgb_scene (estirado sobre TODA la escena), no de rgb_water, para que la
    # tierra brillante no se sature a blanco.
    gray = 0.299 * rgb_scene[:, :, 0] + 0.587 * rgb_scene[:, :, 1] + 0.114 * rgb_scene[:, :, 2]
    base_gray = np.dstack([gray, gray, gray])

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2), dpi=240 if hq else 200)
    # Interpolación de alta calidad SOLO visual (hq): suaviza el pixelado de rásters pequeños
    # al mostrarlos grandes. lanczos = nítido para la foto RGB; bilinear = transición de color
    # suave para el campo de biomasa. Sin hq, se conserva el render original (antialiased).
    _interp_rgb = "lanczos" if hq else None
    # Panel 1: contexto satelital real + contorno del cuerpo de agua
    ax[0].imshow(rgb_water, **({"interpolation": _interp_rgb} if _interp_rgb else {}))
    ax[0].contour(waterf, levels=[0.5], colors="cyan", linewidths=1.0)
    p1_sub = "línea cian = borde del cuerpo de agua analizado"
    if body_note:
        p1_sub += f"\n({body_note})"
    ax[0].set_title(f"1) Imagen satelital real\n{p1_sub}", fontsize=12)

    # Panel 2: tierra en gris, agua coloreada por biomasa, contorno rojo = zona de riesgo
    ax[1].imshow(base_gray, **({"interpolation": _interp_rgb} if _interp_rgb else {}))
    _interp = "lanczos" if hq else None        # gradiente suave, pero con bordes visualmente nítidos
    im = ax[1].imshow(chl_ma, cmap="turbo", vmin=vmin, vmax=vmax,
                      **({"interpolation": _interp} if _interp else {}))
    # dos niveles biológicos: contorno naranja = biomasa elevada; rojo = floración (>= thr).
    # En gradient_focus se aligeran (y se omite el de 'elevada') para no tapar el gradiente.
    lw_elev = 0.0 if gradient_focus else 1.0
    lw_risk = 0.6 if gradient_focus else 1.6
    a_cont  = 0.5 if gradient_focus else 1.0
    if elevf.sum() > 0 and lw_elev > 0:
        ax[1].contour(elevf, levels=[0.5], colors="#ff9800", linewidths=lw_elev, linestyles="--", alpha=a_cont)
    if riskf.sum() > 0:
        ax[1].contour(riskf, levels=[0.5], colors="red", linewidths=lw_risk, alpha=a_cont)
    nowcast = nowcast_level is not None
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label(("Clorofila-a observada (ug/L) — biomasa algal" if nowcast
                  else "Clorofila-a prevista (ug/L) — biomasa algal"), fontsize=10)
    sub = {"heuristic": f"tierra = gris  ·  patrón espacial HEURÍSTICO de hoy, escalado al pronóstico +{h}d",
           "uniform":    "tierra = gris  ·  agua uniforme (horizonte body-level: sin detalle por píxel)",
           "nowcast":    "tierra = gris  ·  nivel OBSERVADO de hoy repartido por el patrón espacial estimado"}[spatial_mode]
    titulo2 = ("2) Biomasa algal actual (hoy, observado)" if nowcast
               else f"2) Dónde se espera más biomasa algal (a +{h} días)")
    ax[1].set_title(f"{titulo2}\n{sub}", fontsize=12)
    # Simbología de color (absoluta, igual para todos los cuerpos y fechas):
    # azul = clorofila baja -> verde media -> rojo alta (floración). Contornos = umbrales.
    leg = [Patch(facecolor="0.6", label="Tierra (gris, fuera del análisis)"),
           Patch(facecolor="#3b4cc0", label="Agua: menor intensidad relativa (azul)"),
           Patch(facecolor="#27a35a", label="Agua: intensidad intermedia (verde)"),
           Patch(facecolor="#c1121f", label="Agua: mayor intensidad relativa (rojo)"),
           Line2D([0], [0], color="#ff9800", lw=2, ls="--", label=f"Límite ELEVADA (>= {thr_elev:.0f} ug/L)"),
           Line2D([0], [0], color="red", lw=2, label=f"Límite FLORACIÓN (>= {thr:.0f} ug/L)")]
    # Simbología fuera del mapa: nunca debe ocultar agua, gradiente ni contornos.
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.018),
               ncol=3, fontsize=8.5, framealpha=0.96, borderaxespad=0.0)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{wb.upper()} ({'lago' if group=='freshwater' else 'costa'}) — "
                 f"{'biomasa algal de hoy (observada)' if nowcast else f'pronóstico de biomasa algal a +{h} días'}"
                 f"  |  escena {t0.date() if t0 is not None else '?'}\n"
                 f"clorofila-a media = {chlmean:.1f} ug/L ({C.LEVEL_ES[nivel_body]})   ·   "
                 f"área en floración (>= {thr:.0f}) = {pct_alert:.0f}%   ·   "
                 f"biomasa elevada (>= {thr_elev:.0f}) = {pct_elev:.0f}%",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    stats = {"chl_mean": float(chlmean), "pct_alert": float(pct_alert),
             "pct_elev": float(pct_elev), "thr": float(thr), "thr_elev": float(thr_elev),
             "nivel": nivel_body, "t0": t0, "n_water_px": int(n_water_full), "h": int(h),
             "group": group, "has_spatial": spatial_mode == "heuristic",
             "spatial_mode": spatial_mode}
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
        fig, stats = build_map_figure(
            wb, h, path, t0, gradient_focus=True, focus_water=True, hq=True)
    except ValueError as e:
        print(f"{wb}: {e}"); return
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, f"mapa_{wb}_h{h}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  {wb} +{h}d -> {out} | chl-a media={stats['chl_mean']:.1f} ug/L "
          f"| área de riesgo/biomasa alta={stats['pct_alert']:.0f}%")


def main():
    # El nivel siempre es body-level; el detalle visible es un gradiente espacial heurístico.
    args = sys.argv[1:]
    wb = args[0] if args else "okeechobee"
    h = int(args[1]) if len(args) > 1 and args[1].isdigit() else 3
    # 3er argumento opcional: fecha de escena YYYY-MM-DD (si no, se elige la mejor)
    scene = next((a for a in args[1:] if "-" in a and len(a) == 10), None)
    make_map(wb, h, scene=scene)


if __name__ == "__main__":
    main()
