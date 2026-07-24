"""
fetch_s2_scenes.py — Descarga MÁS escenas Sentinel-2 L2A (2023-2026) para los cuerpos DÉBILES.

Por qué: el cuello de botella de Yojoa/Cajon/Fonseca/Tampa es el NÚMERO de escenas Sentinel-2
(el predictor), no el target satelital (que es denso). Más escenas claras -> más pares causales
-> validación anidada robusta y, en costa, posibilidad de establecer skill.

Fuente   : COPERNICUS/S2_SR_HARMONIZED (reflectancia de superficie, consistente 2023-2026).
Nubes    : SCL (Scene Classification Layer) + s2cloudless (COPERNICUS/S2_CLOUD_PROBABILITY,
           prob>40%) con DILATACIÓN de bordes 2px/120m -> capta nubes finas/bruma/halos que
           SCL subdetecta. Ambas máscaras se aplican en cascada antes de la mediana diaria.
Salida   : formato IDÉNTICO al existente -> build_scene_state.py lo recoge SIN cambios:
             imagenes/<carpeta>/<año>/S2_<nombre>_<YYYY-MM-DD>_<idx>.tif   (bandas B2,B3,B4,B5,B8)
Incremental: solo descarga fechas que NO existan ya en imagenes/<carpeta>/.

----------------------------------------------------------------------------------------------
REQUIERE AUTENTICACIÓN GEE (una sola vez). En la terminal del proyecto:
    !earthengine authenticate            # abre el navegador; usa tu cuenta Google
luego define tu proyecto Cloud con Earth Engine API habilitada:
    set EE_PROJECT=tu-proyecto-gee       # (PowerShell:  $env:EE_PROJECT="tu-proyecto-gee")
y corre:
    python fetch_s2_scenes.py
----------------------------------------------------------------------------------------------
"""
from __future__ import annotations
import os, re, glob, time, urllib.request
from datetime import date, timedelta
import config as C

# Cuerpos DÉBILES a densificar.
#   nombre -> (carpeta en imagenes/, nombre de archivo, bbox [oeste,sur,este,norte])
BODIES = {
    "yojoa":     ("Lago de Yojoa", "Yojoa",   (-88.02, 14.78, -87.90, 14.95)),
    "cajon":     ("Cajon",         "Cajon",   (-87.80, 14.70, -87.58, 14.95)),
    "fonseca":   ("Golfo_Fonseca", "Fonseca", (-87.85, 12.90, -87.35, 13.45)),
    "tampa_bay": ("TampaBay",      "TampaBay",(-82.75, 27.50, -82.40, 27.95)),
}
# Okeechobee normalmente NO se densifica (ya está sólido), pero para captar escenas
# RECIENTES (evita quedar stale) se incluye con S2_INCLUDE_OKEECHOBEE=1. Carpeta/regex de
# fecha compatibles con build_scene_state; bbox reutilizado de fetch_olci_chl.py.
if os.environ.get("S2_INCLUDE_OKEECHOBEE", "") == "1":
    BODIES["okeechobee"] = ("Okeechobee", "Okeechobee", (-81.10, 26.70, -80.60, 27.20))

# Ventana temporal (override por env para captar lo más reciente):
#   S2_T0=YYYY-MM-DD  S2_T1=YYYY-MM-DD  (T1 exclusivo/inclusivo según el filtro de EE)
T0 = os.environ.get("S2_T0", "2023-01-01")
T1 = os.environ.get("S2_T1", (date.today() + timedelta(days=1)).isoformat())
BANDS = ["B2", "B3", "B4", "B5", "B8"]      # mismo orden que lee build_scene_state
# % nubosidad de ESCENA (filtro grueso de la tile 110x110 km; el SCL afina por pixel y
# build_scene_state exige >=MIN_WATER_PIXELS de agua despejada). Override por env para
# cuerpos muy nubosos (ej. Cajon en montana): S2_MAXCLOUD=85
MAX_CLOUD = int(os.environ.get("S2_MAXCLOUD", "60"))
SCALES = [30, 60, 120]  # m: intenta 30, si el GeoTIFF excede el limite de descarga sube
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
PROJECT = os.environ.get("EE_PROJECT", "")   # tu proyecto Cloud con Earth Engine habilitado


def _existing_dates(folder):
    """Fechas ya presentes en imagenes/<folder>/ -> para descarga incremental."""
    tifs = glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True)
    out = set()
    for t in tifs:
        m = DATE_RE.search(os.path.basename(t))
        if m:
            out.add(m.group(1))
    return out, len(tifs)


def _scl_mask(img):
    """Mantiene pixeles limpios; descarta nodata(0), defective(1), shadow(3), cloud med/high(8,9),
    cirrus(10), snow(11). SCL es la banda 'SCL' de S2_SR."""
    import ee
    scl = img.select("SCL")
    bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())


def _s2cloudless_mask(img, prob_threshold=40):
    """Aplica s2cloudless (COPERNICUS/S2_CLOUD_PROBABILITY) para detectar nubes finas/bruma
    que SCL subdetecta. Probabilidad >40% = nube. Dilatacion de 2 pixeles (120m) para captar
    bordes de nube."""
    import ee
    # S2_SR_HARMONIZED y S2_CLOUD_PROBABILITY comparten el mismo system:index (id de
    # granulo) -> se emparejan por igualdad directa (join estandar de s2cloudless).
    scene_id = ee.String(img.get("system:index"))

    # Buscar la imagen de probabilidad de nubes correspondiente
    s2_cloudless = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                    .filter(ee.Filter.eq("system:index", scene_id))
                    .first())

    # Degradar con elegancia: si NO existe el granulo de probabilidad (fechas/teselas
    # de borde), no aplicar mascara adicional en vez de tronar con un ee.Image(null).
    # Se usa una banda de probabilidad = 0 (todo despejado) para que updateMask sea inerte.
    cloud_prob = ee.Image(
        ee.Algorithms.If(s2_cloudless,
                         s2_cloudless,
                         ee.Image.constant(0).rename("probability"))
    ).select("probability")
    is_cloud = cloud_prob.gt(prob_threshold)

    # Dilatar bordes de nube: kernel circular de 2 pixeles para captar halos/bordes
    # difusos que contaminan la mediana (el alcance en metros depende de la escala de
    # reproyeccion; la banda de probabilidad es 10m nativa).
    kernel = ee.Kernel.circle(radius=2, units="pixels")
    is_cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=1)

    return img.updateMask(is_cloud_dilated.Not())


def _download(img, region, path):
    """Descarga un GeoTIFF multibanda; sube de escala si excede el limite de getDownloadURL."""
    import ee
    last = None
    for sc in SCALES:
        try:
            url = img.getDownloadURL({"region": region, "scale": sc,
                                      "format": "GEO_TIFF", "bands": BANDS})
            urllib.request.urlretrieve(url, path)
            return sc
        except Exception as e:
            last = e
            continue
    raise last


def main():
    import ee
    try:
        ee.Initialize(project=PROJECT) if PROJECT else ee.Initialize()
    except Exception as e:
        print("No se pudo inicializar Earth Engine.")
        print("  1) !earthengine authenticate    2) set EE_PROJECT=tu-proyecto    3) reintenta")
        print(f"  detalle: {type(e).__name__}: {e}")
        return

    import sys
    sel = [a for a in sys.argv[1:] if a in BODIES]      # restringir a cuerpos dados (opcional)
    todo = {b: v for b, v in BODIES.items() if not sel or b in sel}
    print(f"Cuerpos: {list(todo)} | MAX_CLOUD={MAX_CLOUD}%\n")

    total_new = 0
    for body, (folder, name, (w, s, e, n)) in todo.items():
        existing, n_have = _existing_dates(folder)
        region = ee.Geometry.Rectangle([w, s, e, n])
        coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region).filterDate(T0, T1)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
                .map(lambda im: im.set("d", im.date().format("YYYY-MM-dd"))))
        try:
            dates = sorted(set(coll.aggregate_array("d").getInfo()))
        except Exception as ex:
            print(f"  {body:10s}: error listando fechas: {type(ex).__name__}: {ex}")
            continue
        nuevas = [d for d in dates if d not in existing]
        print(f"\n{body:10s} ({folder}): {n_have} en disco | {len(dates)} fechas GEE "
              f"(<{MAX_CLOUD}% nubes) | {len(nuevas)} NUEVAS a descargar")
        idx = n_have
        ok = 0
        for d in nuevas:
            # Aplicar AMBAS mascaras: SCL (nubes obvias) + s2cloudless (nubes finas/bruma)
            day = coll.filter(ee.Filter.eq("d", d)).map(_scl_mask).map(_s2cloudless_mask)
            img = day.median().select(BANDS)        # mosaico diario enmascarado
            year = d[:4]
            outdir = os.path.join(C.DIR_IMAGENES, folder, year)
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"S2_{name}_{d}_{idx}.tif")
            try:
                sc = _download(img, region, path)
                # descartar archivos vacios/diminutos (escena totalmente enmascarada)
                if os.path.getsize(path) < 2000:
                    os.remove(path); continue
                idx += 1; ok += 1
                if ok % 20 == 0:
                    print(f"    {ok}/{len(nuevas)} descargadas...")
                time.sleep(0.2)
            except Exception as ex:
                print(f"    {d}: fallo {type(ex).__name__}")
        print(f"  -> {ok} escenas nuevas en imagenes/{folder}/")
        total_new += ok

    print(f"\nTOTAL escenas nuevas: {total_new}")
    print("Siguiente: python build_scene_state.py && python match_pairs.py && "
          "python evaluate_nested.py")


if __name__ == "__main__":
    main()
