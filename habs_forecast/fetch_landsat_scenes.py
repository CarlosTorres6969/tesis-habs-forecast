"""
fetch_landsat_scenes.py — Descarga escenas Landsat 8/9 (C2 L2 SR) para DENSIFICAR los cuerpos
debiles, COMPLEMENTANDO a Sentinel-2 (revisita 16 d c/u -> mas fechas claras).

LIMITACION DECLARADA: Landsat OLI NO tiene banda red-edge (~705 nm). Por eso una escena Landsat
aporta azul/verde/rojo/NIR + turbidez, pero NO los indices de clorofila que usan red-edge
(NDCI, CI_red, FAI) -> esos quedan NaN para Landsat (XGBoost los maneja nativo). El valor de
Landsat aqui es sumar FECHAS (mas pares), no mejorar la senal espectral. Se valida con el test
anidado: si ayuda se queda, si no se descarta (resultado honesto).

Fuente : LANDSAT/LC08 + LC09 /C02/T1_L2 (reflectancia de superficie escalada a 0-1).
Nubes  : mascara por QA_PIXEL (nube, sombra, cirrus, nube dilatada).
Salida : imagenes/<carpeta>/<anio>/LS_<nombre>_<YYYY-MM-DD>_<idx>.tif  (4 bandas: blue,green,red,NIR).
         build_scene_state.py detecta el prefijo 'LS_' y procesa Landsat (red-edge -> NaN).
Incremental: solo descarga fechas Landsat que no existan ya (mismo cuerpo) como LS_.

Requiere auth GEE (igual que fetch_s2_scenes): EE_PROJECT + earthengine authenticate.
"""
from __future__ import annotations
import os, re, glob, time, urllib.request
import config as C

BODIES = {
    "cajon":     ("Cajon",         "Cajon",   (-87.80, 14.70, -87.58, 14.95)),
    "yojoa":     ("Lago de Yojoa", "Yojoa",   (-88.02, 14.78, -87.90, 14.95)),
    "fonseca":   ("Golfo_Fonseca", "Fonseca", (-87.85, 12.90, -87.35, 13.45)),
    "tampa_bay": ("TampaBay",      "TampaBay",(-82.75, 27.50, -82.40, 27.95)),
}
T0, T1 = "2023-01-01", "2026-06-30"
SR_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]      # blue, green, red, NIR (sin red-edge)
OUT_BANDS = ["blue", "green", "red", "nir"]
MAX_CLOUD = int(os.environ.get("LS_MAXCLOUD", "80"))
SCALES = [30, 60, 120]
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
PROJECT = os.environ.get("EE_PROJECT", "")


def _existing_ls_dates(folder):
    tifs = glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "LS_*.tif"), recursive=True)
    out = set()
    for t in tifs:
        m = DATE_RE.search(os.path.basename(t))
        if m:
            out.add(m.group(1))
    return out, len(tifs)


def _mask_and_scale(img):
    """Mascara nubes por QA_PIXEL y escala SR a reflectancia 0-1."""
    import ee
    qa = img.select("QA_PIXEL")
    # bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow
    bad = (qa.bitwiseAnd(1 << 1).Or(qa.bitwiseAnd(1 << 2))
           .Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4)))
    sr = img.select(SR_BANDS).multiply(0.0000275).add(-0.2)   # DN -> reflectancia 0-1
    return sr.updateMask(bad.eq(0))


def _download(img, region, path):
    import ee
    last = None
    for sc in SCALES:
        try:
            url = img.getDownloadURL({"region": region, "scale": sc,
                                      "format": "GEO_TIFF", "bands": SR_BANDS})
            urllib.request.urlretrieve(url, path)
            return sc
        except Exception as e:
            last = e
    raise last


def main():
    import sys, ee
    try:
        ee.Initialize(project=PROJECT) if PROJECT else ee.Initialize()
    except Exception as e:
        print("No se pudo inicializar Earth Engine. Corre: earthengine authenticate / set EE_PROJECT")
        print(f"  detalle: {type(e).__name__}: {e}"); return

    sel = [a for a in sys.argv[1:] if a in BODIES]
    todo = {b: v for b, v in BODIES.items() if not sel or b in sel}
    print(f"Cuerpos: {list(todo)} | LS_MAXCLOUD={MAX_CLOUD}%\n")
    total = 0
    for body, (folder, name, (w, s, e, n)) in todo.items():
        existing, n_have = _existing_ls_dates(folder)
        region = ee.Geometry.Rectangle([w, s, e, n])
        coll = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                .filterBounds(region).filterDate(T0, T1)
                .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD))
                .map(lambda im: im.set("d", im.date().format("YYYY-MM-dd"))))
        try:
            dates = sorted(set(coll.aggregate_array("d").getInfo()))
        except Exception as ex:
            print(f"  {body}: error listando fechas: {type(ex).__name__}"); continue
        nuevas = [d for d in dates if d not in existing]
        print(f"{body:10s} ({folder}): {n_have} Landsat en disco | {len(dates)} fechas | "
              f"{len(nuevas)} NUEVAS")
        idx = n_have; ok = 0
        for d in nuevas:
            day = coll.filter(ee.Filter.eq("d", d)).map(_mask_and_scale)
            img = day.median()
            year = d[:4]
            outdir = os.path.join(C.DIR_IMAGENES, folder, year)
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"LS_{name}_{d}_{idx}.tif")
            try:
                _download(img, region, path)
                if os.path.getsize(path) < 2000:
                    os.remove(path); continue
                idx += 1; ok += 1
                if ok % 20 == 0:
                    print(f"    {ok}/{len(nuevas)} descargadas...")
                time.sleep(0.2)
            except Exception:
                pass
        print(f"  -> {ok} escenas Landsat nuevas en imagenes/{folder}/")
        total += ok
    print(f"\nTOTAL Landsat nuevas: {total}")
    print("Siguiente: python build_scene_state.py && python match_pairs.py && python check_integrity.py && python evaluate_nested.py")


if __name__ == "__main__":
    main()
