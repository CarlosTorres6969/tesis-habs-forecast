"""
fetch_s2_historical_yojoa.py — Descarga Sentinel-2 L2A de Yojoa en las VENTANAS de los articulos
(nov-2020 huracanes Eta/Iota; jun-2021 y ene-2022 muestreos metatranscriptomica) a una carpeta
AISLADA, para NO contaminar el diseno 2023-2026 del modelo (imagenes/ queda intacto).

Reusa EXACTAMENTE el preprocesamiento del modelo (mascara SCL + s2cloudless, bandas B2,B3,B4,B5,B8,
mediana diaria) importando las funciones de fetch_s2_scenes.py. Solo cambia el rango temporal y el
directorio de salida.

Salida: artifacts/validation_articles/s2_historico/yojoa/<fecha>_<idx>.tif
Uso:    EE_PROJECT=... python fetch_s2_historical_yojoa.py
"""
from __future__ import annotations
import os, time
import config as C
from fetch_s2_scenes import _scl_mask, _s2cloudless_mask, _download, BANDS

BBOX = (-88.02, 14.78, -87.90, 14.95)   # mismo bbox que el modelo para Yojoa
OUTDIR = os.path.join(C.DIR_OUT, "validation_articles", "s2_historico", "yojoa")
PROJECT = os.environ.get("EE_PROJECT", "")
MAX_CLOUD = int(os.environ.get("S2_MAXCLOUD", "60"))

# ventanas alineadas a los muestreos de los articulos
WINDOWS = [
    ("2020-10-15", "2020-12-20", "huracanes Eta/Iota (Fadum 2023)"),
    ("2021-05-15", "2021-07-20", "metatranscriptomica jun-2021 (mSystems 2024)"),
    ("2021-12-15", "2022-02-15", "muestreo ene-2022 (mSystems 2024)"),
]


def main():
    import ee
    ee.Initialize(project=PROJECT) if PROJECT else ee.Initialize()
    os.makedirs(OUTDIR, exist_ok=True)
    w, s, e, n = BBOX
    region = ee.Geometry.Rectangle([w, s, e, n])

    total = 0
    for t0, t1, label in WINDOWS:
        coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region).filterDate(t0, t1)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
                .map(lambda im: im.set("d", im.date().format("YYYY-MM-dd"))))
        try:
            dates = sorted(set(coll.aggregate_array("d").getInfo()))
        except Exception as ex:
            print(f"  {label}: fallo listar {type(ex).__name__}"); continue
        print(f"  {label} [{t0}..{t1}]: {len(dates)} fechas con escena")
        ok = 0
        for d in dates:
            day = coll.filter(ee.Filter.eq("d", d)).map(_scl_mask).map(_s2cloudless_mask)
            img = day.median().select(BANDS)
            path = os.path.join(OUTDIR, f"yojoa_{d}_0.tif")
            if os.path.exists(path):
                ok += 1; continue
            try:
                sc = _download(img, region, path)
                if sc is None or (os.path.exists(path) and os.path.getsize(path) < 2000):
                    if os.path.exists(path): os.remove(path)
                    continue
                ok += 1
                time.sleep(0.2)
            except Exception as ex:
                print(f"    {d}: fallo {type(ex).__name__}")
        print(f"    -> {ok} escenas guardadas")
        total += ok
    print(f"\nTOTAL escenas historicas Yojoa: {total} en {OUTDIR}")


if __name__ == "__main__":
    main()
