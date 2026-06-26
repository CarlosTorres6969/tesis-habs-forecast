"""
fetch_olci_chl.py — TARGET de clorofila Sentinel-3 OLCI 300 m (2023-2026) via openEO / CDSE.

Por que: VIIRS (750 m) es grueso para lagos pequenos (Yojoa, Cajon) y costa estrecha. OLCI
a 300 m densifica/afina el target ahi. openEO agrega la clorofila por bbox EN EL SERVIDOR
(no descarga escenas completas) -> sale una serie diaria por cuerpo.

  >>> REQUIERE TUS CREDENCIALES Copernicus Data Space (las de S2/ERA5). <<<
  Pasos:
    1. pip install openeo
    2. Cuenta gratuita en https://dataspace.copernicus.eu
    3. python fetch_olci_chl.py   (abre el navegador para autenticar la primera vez)

Salida: artifacts/targets/olci_chl_daily.csv  (water_body, fecha, chl_ugl, group)
Formato identico al target VIIRS -> en match_pairs.py basta apuntar TARGET_FILE a este csv,
o concatenar ambos (VIIRS para cuerpos grandes, OLCI para lagos pequenos).

NOTA: el id de coleccion y el nombre de banda de clorofila pueden variar segun el catalogo
CDSE; verificar con connection.list_collections(). Aqui se usa la L2 Water (OC4ME/NN).
Script NO probado en este entorno por requerir login; revisar la primera ejecucion.
"""
from __future__ import annotations
import os
import pandas as pd
import config as C

OUT_DIR = os.path.join(C.DIR_OUT, "targets")
OUT = os.path.join(OUT_DIR, "olci_chl_daily.csv")

# bbox por cuerpo: (oeste, sur, este, norte) + grupo
BODIES = {
    "okeechobee": (-81.10, 26.70, -80.60, 27.20, "freshwater"),
    "tampa_bay":  (-82.75, 27.50, -82.40, 27.95, "marine"),
    "yojoa":      (-88.02, 14.78, -87.90, 14.95, "freshwater"),
    "cajon":      (-87.80, 14.70, -87.58, 14.95, "freshwater"),
    "fonseca":    (-87.85, 12.90, -87.35, 13.45, "marine"),
}
T0, T1 = "2023-01-01", "2026-04-30"

# Coleccion y banda de clorofila en CDSE (verificar con list_collections si falla):
COLLECTION = "SENTINEL3_OLCI_L2_WATER"
CHL_BAND = "CHL_NN"            # alternativa: "CHL_OC4ME"


def build():
    import openeo
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

    years = [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
             ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-04-30")]
    frames = []
    for name, (w, s, e, n, group) in BODIES.items():
        recs = []
        for t0, t1 in years:                       # trocear por anio: evita el 500 de memoria
            try:
                cube = (con.load_collection(
                            COLLECTION, spatial_extent={"west": w, "south": s, "east": e, "north": n},
                            temporal_extent=[t0, t1], bands=[CHL_BAND])
                        .aggregate_spatial(geometries={
                            "type": "Polygon", "coordinates": [[
                                [w, s], [e, s], [e, n], [w, n], [w, s]]]},
                            reducer="median"))
                res = cube.execute()
                for ts, vals in (res.items() if isinstance(res, dict) else []):
                    v = vals
                    while isinstance(v, list) and v:
                        v = v[0]
                    recs.append((ts, v))
            except Exception as ey:
                print(f"      {name} {t0[:4]}: {type(ey).__name__}")
        try:
            df = pd.DataFrame(recs, columns=["fecha", "raw"])
            raw = pd.to_numeric(df["raw"], errors="coerce")
            is_log = (raw.min() < 0) or (raw.max() < 3)        # heuristica de escala log10
            df["chl_ugl"] = (10 ** raw) if is_log else raw
            print(f"      escala detectada: {'log10' if is_log else 'lineal'}")
            df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
            df = df.dropna(subset=["chl_ugl"])
            df = df[(df["chl_ugl"] > 0) & (df["chl_ugl"] < 2000)]
            df["water_body"] = name; df["group"] = group
            frames.append(df[["water_body", "fecha", "chl_ugl", "group"]])
            print(f"  {name:12s}: {len(df)} dias OLCI")
        except Exception as ex:
            print(f"  {name:12s}: FALLO {type(ex).__name__}: {ex}")

    if not frames:
        print("Sin datos OLCI."); return
    out = pd.concat(frames, ignore_index=True).sort_values(["water_body", "fecha"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nTarget OLCI -> {OUT} ({len(out)} dias-cuerpo)")


if __name__ == "__main__":
    build()
