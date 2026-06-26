"""
ingest_insitu.py — Ingesta del TARGET in-situ de alta frecuencia (clorofila-a) desde WQP.

Motivo: el pipeline anterior usaba florida_chlorophyll_limpio.csv (1.413 filas), que
descartaba el 99.7% del in-situ. El raw del Water Quality Portal trae ~525k medidas de
Chlorophyll a con fecha y coordenadas -> serie temporal densa que SI permite construir
targets a t+h para h en {0,1,3,5,7}.

Salida: artifacts/targets/insitu_chl.csv con columnas
    fecha, lat, lon, chl_ugl, station, water_body, group
limpio: fechas validas (2015-2026), chl>0, unidades normalizadas a ug/L.

NO empareja con predictores aqui (eso es match_pairs.py). Solo deja el target denso y
reporta cuantos pares 0-7 d son alcanzables.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import config as C
from build_pairs import assign_water_body

RAW_FILES = [
    os.path.join(C.DIR_DATASETS, "florida", "florida_chlorophyll_wqp_2025_2026.csv"),
    os.path.join(C.DIR_DATASETS, "florida", "florida_chlorophyll_wqp.csv"),
]
OUT_DIR = os.path.join(C.DIR_OUT, "targets")
OUT = os.path.join(OUT_DIR, "insitu_chl.csv")

COLS = {
    "date": "ActivityStartDate",
    "lat":  "ActivityLocation/LatitudeMeasure",
    "lon":  "ActivityLocation/LongitudeMeasure",
    "val":  "ResultMeasureValue",
    "unit": "ResultMeasure/MeasureUnitCode",
    "stn":  "MonitoringLocationIdentifier",
}
# Restriccion del proyecto: SOLO 2023-2026 (ventana de cobertura Sentinel-2).
# En esta ventana el in-situ es escaso (~468 medidas) -> rol = VALIDACION, no target denso.
DATE_MIN, DATE_MAX = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-12-31")


def _load_one(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    use = [v for v in COLS.values()]
    df = pd.read_csv(path, low_memory=False)
    have = {k: v for k, v in COLS.items() if v in df.columns}
    if "date" not in have or "val" not in have:
        return pd.DataFrame()
    out = pd.DataFrame({
        "fecha": pd.to_datetime(df[have["date"]], errors="coerce"),
        "lat":   pd.to_numeric(df.get(have.get("lat")), errors="coerce"),
        "lon":   pd.to_numeric(df.get(have.get("lon")), errors="coerce"),
        "chl":   pd.to_numeric(df[have["val"]], errors="coerce"),
        "unit":  df.get(have.get("unit"), pd.Series(index=df.index, dtype=object)),
        "station": df.get(have.get("stn"), pd.Series(index=df.index, dtype=object)),
    })
    return out


def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a ug/L. ug/L == mg/m3 (equivalentes). mg/L -> *1000."""
    u = df["unit"].astype(str).str.lower().str.strip()
    factor = pd.Series(1.0, index=df.index)
    factor[u.str.contains("mg/l", na=False)] = 1000.0      # mg/L -> ug/L
    # ug/l, mg/m3, ppb, ug/l as chl -> factor 1
    df["chl_ugl"] = df["chl"] * factor
    return df


STATIONS = os.path.join(C.DIR_DATASETS, "wqp_stations.csv")


def _merge_station_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Geolocaliza por MonitoringLocationIdentifier usando wqp_stations.csv.
    Las columnas ActivityLocation vienen vacias en ~99% de las filas."""
    if not os.path.exists(STATIONS):
        return df
    st = pd.read_csv(STATIONS).rename(columns={
        "MonitoringLocationIdentifier": "station",
        "LatitudeMeasure": "lat_st", "LongitudeMeasure": "lon_st"})
    st = st.drop_duplicates("station")[["station", "lat_st", "lon_st"]]
    df = df.merge(st, on="station", how="left")
    df["lat"] = df["lat"].fillna(df["lat_st"])
    df["lon"] = df["lon"].fillna(df["lon_st"])
    return df.drop(columns=["lat_st", "lon_st"])


def build():
    parts = [_load_one(p) for p in RAW_FILES]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    n0 = len(df)

    df = df.dropna(subset=["fecha", "chl"])
    df = df[(df["fecha"] >= DATE_MIN) & (df["fecha"] <= DATE_MAX)]
    df = _normalize_units(df)
    df = df[(df["chl_ugl"] > 0) & (df["chl_ugl"] < 2000)]      # rango fisico plausible
    df = _merge_station_coords(df)
    df = df.dropna(subset=["lat", "lon"])

    wb = df.apply(lambda r: assign_water_body(r["lat"], r["lon"]), axis=1)
    df["water_body"] = [w[0] for w in wb]
    df["group"] = [w[1] for w in wb]

    # colapsar a una medida por estacion-dia (mediana, robusta a outliers de sonda)
    df["dia"] = df["fecha"].dt.normalize()
    daily = (df.groupby(["station", "dia", "lat", "lon", "water_body", "group"],
                        as_index=False)["chl_ugl"].median())
    daily = daily.rename(columns={"dia": "fecha"}).sort_values("fecha")

    os.makedirs(OUT_DIR, exist_ok=True)
    daily.to_csv(OUT, index=False)

    # ------------------------------- reporte -------------------------------
    print(f"Crudo: {n0} filas -> validas estacion-dia: {len(daily)}")
    print(f"Estaciones: {daily['station'].nunique()} | rango "
          f"{daily['fecha'].min().date()} .. {daily['fecha'].max().date()}")
    print(f"Salida: {OUT}\n")

    print("=== medidas por grupo / cuerpo de agua ===")
    print(daily.groupby(["group", "water_body"]).size().to_string())

    # potencial de pares 0-7d: pares consecutivos por estacion con gap<=7
    print("\n=== potencial de targets 0-7 d (gaps consecutivos por estacion) ===")
    for g in ("freshwater", "marine"):
        sub = daily[daily["group"] == g]
        gaps = []
        for s, gg in sub.groupby("station"):
            d = gg["fecha"].drop_duplicates().sort_values().diff().dropna().dt.days
            gaps.extend(d.tolist())
        gaps = np.array(gaps)
        n7 = int((gaps <= 7).sum()) if len(gaps) else 0
        print(f"  {g:11s}: {len(sub):>6} medidas | pares consecutivos gap<=7d = {n7}")
    return daily


if __name__ == "__main__":
    build()
