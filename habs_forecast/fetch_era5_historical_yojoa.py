"""
fetch_era5_historical_yojoa.py — ERA5-Land diario para Yojoa 2020-2022 (via Earth Engine),
en el MISMO esquema que era5_daily.csv del modelo, para reconstruir las 9 features ERA5 en las
fechas de los articulos. Fuente: ECMWF/ERA5_LAND/DAILY_AGGR (reanalisis, 1950-presente).

Unidades alineadas al ERA5 de entrenamiento: t2m [K], ssrd [J/m2/dia], tp [m/dia], sp [Pa],
viento [m/s]. Salida: artifacts/state_series/era5_daily_hist_yojoa.csv
Uso: EE_PROJECT=... python fetch_era5_historical_yojoa.py
"""
from __future__ import annotations
import os
import pandas as pd
import config as C

OUT = os.path.join(C.DIR_STATE, "era5_daily_hist_yojoa.csv")
PROJECT = os.environ.get("EE_PROJECT", "")
CENTROID = (14.87, -87.97)   # mismo que build_era5_daily
T0, T1 = "2020-09-01", "2022-03-01"


def main():
    import ee
    ee.Initialize(project=PROJECT) if PROJECT else ee.Initialize()
    pt = ee.Geometry.Point([CENTROID[1], CENTROID[0]])
    bands = {
        "temperature_2m": "temp_air_2m",
        "surface_solar_radiation_downwards_sum": "solar_radiation",
        "total_precipitation_sum": "precipitation",
        "surface_pressure": "surface_pressure",
        "u_component_of_wind_10m": "wind_u_10m",
        "v_component_of_wind_10m": "wind_v_10m",
    }
    coll = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(T0, T1).select(list(bands)))

    def samp(img):
        d = img.reduceRegion(ee.Reducer.mean(), pt.buffer(2000), 1000)
        return ee.Feature(None, d).set("fecha", img.date().format("YYYY-MM-dd"))

    fc = coll.map(samp).filter(ee.Filter.notNull(list(bands)))
    feats = fc.getInfo()["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({"fecha": p["fecha"], **{bands[k]: p.get(k) for k in bands}})
    df = pd.DataFrame(rows)
    if df.empty:
        print("Sin datos ERA5."); return
    import numpy as np
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["wind_speed_10m"] = np.hypot(df["wind_u_10m"], df["wind_v_10m"])
    df["water_body"] = "yojoa"
    df = df.sort_values("fecha")
    os.makedirs(C.DIR_STATE, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"ERA5 historico Yojoa -> {OUT} ({len(df)} dias | {df['fecha'].min().date()} -> {df['fecha'].max().date()})")
    print(df[["fecha", "temp_air_2m", "solar_radiation", "precipitation",
              "wind_speed_10m", "surface_pressure"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
