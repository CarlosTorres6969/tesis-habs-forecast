"""
build_era5_daily.py — Serie ERA5 DIARIA por cuerpo de agua (2023-2026) desde los NetCDF.

Drivers meteorologicos del pronostico. Extrae el punto de grilla mas cercano al centroide de
cada cuerpo, combina archivos instant (t2m,u10,v10,sp) y accum (ssrd,tp) y agrega a diario:
  - instant -> media diaria ;  accum (radiacion, precip) -> suma diaria.
Calcula wind_speed_10m = hypot(u10,v10). Salida: artifacts/state_series/era5_daily.csv

Nota: Fonseca (~13.2 N) queda al borde sur de la grilla (lat min 14.5) -> se usa el punto mas
cercano como aproximacion (declarar como limitacion).
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import xarray as xr
import config as C

OUT = os.path.join(C.DIR_STATE, "era5_daily.csv")

CENTROIDS = {   # water_body -> (lat, lon)
    "okeechobee": (26.95, -80.85), "tampa_bay": (27.75, -82.55),
    "yojoa": (14.87, -87.97), "cajon": (14.83, -87.70), "fonseca": (13.20, -87.60),
}
RENAME = {"t2m": "temp_air_2m", "ssrd": "solar_radiation", "tp": "precipitation",
          "u10": "wind_u_10m", "v10": "wind_v_10m", "sp": "surface_pressure"}


def _extract(files):
    lats = xr.DataArray([v[0] for v in CENTROIDS.values()], dims="body",
                        coords={"body": list(CENTROIDS)})
    lons = xr.DataArray([v[1] for v in CENTROIDS.values()], dims="body",
                        coords={"body": list(CENTROIDS)})
    parts = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
        except Exception:
            continue
        latname = "latitude" if "latitude" in ds.coords else "lat"
        lonname = "longitude" if "longitude" in ds.coords else "lon"
        pt = ds.sel({latname: lats, lonname: lons}, method="nearest")
        df = pt.to_dataframe().reset_index()
        parts.append(df)
        ds.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build():
    inst = sorted(glob.glob(os.path.join(C.DIR_ERA5_NC, "era5_instant_*.nc")))
    accu = sorted(glob.glob(os.path.join(C.DIR_ERA5_NC, "era5_accum_*.nc")))
    di = _extract(inst)
    da = _extract(accu)
    if di.empty and da.empty:
        print("Sin NetCDF legibles."); return

    tcol = "valid_time" if "valid_time" in di.columns else "time"
    def daily(df, how):
        if df.empty:
            return df
        df["fecha"] = pd.to_datetime(df[tcol]).dt.normalize()
        keep = [c for c in RENAME if c in df.columns]
        agg = {RENAME[c]: (c, how) for c in keep}
        return df.groupby(["body", "fecha"]).agg(**agg).reset_index()

    gi = daily(di, "mean")            # instant -> media diaria
    ga = daily(da, "sum")             # accum   -> suma diaria
    out = gi.merge(ga, on=["body", "fecha"], how="outer") if not ga.empty else gi
    out = out.rename(columns={"body": "water_body"})
    out["wind_speed_10m"] = np.hypot(out.get("wind_u_10m", 0), out.get("wind_v_10m", 0))
    out = out[(out["fecha"] >= "2023-01-01") & (out["fecha"] <= "2026-12-31")]
    out = out.sort_values(["water_body", "fecha"]).reset_index(drop=True)

    os.makedirs(C.DIR_STATE, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"ERA5 diario -> {OUT} ({len(out)} dias-cuerpo)")
    print(out.groupby("water_body").agg(dias=("fecha", "size"),
          desde=("fecha", "min"), hasta=("fecha", "max")).to_string())
    return out


if __name__ == "__main__":
    build()
