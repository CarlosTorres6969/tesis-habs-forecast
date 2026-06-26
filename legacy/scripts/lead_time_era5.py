import numpy as np, pandas as pd, os, warnings, xarray as xr
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\JC\Desktop\Tesis")

BASE = r"C:\Users\JC\Desktop\Tesis"
ERA_DIR = os.path.join(BASE, "era5_temp_nc")

df = pd.read_csv(os.path.join(BASE, "datasets", "dataset_entrenamiento_completo_2023_2026.csv"))
df["fecha"] = pd.to_datetime(df["fecha"])
# Keep only samples with valid coordinates
df = df[df["lat"].notna() & df["lon"].notna()].copy()
print(f"Dataset: {len(df)} samples (filtered to those with coordinates)")

era_cache = {}

def load_era5_month(y, m):
    ym = f"{y}_{m:02d}"
    i = os.path.join(ERA_DIR, f"era5_instant_{ym}.nc")
    a = os.path.join(ERA_DIR, f"era5_accum_{ym}.nc")
    if not os.path.exists(i) or not os.path.exists(a):
        return None
    ds_i = xr.open_dataset(i)
    ds_a = xr.open_dataset(a)
    lats = ds_i.latitude.values; lons = ds_i.longitude.values
    vt = ds_i.valid_time.values
    mask = np.array([t.hour == 12 for t in pd.DatetimeIndex(vt)])
    d = {
        "lats": lats, "lons": lons,
        "t2m": ds_i.t2m.values[mask],
        "sp": ds_i.sp.values[mask] / 100.0,
        "u10": ds_i.u10.values[mask],
        "v10": ds_i.v10.values[mask],
        "ssrd": ds_a.ssrd.values[mask],
        "tp": ds_a.tp.values[mask],
    }
    ds_i.close(); ds_a.close()
    d["wspd"] = np.sqrt(d["u10"]**2 + d["v10"]**2)
    return d

def get_era5_at(fecha, lat, lon):
    ym = (fecha.year, fecha.month)
    if ym not in era_cache:
        era_cache[ym] = load_era5_month(fecha.year, fecha.month)
    if era_cache[ym] is None:
        return None
    d = era_cache[ym]
    day = fecha.day - 1
    if day < 0 or day >= d["t2m"].shape[0]:
        return None
    ilat = np.argmin(np.abs(d["lats"] - lat))
    ilon = np.argmin(np.abs(d["lons"] - lon))
    return {
        "temp_air_2m": float(d["t2m"][day, ilat, ilon]),
        "solar_radiation": float(d["ssrd"][day, ilat, ilon]),
        "precipitation": float(d["tp"][day, ilat, ilon]),
        "wind_speed_10m": float(d["wspd"][day, ilat, ilon]),
        "wind_u_10m": float(d["u10"][day, ilat, ilon]),
        "wind_v_10m": float(d["v10"][day, ilat, ilon]),
        "surface_pressure": float(d["sp"][day, ilat, ilon]),
    }

LEADS = [("t0", 0), ("t-1", -1), ("t-3", -3), ("t-5", -5), ("t-7", -7)]
ERA_VARS = ["temp_air_2m", "solar_radiation", "precipitation", "wind_speed_10m", "wind_u_10m", "wind_v_10m", "surface_pressure"]

rows = []
total = len(df)
for i, (_, row) in enumerate(df.iterrows()):
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{total}")
    
    lat, lon = float(row["lat"]), float(row["lon"])
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        continue
    fecha = row["fecha"]
    entry = {"idx": i}
    
    for lead_name, lead_days in LEADS:
        lead_fecha = fecha + pd.Timedelta(days=lead_days)
        era = get_era5_at(lead_fecha, float(lat), float(lon))
        if era is None:
            for v in ERA_VARS:
                entry[f"{v}_{lead_name}"] = np.nan
        else:
            for v in ERA_VARS:
                entry[f"{v}_{lead_name}"] = era[v]
    
    rows.append(entry)

df_lead = pd.DataFrame(rows)
print(f"\nLead-time ERA5: {len(df_lead)} samples, {len(df_lead.columns)} columns")
print(f"Columns: {[c for c in df_lead.columns if 't0' in c][:3]}...")
print(f"NaN count: {df_lead.isna().sum().sum()}")

df_lead.to_csv(os.path.join(BASE, "datasets", "lead_time_era5.csv"), index=False)
print("Saved to datasets/lead_time_era5.csv")
