import numpy as np, pandas as pd, os, pickle, warnings, glob, re, rasterio, xarray as xr
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\JC\Desktop\Tesis")

BASE = r"C:\Users\JC\Desktop\Tesis"
S2 = ["B2","B3","B4","B5","B8"]
RND = 42; e = 1e-10; np.random.seed(RND)

# Load CHL estimator
with open(os.path.join(BASE, "modelos/florida", "rf_chl_estimator.pkl"), "rb") as f:
    rf_chl = pickle.load(f)["model"]

ERA_DIR = os.path.join(BASE, "era5_temp_nc")

def load_era5_month(y, m):
    ym = f"{y}_{m:02d}"
    i = os.path.join(ERA_DIR, f"era5_instant_{ym}.nc")
    a = os.path.join(ERA_DIR, f"era5_accum_{ym}.nc")
    if not os.path.exists(i) or not os.path.exists(a):
        return None
    ds_i = xr.open_dataset(i)
    ds_a = xr.open_dataset(a)
    lats = ds_i.latitude.values
    lons = ds_i.longitude.values
    # Select 12:00 via valid_time coordinate (time index 2 in each 4-hourly block)
    t_sel = ds_i.valid_time.values[2::4][0]  # first day at 12:00
    # Actually just pick valid_time closest to 12:00 on the 1st day
    from datetime import time
    vt = ds_i.valid_time.values
    mask = np.array([t.hour == 12 for t in pd.DatetimeIndex(vt)])
    t2m = ds_i.t2m.values[mask]
    sp = ds_i.sp.values[mask] / 100.0
    u10 = ds_i.u10.values[mask]
    v10 = ds_i.v10.values[mask]
    if "ssrd" in ds_a:
        ssrd = ds_a.ssrd.values[mask]
        tp = ds_a.tp.values[mask]
    else:
        vt_a = ds_a.valid_time.values
        mask_a = np.array([t.hour == 12 for t in pd.DatetimeIndex(vt_a)])
        ssrd = ds_a.ssrd.values[mask_a]
        tp = ds_a.tp.values[mask_a]
    ds_i.close(); ds_a.close()
    wspd = np.sqrt(u10**2 + v10**2)
    return {"lats": lats, "lons": lons, "t2m": t2m, "sp": sp, "u10": u10, "v10": v10, "wspd": wspd, "ssrd": ssrd, "tp": tp}

def get_era5_bulk(fecha, lat, lon, cache):
    ym = (fecha.year, fecha.month)
    if ym not in cache:
        cache[ym] = load_era5_month(fecha.year, fecha.month)
    if cache[ym] is None:
        return None
    d = cache[ym]
    day = fecha.day - 1  # 0-indexed
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

def prep_bands(b):
    if b.max() > 1.5: b = b / 10000.0
    B2, B3, B4, B5, B8 = b[0], b[1], b[2], b[3], b[4]
    ndci = (B5 - B4) / (B5 + B4 + e)
    ndvi = (B8 - B4) / (B8 + B4 + e)
    fai = B8 - (B4 + (B5 - B4) * (833 - 665) / (705 - 665))
    return np.column_stack([B2, B3, B4, B5, B8, ndci, ndvi, fai, B5/(B4+e), B3/(B2+e), B8/(B3+e)])

REGIONS = {
    "Okeechobee": "florida", "TampaBay": "florida",
    "Golfo_Fonseca": "golfo_fonseca", "Lago de Yojoa": "lago_de_yojoa", "Cajon": "cajon",
}
SAMPLES_PER_TILE = 8
HAB_THRESHOLD = 10.0
era_cache = {}
samples_2026 = []

for reg_name, origen in REGIONS.items():
    tifs = sorted(glob.glob(os.path.join(BASE, "imagenes", reg_name, "2026", "*.tif")))
    print(f"\n{reg_name}: {len(tifs)} images")
    
    for ti, tif_path in enumerate(tifs):
        fn = os.path.basename(tif_path)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", fn)
        if not m: continue
        fecha = pd.Timestamp(m.group(1))
        
        try:
            with rasterio.open(tif_path) as src:
                b = src.read().astype("float32")
                tr = src.transform
                crs_src = src.crs
        except: continue
        
        H, W = b.shape[1], b.shape[2]
        step = max(1, int(np.sqrt(H * W / 5000)))
        b_sub = b[:, ::step, ::step]
        H2, W2 = b_sub.shape[1], b_sub.shape[2]
        
        bf = b_sub.reshape(5, H2 * W2).T
        water = bf.sum(1) > 0
        if water.sum() == 0: continue
        
        xs = np.arange(0, W, step)[:W2]; ys = np.arange(0, H, step)[:H2]
        xx, yy = np.meshgrid(xs, ys)
        xx_f, yy_f = xx.flatten()[water], yy.flatten()[water]
        
        n_take = min(SAMPLES_PER_TILE, water.sum())
        if n_take == 0: continue
        idxs = np.random.choice(water.sum(), n_take, replace=False)
        
        for idx in idxs:
            bw = bf[water][idx]
            col, row = int(xx_f[idx]), int(yy_f[idx])
            utm_x, utm_y = tr * (col + step//2, row + step//2)
            # Convert UTM (EPSG:32617) to geographic (EPSG:4326) for ERA5 lookup
            from rasterio.warp import transform as rio_transform
            lon, lat = rio_transform(crs_src, {"init": "EPSG:4326"}, [utm_x], [utm_y])
            lat, lon = float(lat[0]), float(lon[0])
            
            feats = prep_bands(bw)
            chl_est = float(10 ** rf_chl.predict(feats.reshape(1, -1))[0])
            if chl_est < 0.5: continue
            
            era = get_era5_bulk(fecha, lat, lon, era_cache)
            if era is None: continue
            
            samples_2026.append({
                "fecha": m.group(1), "region": reg_name, "origen": origen,
                "B2": float(bw[0]), "B3": float(bw[1]), "B4": float(bw[2]),
                "B5": float(bw[3]), "B8": float(bw[4]),
                "clorofila_ugl": chl_est,
                "es_floracion": 1 if chl_est >= HAB_THRESHOLD else 0,
                "latitud": lat, "longitud": lon, "archivo_s2": fn, **era
            })
        
        if (ti + 1) % 50 == 0:
            print(f"  {ti+1}/{len(tifs)} - {len(samples_2026)} samples")

print(f"\nTotal 2026 samples: {len(samples_2026)}")
df_2026 = pd.DataFrame(samples_2026)
print(f"HAB: {df_2026['es_floracion'].sum()}/{len(df_2026)} ({df_2026['es_floracion'].mean()*100:.1f}%)")
print(f"CHL mean: {df_2026['clorofila_ugl'].mean():.2f} ug/L")
print(f"By region: {df_2026['region'].value_counts().to_dict()}")

os.makedirs(os.path.join(BASE, "datasets"), exist_ok=True)
df_2026.to_csv(os.path.join(BASE, "datasets", "samples_2026_s2_chl.csv"), index=False)
print("Saved")