import pandas as pd, os
BASE = r"C:\Users\JC\Desktop\Tesis"

# Load existing 2023-2025 dataset (from original, before 2026 was added with NaN ERA5)
old = pd.read_csv(os.path.join(BASE, "datasets", "dataset_entrenamiento_completo_2023_2026.csv"))
old["fecha"] = pd.to_datetime(old["fecha"])
# Keep only samples with valid ERA5 (the original 2023-2025 data)
old = old[old["fecha"].dt.year >= 2023].copy()
old = old[old["temp_air_2m"].notna()].copy()  # only original samples with valid ERA5
print(f"Existing (valid ERA5 only): {len(old)} samples, HAB={old['es_floracion'].sum()}")

# Load 2026 samples
s26 = pd.read_csv(os.path.join(BASE, "datasets", "samples_2026_s2_chl.csv"))
s26["fecha"] = pd.to_datetime(s26["fecha"])
print(f"2026 samples: {len(s26)} samples, HAB={s26['es_floracion'].sum()}")

# Standardize columns before concatenation
old_renamed = old.rename(columns={"lat": "latitud", "lon": "longitud"})
old_renamed["region"] = ""
old_renamed["archivo_s2"] = ""

s26_cols = ["fecha","region","origen","B2","B3","B4","B5","B8","clorofila_ugl","es_floracion",
            "latitud","longitud","archivo_s2","temp_air_2m","solar_radiation","precipitation",
            "wind_speed_10m","wind_u_10m","wind_v_10m","surface_pressure"]

full = pd.concat([old_renamed[s26_cols], s26[s26_cols]], ignore_index=True)
full = full.sort_values("fecha").reset_index(drop=True)
print(f"\nFull 2023-2026: {len(full)} samples, HAB={full['es_floracion'].sum()} ({full['es_floracion'].mean()*100:.1f}%)")
print(f"Years: {sorted(full['fecha'].dt.year.unique())}")
print(f"Origenes: {full['origen'].value_counts().to_dict()}")

full.to_csv(os.path.join(BASE, "datasets", "dataset_entrenamiento_completo_2023_2026.csv"), index=False)
print("Saved")