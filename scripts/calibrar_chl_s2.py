import numpy as np, pandas as pd, os, pickle, warnings, glob, re, json, rasterio
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\JC\Desktop\Tesis")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

BASE = r"C:\Users\JC\Desktop\Tesis"
S2 = ["B2","B3","B4","B5","B8"]
RND = 42
np.random.seed(RND)

# 1. Load 193 samples
df = pd.read_csv(os.path.join(BASE, "datasets", "dataset_entrenamiento_completo_2023_2026.csv"))
df["fecha"] = pd.to_datetime(df["fecha"])
df = df[df["fecha"].dt.year >= 2023].copy()
for c in S2:
    if c in df.columns and df[c].max() > 1.5:
        df[c] /= 10000.0
df = df[(df[S2] > 0).all(axis=1)].copy()
e = 1e-10
df["NDCI"] = (df["B5"] - df["B4"]) / (df["B5"] + df["B4"] + e)
df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + e)
df["FAI"] = df["B8"] - (df["B4"] + (df["B5"] - df["B4"]) * (833 - 665) / (705 - 665))
df["B5_B4_ratio"] = df["B5"] / (df["B4"] + e)
df["B3_B2_ratio"] = df["B3"] / (df["B2"] + e)
df["B8_B3_ratio"] = df["B8"] / (df["B3"] + e)

# Spectral features for CHL regression
F_CHL = ["B2","B3","B4","B5","B8","NDCI","NDVI","FAI","B5_B4_ratio","B3_B2_ratio","B8_B3_ratio"]
chl = df["clorofila_ugl"].values
X = df[F_CHL].values

print(f"Training samples: {len(df)}")
print(f"CHL range: {chl.min():.2f} - {chl.max():.2f} ug/L (median: {np.median(chl):.2f})")
print(f"Log10 CHL range: {np.log10(chl[chl>0]).min():.2f} - {np.log10(chl).max():.2f}")

# 2. Try both raw CHL and log10(CHL) - log is better for optical retrieval
chl_log = np.log10(np.maximum(chl, 0.1))

# 3. Train RandomForest with cross-validation
kf = KFold(5, shuffle=True, random_state=RND)

for name, y in [("CHL_raw", chl), ("CHL_log10", chl_log)]:
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=RND, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error")
    print(f"\n{name}:")
    print(f"  R2 CV: {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"  MAE CV: {mae_scores.mean():.4f} +/- {mae_scores.std():.4f}")

# Train final model on log10(CHL) - better for bio-optical
rf_chl = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=RND, n_jobs=-1)
rf_chl.fit(X, chl_log)

# Predict and evaluate
pred_log = rf_chl.predict(X)
pred_chl = 10 ** pred_log

r_p, _ = pearsonr(chl, pred_chl)
r_s, _ = spearmanr(chl, pred_chl)
rmse = np.sqrt(mean_squared_error(chl, pred_chl))
mae = mean_absolute_error(chl, pred_chl)
print(f"\nFinal model (log10 CHL):")
print(f"  Pearson r: {r_p:.4f}")
print(f"  Spearman rho: {r_s:.4f}")
print(f"  RMSE: {rmse:.2f} ug/L")
print(f"  MAE: {mae:.2f} ug/L")

# Feature importance
fi = pd.DataFrame({"feature": F_CHL, "importance": rf_chl.feature_importances_}).sort_values("importance", ascending=False)
print(f"\nFeature importance:")
for _, r in fi.iterrows():
    print(f"  {r['feature']:15s}: {r['importance']:.4f}")

# Save model
os.makedirs(os.path.join(BASE, "modelos/florida"), exist_ok=True)
with open(os.path.join(BASE, "modelos/florida", "rf_chl_estimator.pkl"), "wb") as f:
    pickle.dump({"model": rf_chl, "features": F_CHL}, f)
print(f"\nModel saved")