import numpy as np, pandas as pd, os, pickle, torch, warnings, glob, re, json, rasterio, time
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r'C:\Users\JC\Desktop\Tesis'; RND = 42
np.random.seed(RND)

S2 = ['B2','B3','B4','B5','B8']
ERA = ['temp_air_2m','solar_radiation','precipitation','wind_speed_10m','wind_u_10m','wind_v_10m','surface_pressure']
ERA_M = [22.5, 18000000, 0.003, 3.5, -0.5, 1.2, 1013.25]
F = ['B2','B3','B4','B5','B8','NDCI','NDVI','FAI','B5_B4_ratio','B3_B2_ratio','B8_B3_ratio'] + ERA
T = 'es_floracion'
print(f'Dispositivo: {DEVICE}')

# ========== Cargar modelos ==========
print('\n--- Cargando modelos ---')
out = os.path.join(BASE, 'modelos', 'florida')

# XGBoost
xgb_m = xgb.XGBClassifier()
xgb_m.load_model(os.path.join(out, 'xgb_model.json'))
print(f'XGBoost cargado')

# Scaler
sc = pickle.load(open(os.path.join(out, 'scaler.pkl'), 'rb'))
print(f'Scaler cargado')

# Spectral shifts por region
try:
    shifts = pickle.load(open(os.path.join(out, 'spectral_shifts.pkl'), 'rb'))
    print(f'Shifts por region cargados: {list(shifts.keys())}')
    for k, v in shifts.items():
        print(f'  {k:<20s}: B4={v[2]:+.4f} B8={v[4]:+.4f}')
except:
    shifts = None
    print('Sin shifts por region')

# Quantile mapping
try:
    q_mapping = pickle.load(open(os.path.join(out, 'quantile_mapping.pkl'), 'rb'))
    print(f'Quantile mapping cargado: {len(q_mapping)} features')
except:
    q_mapping = None
    print('Sin quantile mapping')

SHIFT_REG_MAP = {
    'Cajon': 'cajon',
    'Golfo_Fonseca': 'golfo_fonseca',
    'Lago de Yojoa': 'lago_de_yojoa'
}

# ========== Funciones ==========
def feats(bd):
    B2, B3, B4, B5, B8 = [bd[b] for b in S2]
    e = 1e-10
    return np.column_stack([
        B2, B3, B4, B5, B8,
        (B5-B4)/(B5+B4+e), (B8-B4)/(B8+B4+e),
        B8-(B4+(B5-B4)*(833-665)/(705-665)),
        B5/(B4+e), B3/(B2+e), B8/(B3+e)
    ])

def proc_tif(path, dom, reg=None, st=20):
    with rasterio.open(path) as s:
        b = s.read().astype('float32') / 10000.
        pr = s.profile
        tr = s.transform
    if st > 1:
        b = b[:, ::st, ::st]
    H, W = b.shape[1], b.shape[2]
    bf = b.reshape(5, H*W).T
    # Mascara de agua: NDWI > -0.5 (consistente con entrenamiento) y NDVI < 0.4
    b3, b4, b8 = bf[:, 1], bf[:, 2], bf[:, 4]
    e = 1e-10
    ndwi = (b3 - b8) / (b3 + b8 + e)
    ndvi = (b8 - b4) / (b8 + b4 + e)
    v = (bf.sum(1) > 0) & (ndwi > -0.5) & (ndvi < 0.4)
    if v.sum() == 0:
        return None
    bv = bf[v]
    Xs = feats({sb: bv[:, i] for i, sb in enumerate(S2)})
    if dom != 'Florida' and shifts is not None and reg is not None:
        sk = SHIFT_REG_MAP.get(reg)
        if sk in shifts:
            Xs += shifts[sk]
    Xf = np.column_stack([Xs, np.full((len(Xs), len(ERA_M)), ERA_M)])
    p = xgb_m.predict_proba(sc.transform(Xf.astype('float64')))[:, 1]
    p = np.clip(p, 0, 1)
    pm = np.zeros((H, W), 'float32')
    pm.flat[v] = p
    return pm, pr, tr

# ========== Procesar regiones ==========
regs = {
    'Okeechobee': 'Florida',
    'TampaBay': 'Florida',
    'Cajon': 'Honduras',
    'Golfo_Fonseca': 'Honduras',
    'Lago de Yojoa': 'Honduras'
}

out_dir = os.path.join(BASE, 'mapas_finales')
os.makedirs(out_dir, exist_ok=True)
res = []

for reg, dom in regs.items():
    tifs = sorted(glob.glob(os.path.join(BASE, 'imagenes', reg, '**', '*.tif'), recursive=True))
    t0 = time.time()
    ok = 0
    for t in tifs:
        r = proc_tif(t, dom, reg, 20)
        if r is None:
            continue
        pm, pr, tr = r
        fn = os.path.basename(t)
        ds = re.search(r'(\d{4}-\d{2}-\d{2})', fn)
        ds = ds.group(1) if ds else 'unknown'
        op = pr.copy()
        op.update({
            'count': 1, 'dtype': 'float32',
            'height': pm.shape[0], 'width': pm.shape[1],
            'transform': tr * rasterio.Affine.scale(20, 20)
        })
        with rasterio.open(os.path.join(out_dir, fn.replace('.tif', '_hab_final.tif')), 'w', **op) as d:
            d.write(pm, 1)
        hab_pct = float((pm > .5).mean() * 100)
        res.append({'region': reg, 'file': fn, 'date': ds, 'hab_pct': hab_pct})
        ok += 1
    elapsed = time.time() - t0
    print(f'{reg:<20s}: {ok}/{len(tifs)} mapas generados ({elapsed:.0f}s)')

print(f'\n--- Total: {len(res)} mapas ---')
df_res = pd.DataFrame(res)
for r in df_res['region'].unique():
    s = df_res[df_res['region'] == r]
    print(f'  {r:<20s}: {len(s):>4d} mapas, HAB medio: {s["hab_pct"].mean():.2f}%')

df_res.to_csv(os.path.join(out_dir, 'resumen.csv'), index=False)
print(f'\nResumen guardado en mapas_finales/resumen.csv')
