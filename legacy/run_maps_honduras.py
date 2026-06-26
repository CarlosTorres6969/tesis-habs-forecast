import numpy as np, pandas as pd, os, pickle, torch, warnings, glob, re, json, rasterio, time
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r'C:\Users\JC\Desktop\Tesis'; RND = 42
np.random.seed(RND)

# ========== Cargar modelo Honduras ==========
print('\n--- Cargando modelo Honduras ---')
mdir = os.path.join(BASE, 'modelos', 'honduras')

xgb_m = xgb.XGBClassifier()
xgb_m.load_model(os.path.join(mdir, 'xgb_honduras.json'))
print('XGBoost Honduras cargado')

sc = pickle.load(open(os.path.join(mdir, 'scaler_honduras.pkl'), 'rb'))
print('Scaler cargado')

try:
    calibrated = pickle.load(open(os.path.join(mdir, 'calibrated_honduras.pkl'), 'rb'))
    use_calibrated = True
    print('Calibrador isotonico cargado')
except:
    use_calibrated = False
    print('Sin calibrador')

meta = torch.load(os.path.join(mdir, 'metadata.pth'), weights_only=True)
F = meta['F']
F_ESP = meta['F_ESP']
ERA_M = meta['ERA_M']
S2 = ['B2','B3','B4','B5','B8']
print(f'Features: {len(F)}')

# ========== Funciones ==========
def feats(bd):
    B2, B3, B4, B5, B8 = [bd[b] for b in S2]
    e = 1e-10
    return np.column_stack([
        B2, B3, B4, B5, B8,
        (B5-B4)/(B5+B4+e), (B8-B4)/(B8+B4+e),
        B8-(B4+(B5-B4)*(833-665)/(705-665)),
        B5/(B4+e), B3/(B2+e), B8/(B3+e),
        # Nuevas features
        (B3-B8)/(B3+B8+e), B4/(B3+e), (B4-B3)/(B4+B3+e),
        B8/(B2+e), B5/(B3+e), B2/(B3+e),
        B8-B4, B3+B4
    ])

def proc_tif(path, st=20):
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
    Xf = np.column_stack([Xs, np.full((len(Xs), len(ERA_M)), ERA_M)])
    Xf_s = sc.transform(Xf.astype('float64'))
    if use_calibrated:
        p = calibrated.predict_proba(Xf_s)[:, 1]
    else:
        p = xgb_m.predict_proba(Xf_s)[:, 1]
    p = np.clip(p, 0, 1)
    pm = np.zeros((H, W), 'float32')
    pm.flat[v] = p
    return pm, pr, tr

# ========== Procesar regiones Honduras ==========
regs = ['Cajon', 'Golfo_Fonseca', 'Lago de Yojoa']
out_dir = os.path.join(BASE, 'mapas_finales')
os.makedirs(out_dir, exist_ok=True)
res = []

for reg in regs:
    tifs = sorted(glob.glob(os.path.join(BASE, 'imagenes', reg, '**', '*.tif'), recursive=True))
    t0 = time.time()
    ok = 0
    for t in tifs:
        r = proc_tif(t, 20)
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
        fout = os.path.join(out_dir, fn.replace('.tif', '_hab_final.tif'))
        with rasterio.open(fout, 'w', **op) as d:
            d.write(pm, 1)
        hab_pct = float((pm > .5).mean() * 100)
        res.append({'region': reg, 'file': fn, 'date': ds, 'hab_pct': hab_pct})
        ok += 1
    elapsed = time.time() - t0
    print(f'{reg:<20s}: {ok}/{len(tifs)} mapas generados ({elapsed:.0f}s)')

print(f'\n--- Total Honduras: {len(res)} mapas ---')
df_res = pd.DataFrame(res)
for r in df_res['region'].unique():
    s = df_res[df_res['region'] == r]
    print(f'  {r:<20s}: {len(s):>4d} mapas, HAB medio: {s["hab_pct"].mean():.2f}%')

# Guardar resumen (append al existente o separado)
resumen_path = os.path.join(out_dir, 'resumen_honduras.csv')
df_res.to_csv(resumen_path, index=False)
print(f'\nResumen guardado en: {resumen_path}')

# Estadisticas de probabilidad
all_probs = []
for r in res:
    tif = os.path.join(out_dir, r['file'].replace('.tif', '_hab_final.tif'))
    if os.path.exists(tif):
        with rasterio.open(tif) as s:
            pm = s.read(1)
            all_probs.extend(pm[pm > 0].flatten())

all_probs = np.array(all_probs)
print(f'\nEstadisticas globales de probabilidad (Honduras):')
print(f'  Media: {np.mean(all_probs):.4f}')
print(f'  Std:   {np.std(all_probs):.4f}')
print(f'  P95:   {np.percentile(all_probs, 95):.4f}')
print(f'  P99:   {np.percentile(all_probs, 99):.4f}')
print(f'  Max:   {np.max(all_probs):.4f}')

print('\n=== COMPLETADO ===')
