import numpy as np, pandas as pd, os, pickle, torch, warnings, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r'C:\Users\JC\Desktop\Tesis'; RND = 42
np.random.seed(RND); torch.manual_seed(RND)

S2 = ['B2','B3','B4','B5','B8']
ERA = ['temp_air_2m','solar_radiation','precipitation','wind_speed_10m','wind_u_10m','wind_v_10m','surface_pressure']
ERA_M = [22.5, 18000000, 0.003, 3.5, -0.5, 1.2, 1013.25]
F = ['B2','B3','B4','B5','B8','NDCI','NDVI','FAI','B5_B4_ratio','B3_B2_ratio','B8_B3_ratio'] + ERA
T = 'es_floracion'
print(f'Dispositivo: {DEVICE}')

# ========== HABNet1Dv2 ==========
class FocalLoss(torch.nn.Module):
    def __init__(self, a=0.25, g=2.0, pw=None):
        super().__init__(); self.a=a; self.g=g; self.pw=pw
    def forward(self, l, t):
        b = torch.nn.functional.binary_cross_entropy_with_logits(l, t, pos_weight=self.pw, reduction='none')
        p = torch.sigmoid(l); pt = p*t + (1-p)*(1-t); fw = (1-pt)**self.g
        if self.a >= 0: fw *= (self.a*t + (1-self.a)*(1-t))
        return (fw*b).mean()

class HABNet1Dv2(torch.nn.Module):
    def __init__(self, nf, h=128, nr=2, d=0.2):
        super().__init__()
        self.input_proj = torch.nn.Sequential(torch.nn.Linear(nf,h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(d))
        self.res_blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(h,h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(d), torch.nn.Linear(h,h), torch.nn.BatchNorm1d(h)) for _ in range(nr)])
        self.se = torch.nn.Sequential(torch.nn.Linear(h,max(h//4,8)), torch.nn.ReLU(), torch.nn.Linear(max(h//4,8),h), torch.nn.Sigmoid())
        self.classifier = torch.nn.Sequential(torch.nn.Linear(h,h//2), torch.nn.GELU(), torch.nn.Dropout(d*.5), torch.nn.Linear(h//2,1))

    def forward(self, x):
        x = self.input_proj(x)
        for b in self.res_blocks:
            x = x + b(x)
        return self.classifier(x * self.se(x))

# ========== Carga y preprocesamiento ==========
print('\n--- Cargando datos ---')
df = pd.read_csv(os.path.join(BASE, 'datasets', 'dataset_entrenamiento_completo_2023_2026_clean.csv'))
df['fecha'] = pd.to_datetime(df['fecha'])

# Solo Florida
df = df[df['origen'] == 'florida'].copy()
print(f'Muestras Florida: {len(df)}')

def prep(df):
    for c in S2:
        if c in df.columns and df[c].max() > 1.5: df[c] /= 10000.0
    df = df[(df[S2] > 0).all(axis=1)].copy()
    e = 1e-10
    df['NDCI'] = (df['B5'] - df['B4']) / (df['B5'] + df['B4'] + e)
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + e)
    df['FAI'] = df['B8'] - (df['B4'] + (df['B5'] - df['B4']) * (833 - 665) / (705 - 665))
    df['B5_B4_ratio'] = df['B5'] / (df['B4'] + e)
    df['B3_B2_ratio'] = df['B3'] / (df['B2'] + e)
    df['B8_B3_ratio'] = df['B8'] / (df['B3'] + e)
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + e)
    df = df[df['NDWI'] > -0.5].copy()
    df = df[df['NDVI'] < 0.4].copy()
    return df

fl = prep(df)
X = fl[F].values.astype('float32')
y = fl[T].values.astype('float32')
chl = fl['clorofila_ugl'].values

print(f'Muestras finales: {len(fl)}')
print(f'HAB: {int(y.sum())}/{len(y)} ({y.mean()*100:.1f}%)')

# Split estratificado
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=RND, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
print(f'Train: {len(X_tr)}, Test: {len(X_te)}')

# ========== 5-Fold CV ==========
print('\n--- 5-Fold Stratified CV ---')
skf = StratifiedKFold(5, shuffle=True, random_state=RND)
cvr, mods = [], []

scaler_folds = []
for f, (ti, vi) in enumerate(skf.split(X, y)):
    X_fold_tr, y_fold_tr = X[ti], y[ti]
    X_fold_val, y_fold_val = X[vi], y[vi]

    sc_fold = StandardScaler()
    X_fold_tr_s = sc_fold.fit_transform(X_fold_tr)
    X_fold_val_s = sc_fold.transform(X_fold_val)

    pw = int((y_fold_tr==0).sum()) / max(int(y_fold_tr.sum()), 1)
    dl_tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_fold_tr_s).float(), torch.tensor(y_fold_tr).unsqueeze(1).float()), 32, True)
    dl_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_fold_val_s).float(), torch.tensor(y_fold_val).unsqueeze(1).float()), 32, False)

    m = HABNet1Dv2(len(F)).to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, 100, 1e-5)
    c = FocalLoss(a=.25, g=2, pw=torch.tensor([pw]).to(DEVICE))
    ba, es = -1, 0

    for ep in range(200):
        m.train()
        for xb, yb in dl_tr:
            o.zero_grad()
            c(m(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            o.step()
        m.eval()
        pv, yv = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                pv.extend(torch.sigmoid(m(xb.to(DEVICE))).cpu().numpy().flatten())
                yv.extend(yb.numpy().flatten())
        va = roc_auc_score(yv, pv)
        s.step()
        if va > ba:
            ba = va
            bs = {k: v.clone() for k, v in m.state_dict().items()}
            es = 0
        else:
            es += 1
        if es >= 25: break

    m.load_state_dict(bs)
    mods.append(m)
    scaler_folds.append(sc_fold)
    ap, at = np.array(pv), np.array(yv)
    ad = (ap >= .5).astype(int)
    cvr.append({'fold': f+1, 'auc': roc_auc_score(at, ap), 'f1': f1_score(at, ad), 'acc': accuracy_score(at, ad)})
    print(f'Fold {f+1}: AUC={cvr[-1]["auc"]:.4f} F1={cvr[-1]["f1"]:.4f}')

bix = np.argmax([x['auc'] for x in cvr])
best = mods[bix]
sc = scaler_folds[bix]
print(f'Mejor fold: {bix+1} (AUC={cvr[bix]["auc"]:.4f})')

# ========== Hold-Out ==========
print('\n--- Hold-Out: HABNet1Dv2 vs XGBoost ---')
with torch.no_grad():
    pn = torch.sigmoid(best(torch.tensor(X_te_s, device=DEVICE))).cpu().numpy().flatten()

pw_xgb = int((y_tr==0).sum()) / max(int(y_tr.sum()), 1)
xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                           scale_pos_weight=pw_xgb, random_state=RND, device='cuda', tree_method='hist', verbosity=0)
xgb_m.fit(X_tr_s, y_tr)
px = xgb_m.predict_proba(X_te_s)[:,1]

for n, p in [('HABNet1Dv2', pn), ('XGBoost', px)]:
    auc = roc_auc_score(y_te, p)
    f1 = f1_score(y_te, (p>=.5).astype(int))
    acc = accuracy_score(y_te, (p>=.5).astype(int))
    print(f'{n}: AUC={auc:.4f} F1={f1:.4f} Acc={acc:.4f}')

# ========== Validacion vs Chlorophyll ==========
print('\n--- Correlacion con Clorofila ---')
X_all_s = sc.transform(X.astype('float64')).astype('float32')
with torch.no_grad():
    pn_all = torch.sigmoid(best(torch.tensor(X_all_s, device=DEVICE))).cpu().numpy().flatten()
px_all = xgb_m.predict_proba(X_all_s)[:,1]

y_all = y
r_pn, _ = pearsonr(pn_all, chl)
r_px, _ = pearsonr(px_all, chl)
r_sn, _ = spearmanr(pn_all, chl)
r_sx, _ = spearmanr(px_all, chl)
print(f'HABNet1Dv2: Pearson r={r_pn:.4f}, Spearman rho={r_sn:.4f}')
print(f'XGBoost:    Pearson r={r_px:.4f}, Spearman rho={r_sx:.4f}')

# ========== Metricas Florida ==========
print('\n--- Metricas Florida ---')
n = len(y)
auc_n = roc_auc_score(y, pn_all)
auc_x = roc_auc_score(y, px_all)
r_n, _ = pearsonr(pn_all, chl)
r_x, _ = pearsonr(px_all, chl)
print(f'HABNet1Dv2: AUC={auc_n:.4f}, Pearson r={r_n:.4f}')
print(f'XGBoost:    AUC={auc_x:.4f}, Pearson r={r_x:.4f}')

# ========== Guardar modelos ==========
print('\n--- Guardando modelos ---')
out = os.path.join(BASE, 'modelos', 'florida')
os.makedirs(out, exist_ok=True)
torch.save(best.state_dict(), os.path.join(out, 'best_model.pth'))
pickle.dump(sc, open(os.path.join(out, 'scaler.pkl'), 'wb'))
xgb_m.save_model(os.path.join(out, 'xgb_model.json'))
print('OK')

print('\n=== COMPLETADO ===')
