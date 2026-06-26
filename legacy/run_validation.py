import numpy as np, pandas as pd, os, pickle, torch, warnings, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r'C:\Users\JC\Desktop\Tesis'; RND = 42
OUT = os.path.join(BASE, 'validacion_mapas')
os.makedirs(OUT, exist_ok=True)
np.random.seed(RND)

S2 = ['B2','B3','B4','B5','B8']
ERA = ['temp_air_2m','solar_radiation','precipitation','wind_speed_10m','wind_u_10m','wind_v_10m','surface_pressure']
ERA_M = [22.5, 18000000, 0.003, 3.5, -0.5, 1.2, 1013.25]
F = ['B2','B3','B4','B5','B8','NDCI','NDVI','FAI','B5_B4_ratio','B3_B2_ratio','B8_B3_ratio'] + ERA
T = 'es_floracion'

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

# ========== Cargar modelos entrenados ==========
print('Cargando modelos...')
mod_dir = os.path.join(BASE, 'modelos', 'florida')
sc = pickle.load(open(os.path.join(mod_dir, 'scaler.pkl'), 'rb'))
xgb_m = xgb.XGBClassifier()
xgb_m.load_model(os.path.join(mod_dir, 'xgb_model.json'))

# ========== Cargar y procesar datos ==========
print('Cargando datos...')
df = pd.read_csv(os.path.join(BASE, 'datasets', 'dataset_entrenamiento_completo_2023_2026_clean.csv'))
df['fecha'] = pd.to_datetime(df['fecha'])
fl = prep(df)
X = fl[F].values.astype('float32')
y = fl[T].values.astype('float32')
chl = fl['clorofila_ugl'].values
origen = fl['origen'].values

# Predecir
Xs = sc.transform(X.astype('float64')).astype('float32')
px = xgb_m.predict_proba(Xs)[:, 1]

# ========== APERTURA: antes vs despues ==========
# Cargar modelo VIEJO (backup) para comparacion
try:
    old_regions = json.load(open(os.path.join(BASE, 'backup', 'validacion_mapas', 'resumen_validacion_regiones.json')))
except FileNotFoundError:
    old_regions = {'regiones': []}
    print('Backup no encontrado, se omite comparacion antes/despues')

# ========== GRAFICO 1: Correlacion prob vs clorofila (nuevo) ==========
print('Grafico 1: Scatter prob vs clorofila...')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

regions_plot = [
    ('Okeechobee', 'florida', '#2196F3'),
    ('Golfo Fonseca', 'golfo_fonseca', '#4CAF50'),
    ('Lago de Yojoa', 'lago_de_yojoa', '#FF9800')
]

for ax, (name, rkey, color) in zip(axes, regions_plot):
    mask = origen == rkey
    x_c, y_c = chl[mask], px[mask]
    ax.scatter(x_c, y_c, c=color, alpha=0.5, edgecolors='k', linewidth=0.3, s=30)
    r, p = pearsonr(x_c, y_c)
    r_s, _ = spearmanr(x_c, y_c)
    
    # Linea de regresion
    z = np.polyfit(x_c, y_c, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', lw=2, alpha=0.7)
    
    ax.set_xlabel('Clorofila in-situ (ug/L)', fontsize=11)
    ax.set_ylabel('Probabilidad HAB', fontsize=11)
    ax.set_title(f'{name}\nPearson r={r:.4f} (p={p:.2e})', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(x_c.max() + 5, 60))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'validacion_chl_vs_prob.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  OK')

# ========== GRAFICO 2: AUC por region (comparacion antes/despues) ==========
print('Grafico 2: AUC por region...')
regions_order = ['florida', 'lago_de_yojoa', 'cajon', 'golfo_fonseca']
region_labels = ['Florida', 'Lago de\nYojoa', 'Cajon', 'Golfo\nFonseca']
n_muestras = []

auc_new = []
for r in regions_order:
    mask = origen == r
    y_r, p_r = y[mask], px[mask]
    n_muestras.append(mask.sum())
    if mask.sum() >= 2 and y_r.sum() > 0 and y_r.sum() < len(y_r):
        auc_new.append(roc_auc_score(y_r, p_r))
    else:
        auc_new.append(0)

# Metricas anteriores (del backup)
old_auc_map = {}
if old_regions and 'regiones' in old_regions:
    for reg in old_regions['regiones']:
        if 'xgb_auc' in reg:
            rname = reg['region']
            if rname == 'Okeechobee': rname = 'florida'
            elif rname == 'TampaBay': rname = 'florida'
            elif rname == 'Lago de Yojoa': rname = 'lago_de_yojoa'
            elif rname == 'Cajon': rname = 'cajon'
            elif rname == 'Golfo_Fonseca': rname = 'golfo_fonseca'
            if rname not in old_auc_map:
                old_auc_map[rname] = []
            old_auc_map[rname].append(reg['xgb_auc'])

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(regions_order))
w = 0.3

# Valores anteriores
auc_old = []
for r in regions_order:
    if r in old_auc_map:
        auc_old.append(max(old_auc_map[r]))
    else:
        auc_old.append(0)

bars1 = ax.bar(x_pos - w/2, auc_old, w, label='Anterior (sin DA)', color='#B0BEC5', edgecolor='#78909C', linewidth=1)
bars2 = ax.bar(x_pos + w/2, auc_new, w, label='Actual (con DA)', color='#26A69A', edgecolor='#00796B', linewidth=1)

for i, (v1, v2) in enumerate(zip(auc_old, auc_new)):
    ax.text(x_pos[i] - w/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=8, color='#546E7A')
    ax.text(x_pos[i] + w/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=8, color='#00796B')

ax.set_xticks(x_pos)
ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(region_labels, n_muestras)], fontsize=10)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Comparacion AUC por region: Antes vs Despues', fontsize=14)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'validacion_auc_por_region.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  OK')

# ========== GRAFICO 3: ROC curva (hold-out) ==========
print('Grafico 3: ROC Hold-Out...')
# Recrear hold-out split con DA augmentation (per-region shifts)
regions_hn = ['lago_de_yojoa', 'cajon', 'golfo_fonseca']
fl_mask = (fl['origen'] == 'florida').values
fl_mean_spec = X[fl_mask, :11].mean(axis=0)
X_aug = X.copy()
y_aug = y.copy()
for region in regions_hn:
    r_mask = (fl['origen'] == region).values
    if r_mask.sum() == 0: continue
    r_mean_spec = X[r_mask, :11].mean(axis=0)
    shift_vec = fl_mean_spec - r_mean_spec
    X_r_aug = X[r_mask].copy()
    X_r_aug[:, :11] += shift_vec
    X_aug = np.vstack([X_aug, X_r_aug])
    y_aug = np.concatenate([y_aug, y[r_mask]])

# HABNet1Dv2 inference
class HABNet1Dv2(torch.nn.Module):
    def __init__(self, nf, h=128, nr=2, d=0.2):
        super().__init__()
        self.input_proj = torch.nn.Sequential(torch.nn.Linear(nf,h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(d))
        self.res_blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(h,h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(d), torch.nn.Linear(h,h), torch.nn.BatchNorm1d(h)) for _ in range(nr)])
        self.se = torch.nn.Sequential(torch.nn.Linear(h,max(h//4,8)), torch.nn.ReLU(), torch.nn.Linear(max(h//4,8),h), torch.nn.Sigmoid())
        self.classifier = torch.nn.Sequential(torch.nn.Linear(h,h//2), torch.nn.GELU(), torch.nn.Dropout(d*.5), torch.nn.Linear(h//2,1))
    def forward(self, x):
        x = self.input_proj(x)
        for b in self.res_blocks: x = x + b(x)
        return self.classifier(x * self.se(x))

nn_m = HABNet1Dv2(len(F)).to(DEVICE)
nn_m.load_state_dict(torch.load(os.path.join(mod_dir, 'best_model.pth'), map_location=DEVICE))
nn_m.eval()

X_tr, X_te, y_tr, y_te = train_test_split(X_aug, y_aug, test_size=.2, random_state=RND, stratify=y_aug)
X_te_s = sc.transform(X_te.astype('float64')).astype('float32')

with torch.no_grad():
    pn_te = torch.sigmoid(nn_m(torch.tensor(X_te_s, device=DEVICE))).cpu().numpy().flatten()
px_te = xgb_m.predict_proba(X_te_s)[:, 1]

fig, ax = plt.subplots(figsize=(8, 7))
for n, c, p in [('HABNet1Dv2', '#1976D2', pn_te), ('XGBoost', '#D32F2F', px_te)]:
    fpr, tpr, _ = roc_curve(y_te, p)
    auc_val = roc_auc_score(y_te, p)
    ax.plot(fpr, tpr, color=c, lw=2.5, label=f'{n} (AUC={auc_val:.4f})')

ax.plot([0,1], [0,1], 'k--', alpha=0.5, label='Azar')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
ax.set_title('Curva ROC - Hold-Out (20% datos no vistos)', fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'roc_holdout.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  OK')

# ========== GRAFICO 4: Distribucion de probabilidades por region ==========
print('Grafico 4: Distribucion probabilidades...')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (rname, rkey) in zip(axes, [('Florida', 'florida'), ('Yojoa', 'lago_de_yojoa'), ('Cajon', 'cajon'), ('Golfo Fonseca', 'golfo_fonseca')]):
    mask = origen == rkey
    if mask.sum() < 2:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'{rname} (n={mask.sum()})', fontsize=12)
        continue
    p_r = px[mask]
    y_r = y[mask]
    
    ax.hist(p_r[y_r==0], bins=30, alpha=0.6, color='#4CAF50', label='No HAB', density=True)
    ax.hist(p_r[y_r==1], bins=30, alpha=0.6, color='#F44336', label='HAB', density=True)
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.5, label='Umbral 0.5')
    ax.set_xlabel('Probabilidad HAB', fontsize=10)
    ax.set_ylabel('Densidad', fontsize=10)
    ax.set_title(f'{rname} (n={mask.sum()})', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'distribucion_probabilidades.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  OK')

# ========== GRAFICO 5: Tabla resumen ==========
print('Grafico 5: Tabla resumen...')
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

regions_all = ['florida', 'lago_de_yojoa', 'cajon', 'golfo_fonseca']
table_data = []
for r in regions_all:
    mask = origen == r
    n = mask.sum()
    if n < 2:
        table_data.append([r, str(n), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
        continue
    y_r, p_r, chl_r = y[mask], px[mask], chl[mask]
    hab_pct = y_r.mean() * 100
    auc_v = roc_auc_score(y_r, p_r) if y_r.sum() > 0 and y_r.sum() < n else 0
    r_p, p_v = pearsonr(p_r, chl_r)
    r_s, _ = spearmanr(p_r, chl_r)
    mean_prob = p_r.mean()
    table_data.append([r, str(n), f'{hab_pct:.1f}%', f'{auc_v:.4f}', f'{r_p:.4f}', f'{r_s:.4f}', f'{mean_prob:.4f}'])

columns = ['Region', 'Muestras', '% HAB', 'AUC-ROC', 'Pearson r', 'Spearman rho', 'Prob media']
table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for j, col in enumerate(columns):
    table[0, j].set_facecolor('#263238')
    table[0, j].set_text_props(color='white', weight='bold')

plt.savefig(os.path.join(OUT, 'tabla_resumen.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  OK')

# ========== GRAFICO 6: Correlacion por region (barras) ==========
print('Grafico 6: Correlacion por region...')
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(regions_all))
w = 0.35

pearson_vals = []
spearman_vals = []
for r in regions_all:
    mask = origen == r
    n = mask.sum()
    if n < 2:
        pearson_vals.append(0)
        spearman_vals.append(0)
        continue
    p_r, chl_r = px[mask], chl[mask]
    r_p, _ = pearsonr(p_r, chl_r)
    r_s, _ = spearmanr(p_r, chl_r)
    pearson_vals.append(r_p)
    spearman_vals.append(r_s)

bars1 = ax.bar(x - w/2, pearson_vals, w, label='Pearson r', color='#42A5F5', edgecolor='#1E88E5')
bars2 = ax.bar(x + w/2, spearman_vals, w, label='Spearman rho', color='#EF5350', edgecolor='#E53935')

for bar, val in zip(bars1, pearson_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, spearman_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(['Florida', 'Yojoa', 'Cajon', 'Golfo\nFonseca'], fontsize=11)
ax.set_ylabel('Correlacion', fontsize=12)
ax.set_title('Correlacion Probabilidad HAB vs Clorofila in-situ', fontsize=13)
ax.axhline(0, color='k', lw=0.5)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'correlacion_por_region.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  OK')

# ========== GRAFICO 7: Yojoa time series ==========
print('Grafico 7: Yojoa time series...')
mask_yj = origen == 'lago_de_yojoa'
df_yj = fl[mask_yj].copy()
df_yj['prob_xgb'] = px[mask_yj]
df_yj = df_yj.sort_values('fecha')

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df_yj['fecha'], df_yj['prob_xgb'], 'o-', color='#FF9800', lw=2, markersize=4, label='Probabilidad HAB')
ax1.axhline(0.5, color='k', linestyle='--', alpha=0.4)
ax1.set_ylabel('Probabilidad HAB', fontsize=12, color='#FF9800')
ax1.tick_params(axis='y', labelcolor='#FF9800')
ax1.set_ylim(-0.05, 1.05)

ax2 = ax1.twinx()
ax2.bar(df_yj['fecha'], df_yj['clorofila_ugl'], alpha=0.3, color='#4CAF50', label='Clorofila (ug/L)')
ax2.set_ylabel('Clorofila (ug/L)', fontsize=12, color='#4CAF50')
ax2.tick_params(axis='y', labelcolor='#4CAF50')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
ax1.set_title('Lago de Yojoa: Probabilidad HAB vs Clorofila (Serie Temporal)', fontsize=13)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'yojoa_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  OK')

print(f'\nGraficos guardados en: {OUT}')
print('  - validacion_chl_vs_prob.png')
print('  - validacion_auc_por_region.png')
print('  - roc_holdout.png')
print('  - distribucion_probabilidades.png')
print('  - tabla_resumen.png')
print('  - correlacion_por_region.png')
print('  - yojoa_time_series.png')
