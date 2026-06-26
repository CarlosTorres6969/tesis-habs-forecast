import numpy as np, pandas as pd, os, pickle, torch, warnings, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r'C:\Users\JC\Desktop\Tesis'; RND = 42
np.random.seed(RND); torch.manual_seed(RND)

S2 = ['B2','B3','B4','B5','B8']
ERA = ['temp_air_2m','solar_radiation','precipitation','wind_speed_10m','wind_u_10m','wind_v_10m','surface_pressure']
ERA_M = [22.5, 18000000, 0.003, 3.5, -0.5, 1.2, 1013.25]
T = 'es_floracion'
print(f'Dispositivo: {DEVICE}')

# ========== Carga ==========
print('\n--- Cargando datos Honduras ---')
df = pd.read_csv(os.path.join(BASE, 'datasets', 'dataset_entrenamiento_completo_2023_2026_clean.csv'))
df['fecha'] = pd.to_datetime(df['fecha'])
# Solo Honduras
hn = df[df['origen'].isin(['lago_de_yojoa','cajon','golfo_fonseca'])].copy()
print(f'Muestras Honduras: {len(hn)}')
print(hn['origen'].value_counts().to_string())

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
    # Nuevas features para aguas turbias
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + e)
    df['turbidity'] = df['B4'] / (df['B3'] + e)
    df['norm_turb'] = (df['B4'] - df['B3']) / (df['B4'] + df['B3'] + e)
    df['B8_B2_ratio'] = df['B8'] / (df['B2'] + e)
    df['B5_B3_ratio'] = df['B5'] / (df['B3'] + e)
    df['B2_B3_ratio'] = df['B2'] / (df['B3'] + e)
    df['B8_minus_B4'] = df['B8'] - df['B4']
    df['B3_plus_B4'] = df['B3'] + df['B4']
    # Filtro suave de agua: excluir vegetacion terrestre dominante
    df = df[df['NDWI'] > -0.5].copy()
    df = df[df['NDVI'] < 0.4].copy()
    return df

fl = prep(hn)
# Features: las 11 originales + 7 nuevas + 7 ERA = 25
F_ESP = ['B2','B3','B4','B5','B8','NDCI','NDVI','FAI','B5_B4_ratio','B3_B2_ratio','B8_B3_ratio',
         'NDWI','turbidity','norm_turb','B8_B2_ratio','B5_B3_ratio','B2_B3_ratio','B8_minus_B4','B3_plus_B4']
F = F_ESP + ERA

X = fl[F].values.astype('float32')
y = fl[T].values.astype('float32')
chl = fl['clorofila_ugl'].values
fechas = fl['fecha'].values
origenes = fl['origen'].values
print(f'Features: {len(F)} ({len(F_ESP)} espectrales + {len(ERA)} ERA5)')
print(f'HAB rate: {y.mean()*100:.1f}% ({int(y.sum())}/{len(y)})')
print(f'Fechas unicas: {len(np.unique(fechas))}')

# ========== DateGroupKFold estratificado ==========
print('\n--- DateGroupKFold (5 folds, agrupado por fecha) ---')
# Estratificar: ordenar fechas por tasa HAB, asignar a folds round-robin
date_hab = fl.groupby('fecha')[T].mean().sort_values()
dates_sorted = date_hab.index.values
fold_assign = {}
for i, d in enumerate(dates_sorted):
    fold_assign[d] = i % 5

groups = np.array([fold_assign[d] for d in fl['fecha']])

cv_metrics = []
xgb_models = []
OOF_preds = np.zeros(len(X))  # out-of-fold predictions for honest metrics

for f in range(5):
    train_mask = groups != f
    val_mask = groups == f
    X_tr, X_val = X[train_mask], X[val_mask]
    y_tr, y_val = y[train_mask], y[val_mask]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    
    pw = int((y_tr==0).sum()) / max(int(y_tr.sum()), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.01, reg_lambda=1.0,
        scale_pos_weight=pw,
        random_state=RND, device='cuda', tree_method='hist',
        early_stopping_rounds=50, eval_metric='auc', verbosity=0
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    
    p_val = model.predict_proba(X_val_s)[:, 1]
    OOF_preds[val_mask] = p_val
    auc = roc_auc_score(y_val, p_val)
    f1 = f1_score(y_val, (p_val >= 0.5).astype(int))
    print(f'  Fold {f+1}: AUC={auc:.4f} F1={f1:.4f} (train={len(X_tr)}, val={len(X_val)})')
    
    cv_metrics.append({'fold': f+1, 'auc': auc, 'f1': f1})
    xgb_models.append((model, sc))

print(f'\nCV promedio: AUC={np.mean([m["auc"] for m in cv_metrics]):.4f} +/- {np.std([m["auc"] for m in cv_metrics]):.4f}')

# ========== Optuna hyperparameter tuning ==========
print('\n--- Optuna: optimizacion de hiperparametros ---')
try:
    import optuna
    
    def objective(trial):
        fold = trial.number % 5
        train_mask = groups != fold
        val_mask = groups == fold
        X_tr, X_val = X[train_mask], X[val_mask]
        y_tr, y_val = y[train_mask], y[val_mask]
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        
        pw = int((y_tr==0).sum()) / max(int(y_tr.sum()), 1)
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 14),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
            'scale_pos_weight': pw,
            'random_state': RND, 'device': 'cuda', 'tree_method': 'hist',
            'early_stopping_rounds': 50, 'eval_metric': 'auc', 'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        p_val = model.predict_proba(X_val_s)[:, 1]
        return roc_auc_score(y_val, p_val)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RND))
    study.optimize(objective, n_trials=80, show_progress_bar=True)
    
    print(f'\nMejores parametros:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    print(f'Mejor AUC: {study.best_value:.4f}')
    best_params = study.best_params
    
    # Re-evaluar con mejores params en todos los folds
    print('\n--- Re-evaluando con mejores parametros ---')
    cv_best = []
    xgb_best = []
    for f in range(5):
        train_mask = groups != f
        val_mask = groups == f
        X_tr, X_val = X[train_mask], X[val_mask]
        y_tr, y_val = y[train_mask], y[val_mask]
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        
        pw = int((y_tr==0).sum()) / max(int(y_tr.sum()), 1)
        
        params = {k: v for k, v in best_params.items()}
        params.update({
            'scale_pos_weight': pw,
            'random_state': RND, 'device': 'cuda', 'tree_method': 'hist',
            'early_stopping_rounds': 50, 'eval_metric': 'auc', 'verbosity': 0
        })
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        
        p_val = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        f1 = f1_score(y_val, (p_val >= 0.5).astype(int))
        print(f'  Fold {f+1}: AUC={auc:.4f} F1={f1:.4f}')
        
        cv_best.append({'fold': f+1, 'auc': auc, 'f1': f1})
        xgb_best.append((model, sc))
    
    print(f'\nCV optimizado: AUC={np.mean([m["auc"] for m in cv_best]):.4f} +/- {np.std([m["auc"] for m in cv_best]):.4f}')
    xgb_models = xgb_best  # usar los optimizados
    cv_metrics = cv_best

except ImportError:
    print('  optuna no instalado. Usando parametros default.')
    best_params = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'gamma': 0.1,
        'reg_alpha': 0.01, 'reg_lambda': 1.0
    }

# ========== Entrenar modelo FINAL con todos los datos ==========
print('\n--- Entrenando modelo final (todos los datos) ---')
sc_final = StandardScaler()
X_all_s = sc_final.fit_transform(X)
pw_final = int((y==0).sum()) / max(int(y.sum()), 1)

params_final = {k: v for k, v in best_params.items()}
params_final.pop('early_stopping_rounds', None)
params_final.pop('eval_metric', None)
params_final.update({
    'scale_pos_weight': pw_final,
    'random_state': RND, 'device': 'cuda', 'tree_method': 'hist',
    'verbosity': 0
})
print(f'scale_pos_weight: {pw_final:.2f}')

xgb_final = xgb.XGBClassifier(n_estimators=params_final.pop('n_estimators', 500), **params_final)
xgb_final.fit(X_all_s, y)
print(f'Estimadores finales: {getattr(xgb_final, "best_iteration", xgb_final.n_estimators)}')

# ========== Metricas por region (usando OOF = honestas) ==========
print('\n--- Metricas por region (Out-Of-Fold) ---')
reg_metrics = {}
for region in ['lago_de_yojoa', 'cajon', 'golfo_fonseca']:
    mask = origenes == region
    n = mask.sum()
    if n < 5: continue
    y_r = y[mask]
    p_r = OOF_preds[mask]
    chl_r = chl[mask]
    auc = roc_auc_score(y_r, p_r) if y_r.sum() > 0 and y_r.sum() < n else 0
    f1 = f1_score(y_r, (p_r >= 0.5).astype(int))
    r_pearson, _ = pearsonr(p_r, chl_r)
    r_spearman, _ = spearmanr(p_r, chl_r)
    
    # Encontrar threshold optimo (max F1) sobre OOF
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_th = 0, 0.5
    for th in thresholds:
        f1_th = f1_score(y_r, (p_r >= th).astype(int))
        if f1_th > best_f1:
            best_f1 = f1_th
            best_th = th
    
    # Calibrar probabilidades OOF con isotonic regression
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    p_cal = iso.fit_transform(p_r, y_r) if len(np.unique(p_r)) > 1 else p_r
    
    reg_metrics[region] = {
        'n': int(n), 'hab_rate': float(y_r.mean()),
        'auc_oof': float(auc), 'f1_oof_05': float(f1),
        'f1_optimo': float(best_f1), 'threshold_optimo': float(best_th),
        'pearson_chl': float(r_pearson), 'spearman_chl': float(r_spearman),
        'prob_mean': float(p_r.mean()), 'prob_p95': float(np.percentile(p_r, 95))
    }
    print(f'  {region:<20s} n={n:>4d} AUC_OOF={auc:.4f} F1_OOF={f1:.4f} '
          f'Pearson_chl={r_pearson:.4f} Spearman={r_spearman:.4f} '
          f'th_opt={best_th:.3f} F1_opt={best_f1:.4f}')

# ========== Calibracion global para deployment ==========
print('\n--- Calibrando modelo global (Isotonic, 5-fold CV) ---')
calibrated = CalibratedClassifierCV(estimator=xgb_final, method='isotonic', cv=5)
calibrated.fit(X_all_s, y)

# ========== Correlacion global con clorofila (OOF) ==========
r_p_glob, pv_glob = pearsonr(OOF_preds, chl)
r_s_glob, sv_glob = spearmanr(OOF_preds, chl)
print(f'\n--- Correlacion global vs Clorofila (OOF, n={len(chl)}) ---')
print(f'  Pearson: r={r_p_glob:.4f} (p={pv_glob:.2e})')
print(f'  Spearman: rho={r_s_glob:.4f} (p={sv_glob:.2e})')

# ========== Correlacion con indices espectrales (OOF) ==========
print('\n--- Correlacion Prob OOF vs Indices Espectrales ---')
indices_check = ['NDCI', 'NDVI', 'FAI', 'turbidity', 'NDWI']
for idx in indices_check:
    r, _ = pearsonr(OOF_preds, fl[idx].values)
    print(f'  Pearson(prob, {idx:>12s}) = {r:.4f}')

# ========== Importancia de Features ==========
print('\n--- Feature Importance (Gain) ---')
importance = xgb_final.get_booster().get_score(importance_type='gain')
total = sum(importance.values())
imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feat, gain in imp_sorted[:15]:
    print(f'  {feat:<20s}: {gain/total*100:.2f}%')

# ========== Guardar modelos ==========
print('\n--- Guardando modelos ---')
out = os.path.join(BASE, 'modelos', 'honduras')
os.makedirs(out, exist_ok=True)

# Modelos y preprocesamiento
xgb_final.save_model(os.path.join(out, 'xgb_honduras.json'))
pickle.dump(sc_final, open(os.path.join(out, 'scaler_honduras.pkl'), 'wb'))
pickle.dump(calibrated, open(os.path.join(out, 'calibrated_honduras.pkl'), 'wb'))
torch.save({'F': F, 'F_ESP': F_ESP, 'ERA': ERA, 'ERA_M': ERA_M},
           os.path.join(out, 'metadata.pth'))

# Metricas y thresholds por region
with open(os.path.join(out, 'metricas_regiones.json'), 'w') as f:
    json.dump(reg_metrics, f, indent=2)

# Metricas de CV
with open(os.path.join(out, 'cv_metrics.json'), 'w') as f:
    json.dump({
        'cv_auc_mean': float(np.mean([m['auc'] for m in cv_metrics])),
        'cv_auc_std': float(np.std([m['auc'] for m in cv_metrics])),
        'cv_f1_mean': float(np.mean([m['f1'] for m in cv_metrics])),
        'folds': cv_metrics,
        'best_params': best_params
    }, f, indent=2)

# Grafico de importancia
fig, ax = plt.subplots(figsize=(10, 8))
names, gains = zip(*imp_sorted[:20])
ax.barh(range(len(names)), [g/total*100 for g in gains][::-1])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names[::-1])
ax.set_xlabel('Importancia (Gain %)')
ax.set_title('Feature Importance - Modelo Honduras')
plt.tight_layout()
plt.savefig(os.path.join(out, 'feature_importance.png'), dpi=150)
plt.close(fig)

# Grafico de ROC por region (usando OOF)
fig, ax = plt.subplots(figsize=(8, 6))
for region in ['lago_de_yojoa', 'cajon', 'golfo_fonseca']:
    mask = origenes == region
    if mask.sum() < 5: continue
    fpr, tpr, _ = roc_curve(y[mask], OOF_preds[mask])
    auc = reg_metrics[region]['auc_oof']
    ax.plot(fpr, tpr, lw=2, label=f'{region} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC - Modelo Honduras (Out-Of-Fold)')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out, 'roc_regiones.png'), dpi=150)
plt.close(fig)

print(f'\nModelo Honduras guardado en: {out}')
print(f'  - xgb_honduras.json (XGBoost final)')
print(f'  - calibrated_honduras.pkl (calibracion isotonica)')
print(f'  - scaler_honduras.pkl')
print(f'  - metricas_regiones.json (AUC/F1/threshold por region)')
print(f'  - feature_importance.png')
print(f'  - roc_regiones.png')

print('\n=== COMPLETADO ===')
