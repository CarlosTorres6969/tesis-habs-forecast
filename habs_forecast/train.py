"""
train.py — Entrenamiento por horizonte + validacion LOWBO (lago<->lago, costa<->costa).

Decisiones (Fase 1):
  - MODELOS SEPARADOS por horizonte h en {0,1,3,5,7}.
  - Salida hibrida: regresion de log(chl) -> clase de alerta por umbral.
  - Validacion reina: Leave-One-Water-Body-Out DENTRO del grupo ecologico.
  - Baseline obligatorio: persistencia (ultimo target conocido <= t0). El modelo debe superarlo.
  - Metricas: regresion RMSE(log)/MAE/R2 ; alerta Recall/PR-AUC/F1 (eventos raros).

NO usa shuffle-CV (eso inflo las metricas del pipeline anterior). Reporta numeros honestos.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             average_precision_score, recall_score, f1_score)
import config as C

PAIRS = os.path.join(C.DIR_PAIRS, "pairs_forecast.csv")
REPORT = os.path.join(C.DIR_REPORTS, "lowbo_metrics.json")

SPECTRAL = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
# autorregresivos: trayectoria reciente de clorofila (causal <= t0). Backbone del pronostico.
AUTOREG = ["log_chl_t0", "chl_lag3", "chl_lag7", "chl_roll7", "chl_trend7"]
# dinamica temporal y estacionalidad (causales): se construyen en match_pairs y se PROBARON
# con validacion anidada controlada (con vs sin), pero NO mejoran el skill de forma robusta
# (lavado: +5d mejor pero +3d/+7d peor) -> NO se adoptan. Se dejan definidas y como columnas
# (documentan el experimento / re-test futuro), fuera del set de features del modelo.
DYNAMICS = ["chl_rate3", "chl_accel", "days_since_obs", "chl_roll14", "chl_roll30",
            "chl_std14", "chl_max14"]
SEASONAL = ["doy_sin", "doy_cos", "chl_clim", "chl_anom"]
# drivers meteorologicos ERA5 (en t0 + media movil causal 7d)
ERA5 = ["temp_air_2m", "solar_radiation", "precipitation", "wind_speed_10m", "surface_pressure",
        "temp_air_2m_roll7", "solar_radiation_roll7", "precipitation_roll7", "wind_speed_10m_roll7"]
# contexto in-situ (solo Florida, NaN en Honduras -> XGBoost nativo):
#   fosforo (lento) + calidad de agua (temp agua, OD, pH, turbidez, conductividad, Secchi, amonio)
NUTRIENTS = ["tp_context"]
WATERQUAL = ["water_temp", "do_mgl", "ph", "turbidity_insitu", "spec_cond", "secchi", "ammonia"]
FEATURES = SPECTRAL + AUTOREG + ERA5 + NUTRIENTS + WATERQUAL   # DYNAMICS/SEASONAL probadas, no adoptadas

# Sets de features POR HORIZONTE (de select_features_per_horizon.py, ablacion OOS).
# Cada horizonte usa solo las familias que maximizan su skill (corto=optico/autorreg;
# largo=meteo/susceptibilidad in-situ). Si falta el json -> usa FEATURES completo.
_FAMILY = {"AUTOREG": AUTOREG, "ERA5": ERA5, "SPECTRAL": SPECTRAL,
           "INSITU": NUTRIENTS + WATERQUAL}


def get_features(group, horizon, available):
    import json
    path = os.path.join(C.DIR_REPORTS, "feature_sets.json")
    if os.path.exists(path):
        sets = json.load(open(path))
        node = sets.get(group, {}).get(str(horizon))
        if node:
            feats = []
            for fam in node["families"]:
                feats += _FAMILY.get(fam, [])
            return [f for f in feats if f in available]
    return [f for f in FEATURES if f in available]


def _model():
    # reg_lambda=3.0: regularizacion algo mayor (mejor en afinado OOS; datos modestos).
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                        random_state=C.RANDOM_STATE, n_jobs=4)


def _persistence(train, test):
    """Baseline de PERSISTENCIA real: chl(t0+h) ~= chl(t0). Proyecta el ultimo valor
    conocido en t0. Es el rival honesto que un pronostico util debe superar."""
    if "log_chl_t0" in test:
        return test["log_chl_t0"].values
    return np.full(len(test), train["log_chl_target"].mean())


def _eval(y_log, yhat_log, thr=C.THRESHOLDS["moderate"]):
    rmse = float(np.sqrt(mean_squared_error(y_log, yhat_log)))
    mae = float(mean_absolute_error(y_log, yhat_log))
    r2 = float(r2_score(y_log, yhat_log)) if len(set(y_log)) > 1 else float("nan")
    # clase de alerta por umbral sobre chl reconstruida.
    # thr puede ser escalar (absoluto) o array por-muestra (relativo por cuerpo).
    y_chl = np.expm1(y_log); yhat_chl = np.expm1(yhat_log)
    thr = np.asarray(thr)
    y_cls = (y_chl >= thr).astype(int); yhat_cls = (yhat_chl >= thr).astype(int)
    out = {"rmse_log": rmse, "mae_log": mae, "r2": r2, "n": int(len(y_log)),
           "pos_rate": float(y_cls.mean())}
    if y_cls.sum() > 0 and y_cls.sum() < len(y_cls):
        # alerta derivada de la regresion (regresion->umbral): tiende a subestimar picos
        out["recall_reg"] = float(recall_score(y_cls, yhat_cls, zero_division=0))
        out["pr_auc_reg"] = float(average_precision_score(y_cls, yhat_chl))
    return out


def _clf(y):
    """Clasificador directo de ALERTA (cabeza hibrida). scale_pos_weight maneja el
    desbalance de eventos raros -> mejor Recall/PR-AUC que regresion->umbral."""
    from xgboost import XGBClassifier
    pos = max(int(np.sum(y)), 1); neg = max(len(y) - pos, 1)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=neg / pos, eval_metric="aucpr",
                         random_state=C.RANDOM_STATE, n_jobs=4)


def _eval_clf(y_cls, proba):
    """Metricas del clasificador directo de alerta."""
    out = {}
    if y_cls.sum() > 0 and y_cls.sum() < len(y_cls):
        pred = (proba >= 0.5).astype(int)
        out["recall_clf"] = float(recall_score(y_cls, pred, zero_division=0))
        out["f1_clf"] = float(f1_score(y_cls, pred, zero_division=0))
        out["pr_auc_clf"] = float(average_precision_score(y_cls, proba))
    return out


def _add_clf(tr, te, metrics, feats):
    """Entrena el clasificador directo de alerta y agrega sus metricas (cabeza hibrida)."""
    if tr["hab_target"].nunique() > 1 and len(te):
        c = _clf(tr["hab_target"].values).fit(tr[feats], tr["hab_target"])
        proba = c.predict_proba(te[feats])[:, 1]
        metrics.update(_eval_clf(te["hab_target"].values, proba))
    return metrics


def lowbo(df, group):
    sub = df[df["group"] == group]
    bodies = sorted(sub["water_body"].unique())
    res = {}
    for h in C.HORIZONS:
        dh = sub[sub["horizon"] == h]
        if dh["water_body"].nunique() < 2:
            continue
        feats = get_features(group, h, dh.columns)      # set de features por horizonte
        per_body = {}
        for held in bodies:
            tr = dh[dh["water_body"] != held]
            te = dh[dh["water_body"] == held]
            if len(te) < 10 or len(tr) < 30:
                continue
            m = _model().fit(tr[feats], tr["log_chl_target"])
            yhat = m.predict(te[feats])
            base = _persistence(tr, te)
            thr = te["thr_body"].values if "thr_body" in te else C.THRESHOLDS["moderate"]
            mm = _eval(te["log_chl_target"].values, yhat, thr)
            _add_clf(tr, te, mm, feats)
            per_body[held] = {
                "model": mm,
                "baseline": _eval(te["log_chl_target"].values, base, thr),
            }
        if per_body:
            res[h] = per_body
    return res


def walk_forward(df, group, train_frac=0.7):
    """Validacion temporal DENTRO de cada cuerpo (modo operativo real):
    entrenar con el pasado, predecir el futuro. Split por fecha_t0 al 70%."""
    sub = df[df["group"] == group]
    res = {}
    for h in C.HORIZONS:
        dh = sub[sub["horizon"] == h]
        feats = get_features(group, h, dh.columns)      # set de features por horizonte
        per_body = {}
        for wb, g in dh.groupby("water_body"):
            g = g.sort_values("fecha_t0")
            if len(g) < 40:
                continue
            cut = g["fecha_t0"].quantile(train_frac)
            tr, te = g[g["fecha_t0"] <= cut], g[g["fecha_t0"] > cut]
            if len(te) < 8 or len(tr) < 25:
                continue
            m = _model().fit(tr[feats], tr["log_chl_target"])
            yhat = m.predict(te[feats])
            base = _persistence(tr, te)
            thr = te["thr_body"].values if "thr_body" in te else C.THRESHOLDS["moderate"]
            mm = _eval(te["log_chl_target"].values, yhat, thr)
            _add_clf(tr, te, mm, feats)
            per_body[wb] = {"model": mm,
                            "baseline": _eval(te["log_chl_target"].values, base, thr)}
        if per_body:
            res[h] = per_body
    return res


def _print_block(r):
    for h, bodies in r.items():
        print(f"\n  Horizonte +{h} d:")
        for held, m in bodies.items():
            mm, bb = m["model"], m["baseline"]
            print(f"    test={held:14s} n={mm['n']:>4} | "
                  f"RMSE_log model={mm['rmse_log']:.3f} base={bb['rmse_log']:.3f} | "
                  f"ALERTA clf Recall={mm.get('recall_clf', float('nan')):.2f} "
                  f"PR-AUC={mm.get('pr_auc_clf', float('nan')):.2f} | "
                  f"(reg->umbral Recall={mm.get('recall_reg', float('nan')):.2f})")


def main():
    global FEATURES
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])
    FEATURES = [f for f in FEATURES if f in df.columns]   # robusto a ERA5 ausente
    print(f"Pares: {len(df)} | cuerpos: {df['water_body'].nunique()} | "
          f"features: {len(FEATURES)} | horizontes: {sorted(df['horizon'].unique())}\n")
    report = {}
    for group in ("freshwater", "marine"):
        print(f"############  GRUPO: {group}  ############")
        print(f"\n--- A) WALK-FORWARD temporal dentro del cuerpo (modo operativo) ---")
        wf = walk_forward(df, group); _print_block(wf)
        print(f"\n--- B) LOWBO {group}<->{group} (generalizacion inter-ecosistema, prueba dura) ---")
        lo = lowbo(df, group); _print_block(lo)
        report[group] = {"walk_forward": wf, "lowbo": lo}
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReporte -> {REPORT}")


if __name__ == "__main__":
    main()
