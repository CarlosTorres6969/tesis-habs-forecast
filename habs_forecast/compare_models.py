"""
compare_models.py — Comparacion de ARQUITECTURAS con la evaluacion robusta (OOS expansivo
+ bootstrap IC95%). Valida empiricamente la eleccion de Fase 1 (modelo simple por volumen).

Modelos tabulares (mismo X):
  - Ridge (lineal, baseline de capacidad)
  - MLP pequeno (sklearn, con escalado)         -> la red neuronal "justa" para N~2600
  - HistGradientBoosting (sklearn)
  - XGBoost (actual)

Criterio: skill de regresion vs persistencia, OOS expansivo por cuerpo, agua dulce (senal)
y costa. MLP/Ridge requieren imputar NaN (ERA5) y escalar; XGB/HistGB manejan NaN nativo.
LSTM/GRU se evaluan aparte (compare_lstm.py) por requerir secuencias regulares.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import config as C
from train import FEATURES, PAIRS
from evaluate_robust import _boot, N_FOLDS, MIN_TRAIN_FRAC


def make_models():
    return {
        "Ridge":   make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
        "MLP":     make_pipeline(SimpleImputer(), StandardScaler(),
                                 MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2,
                                              max_iter=600, early_stopping=True,
                                              random_state=C.RANDOM_STATE)),
        "HistGB":  HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                 max_iter=300, random_state=C.RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                random_state=C.RANDOM_STATE, n_jobs=4),
    }


def oos_predictions(d, model_factory):
    """OOS expansivo por cuerpo; devuelve y, yhat, persist concatenados."""
    ys, yh, yp = [], [], []
    for _, g in d.groupby("water_body"):
        g = g.sort_values("fecha_t0").reset_index(drop=True)
        N = len(g); start = int(N * MIN_TRAIN_FRAC)
        if N - start < N_FOLDS * 4:
            continue
        fold = (N - start) // N_FOLDS
        for k in range(N_FOLDS):
            a = start + k * fold
            b = N if k == N_FOLDS - 1 else a + fold
            tr, te = g.iloc[:a], g.iloc[a:b]
            if len(te) < 3 or len(tr) < 20:
                continue
            m = model_factory()
            m.fit(tr[FEATURES], tr["log_chl_target"])
            ys.append(te["log_chl_target"].values); yh.append(m.predict(te[FEATURES]))
            yp.append(te["log_chl_t0"].values)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(yp)


def _skill(y, yhat, persist):
    rp = np.sqrt(mean_squared_error(y, persist))
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / rp if rp > 0 else np.nan


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    FEATURES[:] = [f for f in FEATURES if f in df.columns]
    models = make_models()
    for group in ("freshwater", "marine"):
        d = df[(df["group"] == group) & (df["horizon"].isin([1, 3, 5, 7]))]
        print(f"\n######  {group}  —  skill vs persistencia (OOS, IC95%)  ######")
        for name, _ in models.items():
            res = oos_predictions(d, lambda n=name: make_models()[n])
            if res is None:
                print(f"  {name:9s}: sin datos"); continue
            y, yh, yp = res
            sk = _boot(_skill, y, yh, yp)
            star = " *" if (sk[1] > 0) else ""        # IC no cruza 0
            print(f"  {name:9s}: skill={sk[0]:+.3f} [{sk[1]:+.3f},{sk[2]:+.3f}]{star}  (n={len(y)})")
    print("\n* = IC95% no cruza 0 (skill significativo). Tabular esperado: XGB/HistGB >= MLP >= Ridge.")


if __name__ == "__main__":
    main()
