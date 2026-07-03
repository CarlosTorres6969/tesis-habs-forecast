"""
tune_xgb.py — Afinado de hiperparametros de XGBoost con la EVALUACION ROBUSTA como criterio.

Criterio: skill de regresion vs persistencia, promedio sobre horizontes 1-7 d, con OOS de
ventana expansiva agrupado (mismo protocolo que evaluate_robust). Evita sobreajustar a un
fold. Se afina sobre AGUA DULCE (donde hay senal); se reporta tambien costa.

Busqueda aleatoria sobre una rejilla compacta (datos modestos -> no exagerar capacidad).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import config as C
from train import FEATURES, PAIRS
from evaluate_robust import N_FOLDS, MIN_TRAIN_FRAC
from sklearn.metrics import mean_squared_error

GRID = {
    "max_depth": [3, 4, 5, 6],
    "n_estimators": [200, 300, 500],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_lambda": [1.0, 3.0],
}
N_TRIALS = 20
RNG = np.random.default_rng(C.RANDOM_STATE)


def oos_skill(d, params):
    """Skill medio vs persistencia (OOS expansivo por cuerpo) para un set de params."""
    skills = []
    for _, g in d.groupby("water_body"):
        g = g.sort_values("fecha_t0").reset_index(drop=True)
        N = len(g); start = int(N * MIN_TRAIN_FRAC)
        if N - start < N_FOLDS * 4:
            continue
        fold = (N - start) // N_FOLDS
        ys, yh, yp = [], [], []
        for k in range(N_FOLDS):
            a = start + k * fold
            b = N if k == N_FOLDS - 1 else a + fold
            tr, te = g.iloc[:a], g.iloc[a:b]
            if len(te) < 3 or len(tr) < 20:
                continue
            m = XGBRegressor(random_state=C.RANDOM_STATE, n_jobs=4, **params)
            m.fit(tr[FEATURES], tr["log_chl_target"])
            ys.append(te["log_chl_target"].values); yh.append(m.predict(te[FEATURES]))
            yp.append(te["log_chl_t0"].values)
        if ys:
            y = np.concatenate(ys); h = np.concatenate(yh); p = np.concatenate(yp)
            rp = np.sqrt(mean_squared_error(y, p))
            if rp > 0:
                skills.append(1 - np.sqrt(mean_squared_error(y, h)) / rp)
    return float(np.mean(skills)) if skills else np.nan


def sample_params():
    return {k: v[int(RNG.integers(len(v)))] for k, v in GRID.items()}


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    FEATURES[:] = [f for f in FEATURES if f in df.columns]
    fresh = df[(df["group"] == "freshwater") & (df["horizon"].isin([1, 3, 5, 7]))]

    baseline = {"max_depth": 4, "n_estimators": 300, "learning_rate": 0.05,
                "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}
    base_skill = oos_skill(fresh, baseline)
    print(f"Baseline actual: skill agua dulce = {base_skill:+.3f}\n")

    seen, results = set(), []
    for _ in range(N_TRIALS):
        p = sample_params()
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        results.append((oos_skill(fresh, p), p))
    results.sort(key=lambda x: -(x[0] if np.isfinite(x[0]) else -9))

    print("=== Top 5 configuraciones (skill agua dulce, OOS expansivo) ===")
    for sk, p in results[:5]:
        ps = ", ".join(f"{k}={v}" for k, v in p.items())
        print(f"  skill={sk:+.3f} | {ps}")

    best_sk, best = results[0]
    print(f"\nMejor: skill={best_sk:+.3f} (baseline {base_skill:+.3f}, "
          f"mejora {best_sk-base_skill:+.3f})")
    print("best_params =", best)


if __name__ == "__main__":
    main()
