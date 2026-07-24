"""
evaluate_intervals.py — VALIDA intervalos de incertidumbre (regresión cuantil) en el TEST INTACTO.

Añade una banda de incertidumbre a cada pronóstico de intensidad (clorofila-a) en vez de un solo
punto: cuantiles P10/P50/P90 (XGBoost objetivo cuantil). Un intervalo P10-P90 honesto debe
CONTENER el valor real ~80% de las veces (cobertura nominal). Aquí se mide la COBERTURA EMPÍRICA
y el ANCHO en el test temporal intacto, con el mismo protocolo anidado (DEV agrupado por grupo,
features elegidas solo en DEV con parsimonia). Si la cobertura ~80% -> los intervalos son fiables.

Salida: artifacts/reports/interval_metrics.json
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import config as C
from train import PAIRS, get_features, _model
from evaluate_nested import MIN_DEV, MIN_TEST, TEST_FRAC, PURGE_DAYS
from temporal_validation import (common_temporal_holdout, purged_tail_split,
                                 temporal_block_bootstrap, conformal_quantile)

OUT = os.path.join(C.DIR_REPORTS, "interval_metrics.json")
QLO, QMID, QHI = 0.10, 0.50, 0.90
NOMINAL = QHI - QLO          # cobertura objetivo del intervalo P10-P90 = 0.80


def _qmodel(alpha):
    """XGBoost de regresión cuantil (mismos hiperparámetros que el punto)."""
    from xgboost import XGBRegressor
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                        random_state=C.RANDOM_STATE, n_jobs=4)


def _coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])
    report = {}
    print("INTERVALOS P10-P90 (cobertura nominal 0.80) — validados en TEST INTACTO\n")
    for group in ("freshwater", "marine"):
        print(f"############  {group}  ############")
        report[group] = {}
        for h in [x for x in C.HORIZONS if x != 0]:
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            DEV, TEST, split = common_temporal_holdout(
                d, test_frac=TEST_FRAC, purge_days=PURGE_DAYS)
            if len(DEV) < MIN_DEV or len(TEST) < MIN_TEST:
                print(f"  +{h}d  (datos insuficientes)"); continue
            feats = get_features(group, h, DEV.columns, required=True)
            # CQR: TRAIN y CALIB también quedan separados por fecha_target, sin solape.
            TRAIN, CALIB, calibration_start = purged_tail_split(
                DEV, calibration_frac=0.25, min_train=30, min_calibration=20)
            if TRAIN.empty or CALIB.empty:
                print(f"  +{h}d  (TRAIN/CALIB insuficiente tras purga)"); continue
            mlo = _qmodel(QLO).fit(TRAIN[feats], TRAIN["log_chl_target"])
            mhi = _qmodel(QHI).fit(TRAIN[feats], TRAIN["log_chl_target"])
            # conformidad en CALIB: cuánto se sale el valor real de la banda cruda
            clo, chi = mlo.predict(CALIB[feats]), mhi.predict(CALIB[feats])
            yc = CALIB["log_chl_target"].values
            E = np.maximum(clo - yc, yc - chi)
            Q = conformal_quantile(E, NOMINAL)
            # aplicar a TEST: banda cruda +/- Q
            rlo, rhi = mlo.predict(TEST[feats]), mhi.predict(TEST[feats])
            y = TEST["log_chl_target"].values
            cov_raw = _coverage(y, np.minimum(rlo, rhi), np.maximum(rlo, rhi))
            lo = np.minimum(rlo, rhi) - Q; hi = np.maximum(rlo, rhi) + Q
            # La banda desplegada siempre contiene la prediccion puntual del regresor.
            point = _model().fit(DEV[feats], DEV["log_chl_target"]).predict(TEST[feats])
            lo = np.minimum(lo, point); hi = np.maximum(hi, point)
            cov = temporal_block_bootstrap(
                lambda a, b, c: _coverage(a, b, c), y, lo, hi,
                dates=TEST["fecha_target"].values,
                bodies=TEST["water_body"].values)
            width_ugl = float(np.mean(np.expm1(hi) - np.expm1(lo)))
            report[group][h] = {
                "n_test": int(len(y)), "cobertura_cqr": cov, "cobertura_cruda": float(cov_raw),
                "nominal": NOMINAL, "ancho_ugl": width_ugl, "Q_conformal": Q,
                "test_cutoff": str(split["cutoff"].date()),
                "calibration_start": str(calibration_start.date()),
                "bootstrap": "bloques de 14 dias por cuerpo",
            }
            flag = "OK" if abs(cov[0] - NOMINAL) <= 0.10 else ("ESTRECHO" if cov[0] < NOMINAL else "ANCHO")
            print(f"  +{h}d  n={len(y):>3} | cobertura CQR={cov[0]:.2f} [{cov[1]:.2f},{cov[2]:.2f}] "
                  f"(cruda={cov_raw:.2f}, nominal {NOMINAL:.2f}) [{flag}] | ancho~{width_ugl:.1f} ug/L")
        print()
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"Reporte -> {OUT}")
    print("Lectura: cobertura cercana a 0.80 -> intervalos calibrados (fiables). Mucho menor -> "
          "demasiado estrechos (sobreconfiados); mucho mayor -> demasiado anchos (poco útiles).")


if __name__ == "__main__":
    main()
