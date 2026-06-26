"""
evaluate_intervals.py — VALIDA intervalos de incertidumbre (regresion cuantil) en el TEST INTACTO.

Anade una banda de incertidumbre a cada pronostico de intensidad (clorofila-a) en vez de un solo
punto: cuantiles P10/P50/P90 (XGBoost objetivo cuantil). Un intervalo P10-P90 honesto debe
CONTENER el valor real ~80% de las veces (cobertura nominal). Aqui se mide la COBERTURA EMPIRICA
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
from train import PAIRS
from evaluate_nested import (split_dev_test, inner_select, MIN_DEV, MIN_TEST)
from evaluate_robust import _boot

OUT = os.path.join(C.DIR_REPORTS, "interval_metrics.json")
QLO, QMID, QHI = 0.10, 0.50, 0.90
NOMINAL = QHI - QLO          # cobertura objetivo del intervalo P10-P90 = 0.80


def _qmodel(alpha):
    """XGBoost de regresion cuantil (mismos hiperparametros que el punto)."""
    from xgboost import XGBRegressor
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                        random_state=C.RANDOM_STATE, n_jobs=4)


def _coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    avail = set(df.columns)
    report = {}
    print("INTERVALOS P10-P90 (cobertura nominal 0.80) — validados en TEST INTACTO\n")
    for group in ("freshwater", "marine"):
        print(f"############  {group}  ############")
        report[group] = {}
        for h in [x for x in C.HORIZONS if x != 0]:
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            devs, tests = [], []
            for wb, g in d.groupby("water_body"):
                dev, test, _ = split_dev_test(g)
                if len(dev) < MIN_DEV or len(test) < MIN_TEST:
                    continue
                devs.append(dev); tests.append(test)
            if not devs:
                print(f"  +{h}d  (datos insuficientes)"); continue
            DEV = pd.concat(devs, ignore_index=True).sort_values("fecha_t0")
            TEST = pd.concat(tests, ignore_index=True)
            _, feats = inner_select(DEV, avail)        # mismas features que el punto (solo DEV)
            # CQR: parte DEV en TRAIN (pasado) + CALIB (25% mas reciente, imita el shift al test)
            ncal = max(int(len(DEV) * 0.25), 20)
            TRAIN, CALIB = DEV.iloc[:-ncal], DEV.iloc[-ncal:]
            mlo = _qmodel(QLO).fit(TRAIN[feats], TRAIN["log_chl_target"])
            mhi = _qmodel(QHI).fit(TRAIN[feats], TRAIN["log_chl_target"])
            # conformidad en CALIB: cuanto se sale el valor real de la banda cruda
            clo, chi = mlo.predict(CALIB[feats]), mhi.predict(CALIB[feats])
            yc = CALIB["log_chl_target"].values
            E = np.maximum(clo - yc, yc - chi)
            qlevel = min(1.0, (1 - (1 - NOMINAL)) * (1 + 1.0 / len(E)))   # nivel conformal ajustado
            Q = float(np.quantile(E, qlevel))          # ensanchamiento (puede ser negativo si sobra)
            # aplicar a TEST: banda cruda +/- Q
            rlo, rhi = mlo.predict(TEST[feats]), mhi.predict(TEST[feats])
            y = TEST["log_chl_target"].values
            cov_raw = _coverage(y, np.minimum(rlo, rhi), np.maximum(rlo, rhi))
            lo = np.minimum(rlo, rhi) - Q; hi = np.maximum(rlo, rhi) + Q
            cov = _boot(lambda a, b, c: _coverage(a, b, c), y, lo, hi)
            width_ugl = float(np.mean(np.expm1(hi) - np.expm1(lo)))
            report[group][h] = {
                "n_test": int(len(y)), "cobertura_cqr": cov, "cobertura_cruda": float(cov_raw),
                "nominal": NOMINAL, "ancho_ugl": width_ugl, "Q_conformal": Q,
            }
            flag = "OK" if abs(cov[0] - NOMINAL) <= 0.10 else ("ESTRECHO" if cov[0] < NOMINAL else "ANCHO")
            print(f"  +{h}d  n={len(y):>3} | cobertura CQR={cov[0]:.2f} [{cov[1]:.2f},{cov[2]:.2f}] "
                  f"(cruda={cov_raw:.2f}, nominal {NOMINAL:.2f}) [{flag}] | ancho~{width_ugl:.1f} ug/L")
        print()
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"Reporte -> {OUT}")
    print("Lectura: cobertura cercana a 0.80 -> intervalos calibrados (fiables). Mucho menor -> "
          "demasiado estrechos (sobreconfiados); mucho mayor -> demasiado anchos (poco utiles).")


if __name__ == "__main__":
    main()
