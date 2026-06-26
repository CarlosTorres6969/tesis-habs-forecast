"""
calibrate_alert.py — Calibra la probabilidad de alerta y fija el UMBRAL OPERATIVO.

Problema: con umbral fijo 0.5 el sistema casi nunca dispara (eventos raros, probas bajas).
Solucion:
  1. Genera probabilidades del ENSAMBLE (XGB+Red) FUERA DE MUESTRA (ventana expansiva).
  2. CALIBRA (isotonica) prob -> frecuencia real de floracion.
  3. Elige el UMBRAL OPERATIVO maximizando F-beta (beta=2, prioriza recall: en alerta
     temprana perder un bloom cuesta mas que una falsa alarma).
Guarda calibrador + umbral por grupo -> usado por predict.py.

Salida: artifacts/models/alert_calib_{grupo}.pkl
"""
from __future__ import annotations
import os, joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_fscore_support
import config as C
from train import FEATURES, PAIRS
from train_stack import oos_both

MODELS = C.DIR_MODELS
BETA = 2.0          # >1 prioriza recall


def fbeta(p, r, beta=BETA):
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + 1e-12)


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    feats = [f for f in FEATURES if f in df.columns]
    for group in ("freshwater", "marine"):
        # pool de probabilidades OOS del ensamble sobre todos los horizontes
        probs, labels = [], []
        for h in [1, 3, 5, 7]:
            P = oos_both(df[(df["group"] == group) & (df["horizon"] == h)], feats)
            if not len(P):
                continue
            cm = np.isfinite(P["xgb_proba"].values) & np.isfinite(P["nn_proba"].values)
            ens = 0.5 * P["xgb_proba"].values[cm] + 0.5 * P["nn_proba"].values[cm]
            probs.append(ens); labels.append(P["hab"].values[cm])
        if not probs:
            continue
        prob = np.concatenate(probs); y = np.concatenate(labels).astype(int)

        # calibracion isotonica (prob OOS -> frecuencia real)
        iso = IsotonicRegression(out_of_bounds="clip").fit(prob, y)
        pcal = iso.predict(prob)

        # umbral operativo: maximiza F-beta
        grid = np.linspace(0.05, 0.95, 91)
        best_t, best_f = 0.5, -1
        for t in grid:
            pr, rc, _, _ = precision_recall_fscore_support(
                y, (pcal >= t).astype(int), average="binary", zero_division=0)
            fb = fbeta(pr, rc)
            if fb > best_f:
                best_f, best_t = fb, t
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, (pcal >= best_t).astype(int), average="binary", zero_division=0)
        # comparacion con umbral 0.5 sin calibrar
        pr0, rc0, _, _ = precision_recall_fscore_support(
            y, (prob >= 0.5).astype(int), average="binary", zero_division=0)

        joblib.dump({"iso": iso, "threshold": float(best_t), "beta": BETA},
                    os.path.join(MODELS, f"alert_calib_{group}.pkl"))
        print(f"\n=== {group} | n={len(y)} eventos={y.sum()} ===")
        print(f"  ANTES (prob>=0.5 sin calibrar): recall={rc0:.2f} precision={pr0:.2f}")
        print(f"  DESPUES (calibrado, umbral={best_t:.2f}): recall={rc:.2f} "
              f"precision={pr:.2f} F{BETA:.0f}={best_f:.2f}")
    print(f"\nCalibradores -> {MODELS}")


if __name__ == "__main__":
    main()
