"""Reajusta el calibrador operativo sin reutilizar sus datos para reportar desempeño.

El umbral F2 que se despliega fue seleccionado dentro de DEV por ``evaluate_nested.py`` y
validado una sola vez en su TEST temporal. Este script solo reajusta la curva isotónica de
producción con predicciones OOS purgadas de todo el histórico; sus métricas impresas son un
diagnóstico de ajuste, no las cifras de eficacia que se citan en la tesis.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_fscore_support

import config as C
from train import PAIRS, get_features
from train_stack import oos_both
from temporal_validation import apply_event_thresholds

MODELS = C.DIR_MODELS
NESTED = os.path.join(C.DIR_REPORTS, "nested_metrics.json")
BETA = 2.0


def fbeta(precision, recall, beta=BETA):
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def _fallback_threshold(y, probability):
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.95, 95):
        precision, recall, _, _ = precision_recall_fscore_support(
            y, (probability >= threshold).astype(int),
            average="binary", zero_division=0)
        score = fbeta(precision, recall)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])
    nested = json.load(open(NESTED, encoding="utf-8")) if os.path.exists(NESTED) else {}

    for group in ("freshwater", "marine"):
        parts = []
        for horizon in [1, 3, 5, 7]:
            data = df[(df["group"] == group) & (df["horizon"] == horizon)]
            node = nested.get(group, {}).get(str(horizon), {})
            event_thresholds = node.get("event_thresholds_from_dev")
            if not event_thresholds:
                raise RuntimeError(
                    f"Faltan umbrales validados para {group} +{horizon}d"
                )
            data = apply_event_thresholds(data, event_thresholds)
            feats = get_features(group, horizon, data.columns, required=True)
            predictions = oos_both(data, feats)
            if predictions.empty:
                continue
            mask = np.isfinite(predictions["xgb_proba"]) & np.isfinite(predictions["nn_proba"])
            part = predictions.loc[mask, ["hab"]].copy()
            part["probability"] = (
                0.5 * predictions.loc[mask, "xgb_proba"].to_numpy()
                + 0.5 * predictions.loc[mask, "nn_proba"].to_numpy()
            )
            parts.append(part)
        if not parts:
            continue

        pooled = pd.concat(parts, ignore_index=True)
        probability = pooled["probability"].to_numpy()
        y = pooled["hab"].to_numpy(dtype=int)
        iso = IsotonicRegression(out_of_bounds="clip").fit(probability, y)
        calibrated = iso.predict(probability)

        validation = nested.get(group, {}).get("alert_calibration") or {}
        threshold = validation.get("threshold_selected_in_dev")
        source = "nested_development_only"
        if threshold is None:
            threshold = _fallback_threshold(y, calibrated)
            source = "fallback_oos_refit_not_independently_validated"

        precision, recall, _, _ = precision_recall_fscore_support(
            y, (calibrated >= threshold).astype(int),
            average="binary", zero_division=0)
        artifact = {
            "iso": iso,
            "threshold": float(threshold),
            "beta": BETA,
            "threshold_source": source,
            "n_oos_refit": int(len(y)),
            "validated_test_metrics": validation,
            "label_source": "nested_development_only_per_horizon",
        }
        joblib.dump(artifact, os.path.join(MODELS, f"alert_calib_{group}.pkl"))
        print(f"\n=== {group} | reajuste OOS purgado n={len(y)} eventos={int(y.sum())} ===")
        print(f"  umbral={threshold:.2f} ({source})")
        print(f"  diagnóstico de ajuste, NO desempeño final: recall={recall:.2f} precision={precision:.2f}")

    print(f"\nCalibradores operativos -> {MODELS}")
    print("Las métricas defendibles proceden del TEST de nested_metrics.json.")


if __name__ == "__main__":
    main()
