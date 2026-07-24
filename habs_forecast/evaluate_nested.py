"""Validacion anidada con test final, cortes globales y purga temporal completa.

Cada modelo se entrena agrupando cuerpos del mismo ecosistema. Por ello el corte temporal
tambien es comun a todos los cuerpos: ningun lago o costa puede aportar etiquetas posteriores
a una fecha que se esta evaluando en otro cuerpo. Las familias y el umbral de alerta se eligen
solo dentro de DEV; TEST se usa exclusivamente para medir el procedimiento resultante.
"""
from __future__ import annotations

import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import config as C
from temporal_validation import (
    apply_event_thresholds,
    common_temporal_holdout,
    development_event_thresholds,
    expanding_purged_splits,
    temporal_block_bootstrap,
)
from train import (
    AUTOREG,
    DYNAMICS,
    ERA5,
    NUTRIENTS,
    PAIRS,
    SEASONAL,
    SPECTRAL,
    WATERQUAL,
    _clf,
    _model,
)
from train_nn import _fit

warnings.filterwarnings("ignore")

OUT = os.path.join(C.DIR_REPORTS, "nested_metrics.json")
FEATURE_OUT = os.path.join(C.DIR_REPORTS, "feature_sets.json")
ROBUST = os.path.join(C.DIR_REPORTS, "robust_metrics.json")
PRED_DUMP = os.path.join(C.DIR_REPORTS, "nested_test_predictions.csv")

TEST_FRAC = 0.25
PURGE_DAYS = C.VALIDATION["purge_days"]
N_INNER = 3
MIN_TRAIN_FRAC = 0.45
MIN_TEST = 8
MIN_DEV = 40
SELECT_MARGIN = 0.02
BETA = 2.0

FAMILIES = {
    "ERA5": ERA5,
    "SPECTRAL": SPECTRAL,
    "INSITU": NUTRIENTS + WATERQUAL,
    "DYNAMICS": DYNAMICS,
    "SEASONAL": SEASONAL,
}
OPTIONAL = ["ERA5", "SPECTRAL", "INSITU"]
if os.environ.get("HABS_NEWFEATS"):
    OPTIONAL += ["DYNAMICS", "SEASONAL"]


def _skill(y, yhat, persistence):
    reference = np.sqrt(mean_squared_error(y, persistence))
    if reference <= 0:
        return np.nan
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / reference


def _fbeta(precision, recall, beta=BETA):
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def _precision(y, pred):
    positives = np.sum(pred == 1)
    return float(np.sum((y == 1) & (pred == 1)) / positives) if positives else 0.0


def _recall(y, pred):
    positives = np.sum(y == 1)
    return float(np.sum((y == 1) & (pred == 1)) / positives) if positives else 0.0


def choose_alert_threshold(y, calibrated_probability):
    best = (0.5, -1.0)
    for threshold in np.linspace(0.01, 0.95, 95):
        pred = (calibrated_probability >= threshold).astype(int)
        score = _fbeta(_precision(y, pred), _recall(y, pred))
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def _subsets():
    for size in range(len(OPTIONAL) + 1):
        for combo in itertools.combinations(OPTIONAL, size):
            yield list(combo)


def _feats_of(combo, available):
    feats = list(AUTOREG)
    for family in combo:
        feats += FAMILIES[family]
    return [feature for feature in feats if feature in available]


def _inner_oos_skill(dev, feats):
    """Skill interno del mismo modelo agrupado, con fechas comunes y purga por target."""
    ys, predictions, persistence = [], [], []
    splits = expanding_purged_splits(
        dev,
        n_splits=N_INNER,
        min_train_frac=MIN_TRAIN_FRAC,
        min_train=20,
        min_test=3,
    )
    for train, test, _ in splits:
        model = _model().fit(train[feats], train["log_chl_target"])
        ys.append(test["log_chl_target"].to_numpy())
        predictions.append(model.predict(test[feats]))
        persistence.append(test["log_chl_t0"].to_numpy())
    if not ys:
        return np.nan
    return _skill(np.concatenate(ys), np.concatenate(predictions), np.concatenate(persistence))


def inner_select(dev, available):
    """Seleccion por skill purgado en DEV, con regla de parsimonia."""
    scored = []
    for combo in _subsets():
        score = _inner_oos_skill(dev, _feats_of(combo, available))
        scored.append((score if np.isfinite(score) else -9.0, combo))
    best = max(score for score, _ in scored)
    candidates = [
        (len(combo), len(_feats_of(combo, available)), combo)
        for score, combo in scored
        if score >= best - SELECT_MARGIN
    ]
    candidates.sort()
    combo = candidates[0][2]
    return ["AUTOREG"] + combo, _feats_of(combo, available)


def nn_proba(train, test, feats):
    """Probabilidad de la red con imputacion y escalado aprendidos solo en TRAIN."""
    imputer = SimpleImputer(keep_empty_features=True).fit(train[feats])
    scaler = StandardScaler().fit(imputer.transform(train[feats]))
    x_train = scaler.transform(imputer.transform(train[feats]))
    x_test = scaler.transform(imputer.transform(test[feats]))
    net = _fit(
        x_train,
        train["log_chl_target"].to_numpy(),
        train["hab_target"].to_numpy(dtype=float),
        len(feats),
    )
    with torch.no_grad():
        _, logits = net(torch.tensor(x_test, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def _development_alert_oos(dev, feats):
    """Probabilidades internas para calibrar isotonia/umbral sin usar TEST."""
    rows = []
    for train, validation, _ in expanding_purged_splits(
        dev, n_splits=N_INNER, min_train_frac=MIN_TRAIN_FRAC, min_train=30, min_test=3
    ):
        if train["hab_target"].nunique() < 2:
            continue
        xgb = _clf(train["hab_target"].to_numpy()).fit(
            train[feats], train["hab_target"]
        ).predict_proba(validation[feats])[:, 1]
        neural = nn_proba(train, validation, feats)
        part = validation[["water_body", "fecha_t0", "fecha_target", "hab_target"]].copy()
        part["probability"] = 0.5 * xgb + 0.5 * neural
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _alert_summary(development, test):
    """Ajusta calibrador/umbral en OOS de DEV y los evalua una vez en TEST."""
    if development.empty or test.empty:
        return None
    y_dev = development["hab_target"].to_numpy(dtype=int)
    if y_dev.min() == y_dev.max():
        return None
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        development["probability"].to_numpy(), y_dev
    )
    dev_calibrated = iso.predict(development["probability"].to_numpy())
    threshold, dev_f2 = choose_alert_threshold(y_dev, dev_calibrated)

    y_test = test["hab_target"].to_numpy(dtype=int)
    test_calibrated = iso.predict(test["probability"].to_numpy())
    pred = (test_calibrated >= threshold).astype(int)
    dates = test["fecha_target"].to_numpy()
    bodies = test["water_body"].to_numpy()
    precision = temporal_block_bootstrap(
        _precision, y_test, pred, dates=dates, bodies=bodies
    )
    recall = temporal_block_bootstrap(_recall, y_test, pred, dates=dates, bodies=bodies)
    f2 = temporal_block_bootstrap(
        lambda y, p: _fbeta(_precision(y, p), _recall(y, p)),
        y_test,
        pred,
        dates=dates,
        bodies=bodies,
    )
    return {
        "threshold_selected_in_dev": threshold,
        "f2_development": dev_f2,
        "n_development_oos": int(len(development)),
        "n_test": int(len(test)),
        "events_test": int(y_test.sum()),
        "precision_test": precision,
        "recall_test": recall,
        "f2_test": f2,
    }


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])
    available = set(df.columns)
    optimistic = json.load(open(ROBUST)) if os.path.exists(ROBUST) else {}
    report, selected_sets, dump_rows = {}, {}, []

    print("PROTOCOLO ANIDADO PURGADO: corte comun por grupo; TEST solo para medicion.")
    print(f"TEST = ultimo {int(TEST_FRAC * 100)}% de fechas | embargo minimo {PURGE_DAYS} d\n")

    for group in ("freshwater", "marine"):
        print(f"############  {group} - TEST FINAL TEMPORAL  ############")
        report[group], selected_sets[group] = {}, {}
        alert_dev_parts, alert_test_parts = [], []

        for horizon in [value for value in C.HORIZONS if value != 0]:
            data = df[(df["group"] == group) & (df["horizon"] == horizon)].copy()
            dev, test, split = common_temporal_holdout(
                data, test_frac=TEST_FRAC, purge_days=PURGE_DAYS
            )
            if len(dev) < MIN_DEV or len(test) < MIN_TEST:
                print(f"  +{horizon}d  (DEV/TEST insuficiente)")
                continue

            event_thresholds = development_event_thresholds(dev, C.RELATIVE_PERCENTILE)
            dev = apply_event_thresholds(dev, event_thresholds)
            test = apply_event_thresholds(test, event_thresholds)

            families, feats = inner_select(dev, available)
            selected_sets[group][str(horizon)] = {
                "families": families,
                "source": "nested_dev_purged",
                "cutoff": str(split["cutoff"].date()),
            }

            regressor = _model().fit(dev[feats], dev["log_chl_target"])
            prediction = regressor.predict(test[feats])
            valid = np.isfinite(test["log_chl_t0"].to_numpy())
            skill = temporal_block_bootstrap(
                _skill,
                test["log_chl_target"].to_numpy()[valid],
                prediction[valid],
                test["log_chl_t0"].to_numpy()[valid],
                dates=test["fecha_target"].to_numpy()[valid],
                bodies=test["water_body"].to_numpy()[valid],
            )

            probability = np.full(len(test), np.nan)
            pr_auc = (np.nan, np.nan, np.nan)
            if dev["hab_target"].nunique() > 1:
                xgb_probability = _clf(dev["hab_target"].to_numpy()).fit(
                    dev[feats], dev["hab_target"]
                ).predict_proba(test[feats])[:, 1]
                probability = 0.5 * xgb_probability + 0.5 * nn_proba(dev, test, feats)
                y_alert = test["hab_target"].to_numpy(dtype=int)
                if 0 < y_alert.sum() < len(y_alert):
                    pr_auc = temporal_block_bootstrap(
                        lambda y, p: average_precision_score(y, p)
                        if 0 < y.sum() < len(y)
                        else None,
                        y_alert,
                        probability,
                        dates=test["fecha_target"].to_numpy(),
                        bodies=test["water_body"].to_numpy(),
                    )

            dev_alert = _development_alert_oos(dev, feats)
            if len(dev_alert):
                alert_dev_parts.append(dev_alert)
            test_alert = test[["water_body", "fecha_t0", "fecha_target", "hab_target"]].copy()
            test_alert["probability"] = probability
            test_alert = test_alert[np.isfinite(test_alert["probability"])]
            if len(test_alert):
                alert_test_parts.append(test_alert)

            dump = test[[
                "group", "horizon", "water_body", "fecha_t0", "fecha_target",
                "log_chl_target", "log_chl_t0", "hab_target",
            ]].copy()
            dump["pred_log"] = prediction
            dump["alert_probability"] = probability
            dump["chl_real"] = np.expm1(dump["log_chl_target"])
            dump["chl_pred"] = np.clip(np.expm1(dump["pred_log"]), 0, None)
            dump["chl_persist"] = np.expm1(dump["log_chl_t0"])
            dump_rows.append(dump)

            old = optimistic.get(group, {}).get(str(horizon), {}).get("skill_reg", [None])
            old_point = old[0] if isinstance(old, list) else None
            report[group][str(horizon)] = {
                "n_test": int(len(test)),
                "pos_test": int(test["hab_target"].sum()),
                "skill_nested": skill,
                "pr_auc_nested": pr_auc,
                "skill_legacy_oos": old_point,
                "features_per_body": {"_grupo": "+".join(families)},
                "test_cutoff": str(split["cutoff"].date()),
                "embargo_end": str(split["embargo_end"].date()),
                "test_bodies": split["bodies"],
                "event_thresholds_from_dev": event_thresholds,
                "bootstrap": "bloques de 14 dias por cuerpo",
            }
            print(
                f"  +{horizon}d n_test={len(test):>3} eventos={int(test['hab_target'].sum()):>2} | "
                f"SKILL={skill[0]:+.2f} [{skill[1]:+.2f},{skill[2]:+.2f}] | "
                f"PR-AUC={pr_auc[0]:.2f} [{pr_auc[1]:.2f},{pr_auc[2]:.2f}] | "
                f"corte={split['cutoff'].date()}"
            )

        development = pd.concat(alert_dev_parts, ignore_index=True) if alert_dev_parts else pd.DataFrame()
        alert_test = pd.concat(alert_test_parts, ignore_index=True) if alert_test_parts else pd.DataFrame()
        report[group]["alert_calibration"] = _alert_summary(development, alert_test)
        print(f"  alerta calibrada: {report[group]['alert_calibration']}\n")

    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(FEATURE_OUT, "w", encoding="utf-8") as handle:
        json.dump(selected_sets, handle, indent=2, ensure_ascii=False)
    if dump_rows:
        pd.concat(dump_rows, ignore_index=True).to_csv(PRED_DUMP, index=False)
    print(f"Metricas -> {OUT}")
    print(f"Features validadas y usadas por produccion -> {FEATURE_OUT}")


if __name__ == "__main__":
    main()
