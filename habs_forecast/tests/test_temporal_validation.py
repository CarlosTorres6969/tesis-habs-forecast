import numpy as np
import pandas as pd

from temporal_validation import (
    apply_event_thresholds,
    assert_no_temporal_overlap,
    common_temporal_holdout,
    conformal_quantile,
    development_event_thresholds,
    expanding_purged_splits,
    purged_tail_split,
    temporal_block_bootstrap,
)


def _panel(days=80):
    rows = []
    origin = pd.Timestamp("2025-01-01")
    for body, offset in (("a", 0), ("b", 12)):
        for day in range(days):
            t0 = origin + pd.Timedelta(days=day + offset)
            # Dos escenas del mismo dia: nunca deben separarse entre folds.
            for scene in range(2):
                rows.append({
                    "water_body": body,
                    "fecha_t0": t0,
                    "fecha_target": t0 + pd.Timedelta(days=7),
                    "chl_target": float(day + scene),
                })
    return pd.DataFrame(rows)


def test_common_holdout_is_global_and_purged():
    dev, test, metadata = common_temporal_holdout(_panel(), test_frac=0.25, purge_days=8)
    assert metadata["cutoff"] is not None
    assert dev["fecha_target"].max() < test["fecha_t0"].min()
    assert_no_temporal_overlap(dev, test)
    assert set(dev["water_body"]) == {"a", "b"}
    assert set(test["water_body"]) == {"a", "b"}


def test_expanding_splits_keep_dates_together_and_purge_targets():
    splits = expanding_purged_splits(
        _panel(), n_splits=4, min_train_frac=0.4, min_train=20, min_test=3)
    assert len(splits) == 4
    for train, test, _ in splits:
        assert train["fecha_target"].max() < test["fecha_t0"].min()
        assert set(train["fecha_t0"]).isdisjoint(set(test["fecha_t0"]))
        # Las dos escenas de cada cuerpo-fecha permanecen juntas.
        counts = test.groupby(["water_body", "fecha_t0"]).size()
        assert (counts == 2).all()


def test_tail_calibration_is_purged():
    train, calibration, _ = purged_tail_split(
        _panel(), calibration_frac=0.25, min_train=20, min_calibration=20)
    assert len(train) and len(calibration)
    assert train["fecha_target"].max() < calibration["fecha_t0"].min()


def test_event_threshold_uses_development_only():
    dev = _panel(days=20)
    threshold = development_event_thresholds(dev, percentile=85)
    future = dev.iloc[:4].copy()
    future["chl_target"] = 10_000.0
    labelled = apply_event_thresholds(future, threshold)
    assert labelled["hab_target"].eq(1).all()
    assert all(value < 10_000 for value in threshold.values())


def test_block_bootstrap_returns_actual_point_estimate():
    values = np.array([1.0, 1.0, 3.0, 3.0])
    dates = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"])
    point, low, high = temporal_block_bootstrap(
        np.mean, values, dates=dates, bodies=np.array(["a"] * 4), n=100)
    assert point == 2.0
    assert low <= point <= high


def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(1.0, 10.0)
    # ceil((9 + 1) * .80) / 9 = 1.0: debe seleccionar el maximo.
    assert conformal_quantile(scores, coverage=0.80) == 9.0
