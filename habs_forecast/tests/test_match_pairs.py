import pandas as pd

import config as C
from match_pairs import autoregressive_features


def _target(last_date):
    dates = pd.date_range(pd.Timestamp(last_date) - pd.Timedelta(days=20), last_date)
    return pd.DataFrame({"fecha": dates, "chl_target": range(1, len(dates) + 1)})


def test_autoregressive_features_reject_stale_target():
    t0 = pd.Timestamp("2026-07-20")
    target = _target(t0 - pd.Timedelta(days=C.MAX_TARGET_AGE_DAYS + 1))
    assert autoregressive_features(target, t0) is None


def test_autoregressive_features_accept_fresh_target_and_is_causal():
    t0 = pd.Timestamp("2026-07-20")
    target = _target(t0)
    future = pd.DataFrame({"fecha": [t0 + pd.Timedelta(days=1)], "chl_target": [9999.0]})
    result = autoregressive_features(pd.concat([target, future]), t0)
    assert result is not None
    assert result["chl_t0"] == target.iloc[-1]["chl_target"]
    assert result["target_date_t0"] == t0
