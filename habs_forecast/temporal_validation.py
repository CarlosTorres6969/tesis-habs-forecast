"""Utilidades compartidas para validacion temporal estrictamente causal.

Las reglas de este modulo son deliberadamente simples:

* un modelo agrupado usa una sola fecha de corte para todos los cuerpos;
* ninguna etiqueta de entrenamiento puede alcanzar la fecha predictora del test;
* todas las filas de una misma fecha permanecen en el mismo bloque;
* la incertidumbre se remuestrea por bloques cuerpo-tiempo, no por filas iid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DAY_NS = 86_400_000_000_000


def _dated(df):
    """Devuelve una copia con las dos fechas del protocolo normalizadas."""
    out = df.copy()
    for col in ("fecha_t0", "fecha_target"):
        if col not in out:
            raise ValueError(f"Falta la columna temporal obligatoria: {col}")
        out[col] = pd.to_datetime(out[col])
    return out


def assert_no_temporal_overlap(train, test):
    """Falla si una etiqueta de TRAIN llega a la fecha de alguna feature de TEST."""
    if train.empty or test.empty:
        return
    last_label = pd.Timestamp(train["fecha_target"].max())
    first_feature = pd.Timestamp(test["fecha_t0"].min())
    if last_label >= first_feature:
        raise AssertionError(
            "Fuga temporal: la ultima etiqueta de TRAIN "
            f"({last_label.date()}) alcanza el primer t0 de TEST ({first_feature.date()})"
        )


def common_temporal_holdout(df, test_frac=0.25, purge_days=8):
    """Corte final comun para un modelo agrupado (un grupo y un horizonte).

    La fecha se calcula sobre fechas unicas, no sobre filas, para que escenas repetidas
    del mismo dia nunca queden repartidas entre DEV y TEST. El inicio real del test queda
    despues tanto del embargo nominal como de la ultima etiqueta presente en DEV.
    """
    d = _dated(df).sort_values(["fecha_t0", "water_body"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(d["fecha_t0"].dropna().unique()))
    if len(dates) < 4:
        return d.iloc[0:0], d.iloc[0:0], {
            "cutoff": None, "embargo_end": None, "bodies": []
        }
    cut_idx = int(np.ceil((1.0 - test_frac) * len(dates))) - 1
    cut_idx = max(0, min(cut_idx, len(dates) - 2))
    cutoff = pd.Timestamp(dates[cut_idx])
    dev = d[d["fecha_t0"] <= cutoff].copy()
    nominal_end = cutoff + pd.Timedelta(days=purge_days)
    last_dev_label = pd.Timestamp(dev["fecha_target"].max())
    embargo_end = max(nominal_end, last_dev_label)
    test = d[d["fecha_t0"] > embargo_end].copy()
    assert_no_temporal_overlap(dev, test)
    return dev, test, {
        "cutoff": cutoff,
        "embargo_end": embargo_end,
        "bodies": sorted(test["water_body"].dropna().unique().tolist()),
    }


def expanding_purged_splits(
    df, n_splits=4, min_train_frac=0.4, min_train=20, min_test=3
):
    """Ventana expansiva por fechas comunes con purga basada en ``fecha_target``.

    Es valida tanto para un cuerpo como para un modelo agrupado. En el segundo caso evita
    que datos cronologicamente futuros de otro cuerpo entrenen el modelo del mismo fold.
    """
    d = _dated(df).sort_values(["fecha_t0", "water_body"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(d["fecha_t0"].dropna().unique()))
    start = int(np.floor(len(dates) * min_train_frac))
    if start >= len(dates) or len(dates) - start < n_splits:
        return []
    chunks = [x for x in np.array_split(dates[start:], n_splits) if len(x)]
    splits = []
    for chunk in chunks:
        test_start, test_end = pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1])
        train = d[d["fecha_target"] < test_start].copy()
        test = d[(d["fecha_t0"] >= test_start) & (d["fecha_t0"] <= test_end)].copy()
        if len(train) < min_train or len(test) < min_test:
            continue
        assert_no_temporal_overlap(train, test)
        splits.append((train, test, {"test_start": test_start, "test_end": test_end}))
    return splits


def purged_tail_split(df, calibration_frac=0.25, min_train=30, min_calibration=20):
    """Divide TRAIN/CALIB por fecha y purga etiquetas que alcanzan CALIB."""
    d = _dated(df).sort_values(["fecha_t0", "water_body"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(d["fecha_t0"].dropna().unique()))
    if len(dates) < 3:
        return d.iloc[0:0], d.iloc[0:0], None
    idx = max(1, min(len(dates) - 1, int(np.floor((1 - calibration_frac) * len(dates)))))
    while idx > 1:
        start = pd.Timestamp(dates[idx])
        cal = d[d["fecha_t0"] >= start]
        if len(cal) >= min_calibration:
            break
        idx -= 1
    start = pd.Timestamp(dates[idx])
    train = d[d["fecha_target"] < start].copy()
    cal = d[d["fecha_t0"] >= start].copy()
    if len(train) < min_train or len(cal) < min_calibration:
        return d.iloc[0:0], d.iloc[0:0], start
    assert_no_temporal_overlap(train, cal)
    return train, cal, start


def development_event_thresholds(dev, percentile=85):
    """Umbral por cuerpo calculado solo con targets unicos del bloque de desarrollo."""
    base = _dated(dev).drop_duplicates(["water_body", "fecha_target"])
    return (
        base.groupby("water_body")["chl_target"]
        .quantile(float(percentile) / 100.0)
        .to_dict()
    )


def apply_event_thresholds(df, thresholds):
    """Reetiqueta una copia usando umbrales aprendidos fuera del bloque evaluado."""
    out = df.copy()
    thr = out["water_body"].map(thresholds)
    if thr.isna().any():
        missing = sorted(out.loc[thr.isna(), "water_body"].unique().tolist())
        raise ValueError(f"No hay umbral de desarrollo para: {missing}")
    out["thr_body_eval"] = thr.astype(float)
    out["hab_target"] = (out["chl_target"] >= out["thr_body_eval"]).astype(int)
    return out


def conformal_quantile(scores, coverage=0.80):
    """Cuantil split-conformal finito con la correccion ``ceil((n+1)*coverage)/n``.

    ``method='higher'`` evita interpolar hacia abajo entre scores y perder la garantia
    finito-muestral por un detalle numerico.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No hay scores de conformidad finitos")
    level = min(1.0, np.ceil((len(values) + 1) * float(coverage)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def temporal_block_bootstrap(
    fn, *arrays, dates, bodies=None, block_days=14, n=1000, random_state=42
):
    """Estimacion puntual + IC95% mediante bootstrap de bloques cuerpo-tiempo."""
    values = [np.asarray(x) for x in arrays]
    if not values or not len(values[0]):
        return (np.nan, np.nan, np.nan)
    size = len(values[0])
    if any(len(x) != size for x in values):
        raise ValueError("Los arrays del bootstrap deben estar alineados")
    dt = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    if len(dt) != size:
        raise ValueError("dates debe estar alineado con los arrays")
    body = np.asarray(bodies if bodies is not None else np.repeat("all", size)).astype(str)
    day = dt.asi8 // DAY_NS
    block = day // int(block_days)
    labels = np.char.add(np.char.add(body, "|"), block.astype(str))
    unique = np.unique(labels)
    members = {label: np.flatnonzero(labels == label) for label in unique}

    point = fn(*values)
    if point is None or not np.isfinite(point):
        point = np.nan
    rng = np.random.default_rng(random_state)
    boot = []
    for _ in range(n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[label] for label in chosen])
        value = fn(*[x[idx] for x in values])
        if value is not None and np.isfinite(value):
            boot.append(float(value))
    if not boot:
        return (float(point), np.nan, np.nan)
    return (
        float(point),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )
