"""
evaluate_robust.py — Evaluacion ROBUSTA (estabiliza la alerta de alta varianza).

Problema: el fold unico 70/30 por cuerpo-horizonte deja test pequenos (15-25 muestras)
-> Recall/PR-AUC inestables y muchos nan. Solucion:

  1. Walk-forward de VENTANA EXPANSIVA (multiples folds) por cuerpo -> recoge predicciones
     FUERA DE MUESTRA (OOS) de toda la serie, no solo del ultimo 30%.
  2. AGRUPA las OOS por (grupo, horizonte) a traves de cuerpos y folds -> n grande.
  3. Intervalos de confianza 95% por BOOTSTRAP sobre las OOS agrupadas.

Reporta el producto PRINCIPAL (skill de regresion vs persistencia) y el COMPLEMENTARIO
(alerta) con su incertidumbre. Reusa FEATURES/_model/_clf de train.py.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import (mean_squared_error, average_precision_score,
                             recall_score, f1_score)
import config as C
from train import FEATURES, _model, _clf, PAIRS

OUT = os.path.join(C.DIR_REPORTS, "robust_metrics.json")
N_FOLDS = 4
MIN_TRAIN_FRAC = 0.4
B = 1000          # repeticiones bootstrap
RNG = np.random.default_rng(C.RANDOM_STATE)


def expanding_oos(g):
    """Ventana expansiva: predicciones OOS de cada bloque futuro. Devuelve DataFrame OOS."""
    g = g.sort_values("fecha_t0").reset_index(drop=True)
    N = len(g)
    start = int(N * MIN_TRAIN_FRAC)
    if N - start < N_FOLDS * 4:
        return pd.DataFrame()
    fold = (N - start) // N_FOLDS
    out = []
    for k in range(N_FOLDS):
        a = start + k * fold
        b = N if k == N_FOLDS - 1 else a + fold
        tr, te = g.iloc[:a], g.iloc[a:b]
        if len(te) < 3 or len(tr) < 20:
            continue
        reg = _model().fit(tr[FEATURES], tr["log_chl_target"])
        yhat = reg.predict(te[FEATURES])
        proba = np.full(len(te), np.nan)
        if tr["hab_target"].nunique() > 1:
            clf = _clf(tr["hab_target"].values).fit(tr[FEATURES], tr["hab_target"])
            proba = clf.predict_proba(te[FEATURES])[:, 1]
        out.append(pd.DataFrame({
            "y_log": te["log_chl_target"].values, "yhat_log": yhat,
            "persist_log": te["log_chl_t0"].values if "log_chl_t0" in te else np.nan,
            "proba": proba, "hab": te["hab_target"].values,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _boot(fn, *arrays, n=B):
    """IC 95% por bootstrap de un estadistico fn sobre arrays alineados."""
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        v = fn(*[a[idx] for a in arrays])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(vals)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def _skill(y, yhat, persist):
    rm = np.sqrt(mean_squared_error(y, yhat))
    rp = np.sqrt(mean_squared_error(y, persist))
    return 1 - rm / rp if rp > 0 else np.nan          # skill score vs persistencia


def _recall(hab, proba):
    if hab.sum() == 0 or hab.sum() == len(hab):
        return None
    return recall_score(hab, (proba >= 0.5).astype(int), zero_division=0)


def _prauc(hab, proba):
    if hab.sum() == 0 or hab.sum() == len(hab):
        return None
    return average_precision_score(hab, proba)


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    feats = [f for f in FEATURES if f in df.columns]
    FEATURES[:] = feats
    report = {}
    for group in ("freshwater", "marine"):
        print(f"\n############  {group}  —  OOS agrupado + IC95% bootstrap  ############")
        report[group] = {}
        for h in [x for x in C.HORIZONS if x != 0]:          # h=0 es degenerado
            pooled = []
            for wb, g in df[(df["group"] == group) & (df["horizon"] == h)].groupby("water_body"):
                oos = expanding_oos(g)
                if len(oos):
                    oos["water_body"] = wb
                    pooled.append(oos)
            if not pooled:
                continue
            P = pd.concat(pooled, ignore_index=True)
            y, yhat, per = P["y_log"].values, P["yhat_log"].values, P["persist_log"].values
            mask = np.isfinite(per)
            skill = _boot(_skill, y[mask], yhat[mask], per[mask])
            hab, proba = P["hab"].values, P["proba"].values
            cm = np.isfinite(proba)
            rec = _boot(lambda a, b: _recall(a, b), hab[cm], proba[cm]) if cm.sum() else (np.nan,)*3
            pra = _boot(lambda a, b: _prauc(a, b), hab[cm], proba[cm]) if cm.sum() else (np.nan,)*3
            report[group][h] = {"n": int(len(P)), "pos": int(hab.sum()),
                                "skill_reg": skill, "recall_clf": rec, "pr_auc_clf": pra}
            print(f"  +{h}d  n={len(P):>4} eventos={int(hab.sum()):>3} | "
                  f"SKILL reg={skill[0]:+.2f} [{skill[1]:+.2f},{skill[2]:+.2f}] | "
                  f"Recall={rec[0]:.2f} [{rec[1]:.2f},{rec[2]:.2f}] | "
                  f"PR-AUC={pra[0]:.2f} [{pra[1]:.2f},{pra[2]:.2f}]")
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReporte -> {OUT}")
    print("Lectura: SKILL>0 => mejor que persistencia; IC que no cruza 0 => skill significativo.")


if __name__ == "__main__":
    main()
