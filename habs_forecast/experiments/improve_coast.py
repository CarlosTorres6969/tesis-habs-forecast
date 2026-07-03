"""
improve_coast.py — Gana potencia estadistica en la COSTA (Fonseca/Tampa), donde el skill
no estaba establecido por pocos eventos (5-8 en n~85).

Dos palancas:
  A) POOLING de grupo: un solo modelo marino entrenado con ambos cuerpos a la vez
     (walk-forward expansivo a nivel de grupo) -> mas datos por fold que el per-cuerpo.
  B) SENSIBILIDAD del umbral de evento: P70/P75/P80/P85 de la climatologia local ->
     cuantos eventos y que Recall/PR-AUC resultan (mas eventos = mas potencia).

Compara skill de regresion vs persistencia con bootstrap IC95% (pooled vs per-cuerpo).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C
from train import FEATURES, _model, _clf, PAIRS
from evaluate_robust import _boot, _skill, _recall, _prauc, N_FOLDS, MIN_TRAIN_FRAC

TARGET = __import__("os").path.join(C.DIR_OUT, "targets", "satellite_chl_daily.csv")


def group_expanding_oos(d):
    """Walk-forward expansivo a NIVEL DE GRUPO: entrena un modelo con todos los cuerpos
    hasta T, predice el bloque futuro. Pool natural -> mas datos."""
    d = d.sort_values("fecha_t0").reset_index(drop=True)
    N = len(d); start = int(N * MIN_TRAIN_FRAC)
    if N - start < N_FOLDS * 4:
        return pd.DataFrame()
    fold = (N - start) // N_FOLDS
    out = []
    for k in range(N_FOLDS):
        a = start + k * fold
        b = N if k == N_FOLDS - 1 else a + fold
        tr, te = d.iloc[:a], d.iloc[a:b]
        if len(te) < 3 or len(tr) < 20:
            continue
        reg = _model().fit(tr[FEATURES], tr["log_chl_target"])
        proba = np.full(len(te), np.nan)
        if tr["hab_target"].nunique() > 1:
            clf = _clf(tr["hab_target"].values).fit(tr[FEATURES], tr["hab_target"])
            proba = clf.predict_proba(te[FEATURES])[:, 1]
        out.append(pd.DataFrame({
            "y_log": te["log_chl_target"].values, "yhat_log": reg.predict(te[FEATURES]),
            "persist_log": te["log_chl_t0"].values, "proba": proba,
            "chl_target": te["chl_target"].values, "water_body": te["water_body"].values,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    FEATURES[:] = [f for f in FEATURES if f in df.columns]
    clim = pd.read_csv(TARGET)

    print("######  COSTA: pooling de grupo (un modelo marino) + IC95%  ######")
    for h in [x for x in C.HORIZONS if x != 0]:
        d = df[(df["group"] == "marine") & (df["horizon"] == h)]
        P = group_expanding_oos(d)
        if not len(P):
            continue
        m = np.isfinite(P["persist_log"])
        sk = _boot(_skill, P["y_log"].values[m], P["yhat_log"].values[m], P["persist_log"].values[m])
        print(f"  +{h}d n={len(P):>3} | SKILL pooled={sk[0]:+.2f} [{sk[1]:+.2f},{sk[2]:+.2f}]")

    print("\n######  COSTA: sensibilidad del umbral de evento (pooled OOS)  ######")
    # umbrales por cuerpo a varios percentiles
    pcts = [70, 75, 80, 85]
    thr = {p: clim[clim["group"] == "marine"].groupby("water_body")["chl_ugl"]
              .quantile(p / 100).to_dict() for p in pcts}
    for h in [1, 3, 5, 7]:
        d = df[(df["group"] == "marine") & (df["horizon"] == h)]
        P = group_expanding_oos(d)
        if not len(P):
            continue
        line = f"  +{h}d:"
        for p in pcts:
            tvec = P["water_body"].map(thr[p]).values
            hab = (P["chl_target"].values >= tvec).astype(int)
            cm = np.isfinite(P["proba"].values)
            rec = _boot(lambda a, b: _recall(a, b), hab[cm], P["proba"].values[cm])
            line += f"  P{p}(ev={int(hab.sum())},Rec={rec[0]:.2f})"
        print(line)

    print("\nLectura: si el skill pooled gana significancia (IC no cruza 0) o un percentil mas bajo")
    print("da suficientes eventos con Recall razonable, ese es el ajuste recomendado para costa.")


if __name__ == "__main__":
    main()
