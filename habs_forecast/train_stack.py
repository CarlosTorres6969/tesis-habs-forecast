"""
train_stack.py — Sistema UNIFICADO XGBoost + Red Neuronal (ensamble / stacking).

Combina los dos modelos en cada fold de la ventana expansiva:
  - REGRESION (intensidad): blend = w*XGB + (1-w)*NN, para varios w -> mejor combinacion.
  - ALERTA (clasificacion): promedio de probabilidades XGB y NN (ensamble blando).

Como arboles y red tienen sesgos inductivos distintos, sus errores se decorrelacionan y la
mezcla puede superar a cada uno. Evaluado con bootstrap IC95% (mismo protocolo). El sistema
final 'aplica redes neuronales' (la red es parte del modelo) ademas de XGBoost.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, average_precision_score, recall_score
import torch
import config as C
from train import FEATURES, PAIRS, _model, _clf
from train_nn import HABNet, _fit
from evaluate_robust import _boot, N_FOLDS, MIN_TRAIN_FRAC

OUT = os.path.join(C.DIR_REPORTS, "stack_metrics.json")
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]      # w sobre XGB (0=solo red, 1=solo XGB)


def oos_both(d, feats):
    """Ventana expansiva: predicciones OOS de XGB y NN alineadas."""
    d = d.sort_values("fecha_t0").reset_index(drop=True)
    N = len(d); start = int(N * MIN_TRAIN_FRAC)
    if N - start < N_FOLDS * 4:
        return pd.DataFrame()
    fold = (N - start) // N_FOLDS
    rows = []
    for k in range(N_FOLDS):
        a = start + k * fold
        b = N if k == N_FOLDS - 1 else a + fold
        tr, te = d.iloc[:a], d.iloc[a:b]
        if len(te) < 3 or len(tr) < 30:
            continue
        # --- XGBoost (regresion + alerta) ---
        xr = _model().fit(tr[feats], tr["log_chl_target"]).predict(te[feats])
        xp = np.full(len(te), np.nan)
        if tr["hab_target"].nunique() > 1:
            xp = _clf(tr["hab_target"].values).fit(tr[feats], tr["hab_target"]).predict_proba(te[feats])[:, 1]
        # --- Red neuronal (regresion + alerta), con imputacion+escalado ---
        imp = SimpleImputer().fit(tr[feats]); sc = StandardScaler().fit(imp.transform(tr[feats]))
        Xtr = sc.transform(imp.transform(tr[feats])); Xte = sc.transform(imp.transform(te[feats]))
        net = _fit(Xtr, tr["log_chl_target"].values, tr["hab_target"].values.astype(float), len(feats))
        with torch.no_grad():
            nr, nc = net(torch.tensor(Xte, dtype=torch.float32))
        rows.append(pd.DataFrame({
            "y_log": te["log_chl_target"].values, "persist": te["log_chl_t0"].values,
            "xgb_reg": xr, "nn_reg": nr.numpy(),
            "xgb_proba": xp, "nn_proba": torch.sigmoid(nc).numpy(),
            "hab": te["hab_target"].values,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _skill(y, yhat, per):
    rp = np.sqrt(mean_squared_error(y, per))
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / rp if rp > 0 else np.nan


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    feats = [f for f in FEATURES if f in df.columns]
    report = {}
    for group in ("freshwater", "marine"):
        print(f"\n############  UNIFICADO XGB+RED — {group}  ############")
        report[group] = {}
        for h in [1, 3, 5, 7]:
            P = oos_both(df[(df["group"] == group) & (df["horizon"] == h)], feats)
            if not len(P):
                continue
            m = np.isfinite(P["persist"]); y = P["y_log"].values[m]; per = P["persist"].values[m]
            xg = P["xgb_reg"].values[m]; nn = P["nn_reg"].values[m]
            # skill de regresion por peso de blend
            sk = {}
            for w in WEIGHTS:
                blend = w * xg + (1 - w) * nn
                sk[w] = _boot(_skill, y, blend, per)
            best_w = max(sk, key=lambda w: sk[w][0] if np.isfinite(sk[w][0]) else -9)
            # alerta: ensamble blando (promedio de probas)
            cm = np.isfinite(P["xgb_proba"].values) & np.isfinite(P["nn_proba"].values)
            hab = P["hab"].values[cm]
            ens = 0.5 * P["xgb_proba"].values[cm] + 0.5 * P["nn_proba"].values[cm]
            pra_x = _boot(lambda a, b: average_precision_score(a, b) if 0 < a.sum() < len(a) else None,
                          hab, P["xgb_proba"].values[cm])
            pra_e = _boot(lambda a, b: average_precision_score(a, b) if 0 < a.sum() < len(a) else None,
                          hab, ens)
            report[group][h] = {"skill_xgb": sk[1.0], "skill_nn": sk[0.0],
                                "skill_best": sk[best_w], "best_w": best_w,
                                "pr_auc_xgb": pra_x, "pr_auc_ensemble": pra_e}
            print(f"  +{h}d | XGB={sk[1.0][0]:+.3f}  NN={sk[0.0][0]:+.3f}  "
                  f"MEJOR(w={best_w})={sk[best_w][0]:+.3f} [{sk[best_w][1]:+.3f},{sk[best_w][2]:+.3f}] "
                  f"| PR-AUC alerta: XGB={pra_x[0]:.2f} ENSAMBLE={pra_e[0]:.2f}")
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"\nReporte -> {OUT}")
    print("w=1 solo XGB, w=0 solo red. Si MEJOR usa w intermedio y sube el skill -> la union ayuda.")


if __name__ == "__main__":
    main()
