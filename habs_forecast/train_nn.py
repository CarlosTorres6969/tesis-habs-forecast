"""
train_nn.py — RED NEURONAL multitarea para pronostico temprano de HAB (modelo central).

Aplica el titulo de la tesis: red neuronal sobre analisis multiespectral (+ clima + in-situ).
Disenada para el tamano de datos (~2600 pares) con regularizacion fuerte:

  Arquitectura (multitarea, un tronco compartido):
    entrada (espectral S2 + autorregresivo chl + ERA5 + in-situ)
      -> [Linear-BatchNorm-ReLU-Dropout] x2  (tronco)
      -> cabeza REGRESION  : log(chl) en t0+h   (intensidad)
      -> cabeza ALERTA     : prob. de floracion (sigmoide)
    perdida = MSE(intensidad) + lambda * BCE(alerta)

  Regularizacion: BatchNorm + Dropout 0.3 + weight_decay + EARLY STOPPING (val interna).
  Datos: estandarizados + imputados (fit solo en train, sin fuga).
  Evaluacion: ventana expansiva a nivel de grupo (mas datos para la red) + bootstrap IC95%,
  mismo protocolo que XGBoost -> comparacion justa. Modelo SEPARADO por horizonte.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, average_precision_score, recall_score
import config as C
from train import FEATURES, PAIRS
from evaluate_robust import _boot, N_FOLDS, MIN_TRAIN_FRAC

torch.manual_seed(C.RANDOM_STATE)
np.random.seed(C.RANDOM_STATE)
OUT = os.path.join(C.DIR_REPORTS, "nn_metrics.json")
LAMBDA_CLS = 0.5          # peso de la cabeza de alerta en la perdida
MAX_EPOCHS, PATIENCE = 400, 30


class HABNet(nn.Module):
    """Red multitarea: tronco compartido + cabeza de regresion y de alerta."""
    def __init__(self, n_in, hidden=(64, 32), p=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.reg_head = nn.Linear(d, 1)        # intensidad log-chl
        self.cls_head = nn.Linear(d, 1)        # logit de alerta

    def forward(self, x):
        z = self.trunk(x)
        return self.reg_head(z).squeeze(-1), self.cls_head(z).squeeze(-1)


def _fit(Xtr, ytr, ctr, n_in):
    """Entrena con early stopping sobre un 15% de validacion interna."""
    n = len(Xtr); idx = np.arange(n); rng = np.random.default_rng(C.RANDOM_STATE)
    rng.shuffle(idx)
    cut = int(n * 0.85)
    tr_i, va_i = idx[:cut], idx[cut:]
    Xt = torch.tensor(Xtr[tr_i], dtype=torch.float32)
    yt = torch.tensor(ytr[tr_i], dtype=torch.float32)
    ct = torch.tensor(ctr[tr_i], dtype=torch.float32)
    Xv = torch.tensor(Xtr[va_i], dtype=torch.float32)
    yv = torch.tensor(ytr[va_i], dtype=torch.float32)
    cv = torch.tensor(ctr[va_i], dtype=torch.float32)

    net = HABNet(n_in)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-3)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    best, best_state, wait = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
        net.train(); opt.zero_grad()
        pr, pc = net(Xt)
        loss = mse(pr, yt) + LAMBDA_CLS * bce(pc, ct)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vr, vc = net(Xv)
            vloss = (mse(vr, yv) + LAMBDA_CLS * bce(vc, cv)).item() if len(va_i) else loss.item()
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    return net


def group_oos(d, feats):
    """Ventana expansiva a nivel de grupo -> predicciones OOS (red)."""
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
        imp = SimpleImputer().fit(tr[feats]); sc = StandardScaler().fit(imp.transform(tr[feats]))
        Xtr = sc.transform(imp.transform(tr[feats]))
        Xte = sc.transform(imp.transform(te[feats]))
        net = _fit(Xtr, tr["log_chl_target"].values, tr["hab_target"].values.astype(float), len(feats))
        with torch.no_grad():
            pr, pc = net(torch.tensor(Xte, dtype=torch.float32))
        rows.append(pd.DataFrame({
            "y_log": te["log_chl_target"].values, "yhat_log": pr.numpy(),
            "persist_log": te["log_chl_t0"].values,
            "proba": torch.sigmoid(pc).numpy(), "hab": te["hab_target"].values,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _skill(y, yhat, per):
    rp = np.sqrt(mean_squared_error(y, per))
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / rp if rp > 0 else np.nan


def _recall(h, p):
    return recall_score(h, (p >= 0.5).astype(int), zero_division=0) if 0 < h.sum() < len(h) else None


def _prauc(h, p):
    return average_precision_score(h, p) if 0 < h.sum() < len(h) else None


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    feats = [f for f in FEATURES if f in df.columns]
    report = {}
    for group in ("freshwater", "marine"):
        print(f"\n############  RED NEURONAL — {group}  ############")
        report[group] = {}
        for h in [1, 3, 5, 7]:
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            P = group_oos(d, feats)
            if not len(P):
                continue
            m = np.isfinite(P["persist_log"])
            sk = _boot(_skill, P["y_log"].values[m], P["yhat_log"].values[m], P["persist_log"].values[m])
            cm = np.isfinite(P["proba"].values)
            rec = _boot(lambda a, b: _recall(a, b), P["hab"].values[cm], P["proba"].values[cm])
            pra = _boot(lambda a, b: _prauc(a, b), P["hab"].values[cm], P["proba"].values[cm])
            report[group][h] = {"n": int(len(P)), "skill": sk, "recall": rec, "pr_auc": pra}
            print(f"  +{h}d n={len(P):>4} | SKILL={sk[0]:+.3f} [{sk[1]:+.3f},{sk[2]:+.3f}] | "
                  f"Recall={rec[0]:.2f} | PR-AUC={pra[0]:.2f}")
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"\nReporte -> {OUT}")


if __name__ == "__main__":
    main()
