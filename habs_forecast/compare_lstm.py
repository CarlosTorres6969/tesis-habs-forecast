"""
compare_lstm.py — Comparacion con LSTM (torch). Cierra la evaluacion de arquitecturas.

Diseno justo: LSTM sobre la secuencia reciente de clorofila [log(chl) en t0-7, t0-3, t0]
concatenada con features estaticos (espectral + ERA5) en una capa densa final -> predice
log(chl) en t0+h. Mismo protocolo OOS expansivo por cuerpo (agua dulce) que el resto.

Hipotesis (Fase 1): con N~2600 y secuencias cortas/irregulares por nubosidad, las redes
recurrentes no se justifican. Se espera skill <= XGBoost.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import config as C
from train import FEATURES, SPECTRAL, ERA5, PAIRS
from evaluate_robust import _boot, N_FOLDS, MIN_TRAIN_FRAC

torch.manual_seed(C.RANDOM_STATE)
SEQ = ["chl_lag7", "chl_lag3", "chl_t0"]          # secuencia temporal (3 pasos)
STATIC = SPECTRAL + ERA5                            # contexto estatico


class LSTMHybrid(nn.Module):
    def __init__(self, n_static, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, seq, stat):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], stat], dim=1)).squeeze(-1)


def _prep(tr, te, stat_cols):
    imp = SimpleImputer().fit(tr[stat_cols]); sc = StandardScaler().fit(imp.transform(tr[stat_cols]))
    def X(d):
        seq = np.log1p(np.clip(d[SEQ].values, 0, None))[:, :, None].astype("float32")
        st = sc.transform(imp.transform(d[stat_cols])).astype("float32")
        return torch.tensor(seq), torch.tensor(st)
    return X(tr), X(te)


def _fit_predict(tr, te, stat_cols):
    (seq_tr, st_tr), (seq_te, st_te) = _prep(tr, te, stat_cols)
    y_tr = torch.tensor(tr["log_chl_target"].values.astype("float32"))
    net = LSTMHybrid(n_static=len(stat_cols))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    lossf = nn.MSELoss()
    net.train()
    for _ in range(150):
        opt.zero_grad()
        loss = lossf(net(seq_tr, st_tr), y_tr)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(seq_te, st_te).numpy()


def oos_lstm(d, stat_cols):
    ys, yh, yp = [], [], []
    for _, g in d.groupby("water_body"):
        g = g.sort_values("fecha_t0").reset_index(drop=True)
        N = len(g); start = int(N * MIN_TRAIN_FRAC)
        if N - start < N_FOLDS * 4:
            continue
        fold = (N - start) // N_FOLDS
        for k in range(N_FOLDS):
            a = start + k * fold
            b = N if k == N_FOLDS - 1 else a + fold
            tr, te = g.iloc[:a], g.iloc[a:b]
            if len(te) < 3 or len(tr) < 20:
                continue
            yh.append(_fit_predict(tr, te, stat_cols))
            ys.append(te["log_chl_target"].values); yp.append(te["log_chl_t0"].values)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(yp)


def _skill(y, yhat, persist):
    rp = np.sqrt(mean_squared_error(y, persist))
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / rp if rp > 0 else np.nan


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    stat = [c for c in STATIC if c in df.columns]
    d = df[(df["group"] == "freshwater") & (df["horizon"].isin([1, 3, 5, 7]))]
    print("######  LSTM hibrido (secuencia chl + estatico) — agua dulce  ######")
    res = oos_lstm(d, stat)
    if res:
        y, yh, yp = res
        sk = _boot(_skill, y, yh, yp)
        star = " *" if sk[1] > 0 else ""
        print(f"  LSTM: skill={sk[0]:+.3f} [{sk[1]:+.3f},{sk[2]:+.3f}]{star}  (n={len(y)})")
    print("  (comparar con XGBoost +0.168 [+0.135,+0.200] de compare_models.py)")


if __name__ == "__main__":
    main()
