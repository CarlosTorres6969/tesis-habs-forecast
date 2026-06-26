"""
train_final.py — Entrena y GUARDA los modelos de produccion (para despliegue en predict.py).

Para cada (grupo, horizonte) entrena con TODOS los pares y guarda:
  - XGBoost regresor (intensidad log-chl)         -> reg
  - XGBoost clasificador de alerta                -> clf
  - Red neuronal (HABNet) + imputador + escalador -> nn
  - lista de features de ese horizonte
El sistema de ALERTA en produccion = ensamble (XGB_clf + NN) [promedio de probas].
Tambien guarda el umbral de alerta por cuerpo (thr_body).

Salida: artifacts/models/{grupo}_h{h}.pkl  y  {grupo}_h{h}_nn.pt  + thr_body.pkl
"""
from __future__ import annotations
import os, joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import config as C
from train import get_features, _model, _clf, PAIRS
from train_nn import _fit

MODELS = C.DIR_MODELS
HORIZONS = [1, 3, 5, 7]
QLO, QHI, NOMINAL = 0.10, 0.90, 0.80      # intervalos de incertidumbre (CQR P10-P90)


def _qmodel(alpha):
    from xgboost import XGBRegressor
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                        random_state=C.RANDOM_STATE, n_jobs=4)


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    os.makedirs(MODELS, exist_ok=True)
    # umbral de alerta por cuerpo (climatologia relativa)
    thr_body = df.groupby("water_body")["thr_body"].first().to_dict()
    joblib.dump(thr_body, os.path.join(MODELS, "thr_body.pkl"))

    for group in ("freshwater", "marine"):
        for h in HORIZONS:
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            if len(d) < 40:
                continue
            feats = get_features(group, h, d.columns)
            reg = _model().fit(d[feats], d["log_chl_target"])
            clf = None
            if d["hab_target"].nunique() > 1:
                clf = _clf(d["hab_target"].values).fit(d[feats], d["hab_target"])
            imp = SimpleImputer().fit(d[feats])
            sc = StandardScaler().fit(imp.transform(d[feats]))
            X = sc.transform(imp.transform(d[feats]))
            net = _fit(X, d["log_chl_target"].values, d["hab_target"].values.astype(float), len(feats))

            # intervalos de incertidumbre (CQR): P10/P90 + offset conformal validado (cobertura ~0.80).
            # Se ajusta en TRAIN (pasado) y se calibra Q en el 25% mas reciente (split-conformal).
            ds = d.sort_values("fecha_t0")
            ncal = max(int(len(ds) * 0.25), 20)
            tr, cal = ds.iloc[:-ncal], ds.iloc[-ncal:]
            qlo_m = _qmodel(QLO).fit(tr[feats], tr["log_chl_target"])
            qhi_m = _qmodel(QHI).fit(tr[feats], tr["log_chl_target"])
            elo, ehi = qlo_m.predict(cal[feats]), qhi_m.predict(cal[feats])
            yc = cal["log_chl_target"].values
            E = np.maximum(elo - yc, yc - ehi)
            Qc = float(np.quantile(E, min(1.0, NOMINAL * (1 + 1.0 / len(E)))))

            tag = f"{group}_h{h}"
            joblib.dump({"reg": reg, "clf": clf, "feats": feats, "imp": imp, "sc": sc,
                         "n_in": len(feats), "group": group, "horizon": h,
                         "qlo": qlo_m, "qhi": qhi_m, "q_conformal": Qc},
                        os.path.join(MODELS, f"{tag}.pkl"))
            torch.save(net.state_dict(), os.path.join(MODELS, f"{tag}_nn.pt"))
            print(f"  guardado {tag}: {len(d)} pares, {len(feats)} features")
    print(f"\nModelos de produccion -> {MODELS}")


if __name__ == "__main__":
    main()
