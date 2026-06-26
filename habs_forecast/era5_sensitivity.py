"""
era5_sensitivity.py — SENSIBILIDAD a ERA5: reanalisis (entrenamiento) vs pronostico (operacion).

Honestidad operativa. El modelo se entreno y valido con ERA5 *reanalisis* (hindcast "perfecto",
con latencia de ~5 dias). En operacion real, para pronosticar t0+h se dispondria de ERA5 de
*pronostico/tiempo casi-real*, que trae error. Pregunta: cuanto depende el skill de ERA5 y cuanto
se degradaria si en vez de reanalisis se alimentara ERA5 de calidad-pronostico?

Dos analisis complementarios (XGB, ventana expansiva OOS por grupo, mismo protocolo que el resto):

  (A) ABLACION (definitiva): skill CON ERA5 vs SIN ERA5 (resto de familias igual).
      Cota superior del aporte de ERA5. Si el aporte es pequeno -> la distincion
      reanalisis/pronostico es casi irrelevante (buena noticia operativa).

  (B) ESTRES de error de pronostico: en INFERENCIA se perturban SOLO las columnas ERA5 con
      ruido gaussiano de desviacion = NIVEL * std(driver en entrenamiento). Es un proxy
      DOCUMENTADO de la degradacion reanalisis->pronostico (el error de un pronostico a 1-7 d
      crece como una fraccion de la variabilidad del driver). Barrido de NIVEL -> curva de
      degradacion del skill. El modelo se entrena con datos limpios (reanalisis) y solo cambia
      la calidad del input en prediccion, que es exactamente el escenario operativo.

Limitacion declarada: el ruido gaussiano e independiente es una aproximacion; el error real de
pronostico esta correlacionado en el tiempo y entre variables. Es una cota de robustez, no una
replica del producto de pronostico.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
import config as C
from train import FEATURES, ERA5, _model, PAIRS
from evaluate_robust import _boot, N_FOLDS, MIN_TRAIN_FRAC

OUT = os.path.join(C.DIR_REPORTS, "era5_sensitivity.json")
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]   # fraccion de la std del driver (proxy error pronostico)
RNG = np.random.default_rng(C.RANDOM_STATE)


def _skill(y, yhat, per):
    rp = np.sqrt(mean_squared_error(y, per))
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / rp if rp > 0 else np.nan


def _folds(d):
    """Indices de ventana expansiva a nivel de grupo (mismo esquema que evaluate_robust)."""
    d = d.sort_values("fecha_t0").reset_index(drop=True)
    N = len(d); start = int(N * MIN_TRAIN_FRAC)
    if N - start < N_FOLDS * 4:
        return d, []
    fold = (N - start) // N_FOLDS
    spans = []
    for k in range(N_FOLDS):
        a = start + k * fold
        b = N if k == N_FOLDS - 1 else a + fold
        if b - a >= 3 and a >= 30:
            spans.append((a, b))
    return d, spans


def ablation_oos(d, feats):
    """Predicciones OOS limpias para un set de features dado (para comparar CON vs SIN ERA5)."""
    d, spans = _folds(d)
    ys, yh, yp = [], [], []
    for a, b in spans:
        tr, te = d.iloc[:a], d.iloc[a:b]
        m = _model().fit(tr[feats], tr["log_chl_target"])
        ys.append(te["log_chl_target"].values); yh.append(m.predict(te[feats]))
        yp.append(te["log_chl_t0"].values)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(yp)


def noise_oos(d, feats, era5_cols):
    """Entrena limpio; predice el test con ERA5 perturbado a varios niveles de ruido."""
    d, spans = _folds(d)
    if not spans:
        return None
    cols = [c for c in era5_cols if c in feats]
    store = {lv: {"y": [], "yh": [], "yp": []} for lv in NOISE_LEVELS}
    for a, b in spans:
        tr, te = d.iloc[:a], d.iloc[a:b]
        m = _model().fit(tr[feats], tr["log_chl_target"])
        sigma = {c: float(np.nanstd(tr[c].values)) for c in cols}
        for lv in NOISE_LEVELS:
            Xte = te[feats].copy()
            if lv > 0:
                for c in cols:
                    s = sigma[c]
                    if s > 0:
                        Xte[c] = Xte[c].values + RNG.normal(0.0, lv * s, size=len(Xte))
            store[lv]["y"].append(te["log_chl_target"].values)
            store[lv]["yh"].append(m.predict(Xte))
            store[lv]["yp"].append(te["log_chl_t0"].values)
    out = {}
    for lv in NOISE_LEVELS:
        out[lv] = (np.concatenate(store[lv]["y"]), np.concatenate(store[lv]["yh"]),
                   np.concatenate(store[lv]["yp"]))
    return out


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    feats_full = [f for f in FEATURES if f in df.columns]
    era5_cols = [c for c in ERA5 if c in df.columns]
    feats_noera5 = [f for f in feats_full if f not in era5_cols]
    report = {}
    print(f"ERA5 features presentes: {era5_cols}\n")

    for group in ("freshwater", "marine"):
        print(f"############  {group}  —  sensibilidad ERA5  ############")
        report[group] = {}
        for h in [x for x in C.HORIZONS if x != 0]:
            d = df[(df["group"] == group) & (df["horizon"] == h)]

            # (A) ABLACION CON vs SIN ERA5
            full = ablation_oos(d, feats_full)
            noe = ablation_oos(d, feats_noera5)
            if full is None or noe is None:
                print(f"  +{h}d  (datos insuficientes)")
                continue
            sk_full = _boot(_skill, *full)
            sk_noe = _boot(_skill, *noe)

            # (B) ESTRES de ruido en ERA5
            noise = noise_oos(d, feats_full, era5_cols)
            curve = {}
            for lv in NOISE_LEVELS:
                y, yh, yp = noise[lv]
                curve[lv] = _boot(_skill, y, yh, yp)

            report[group][h] = {
                "skill_con_era5": sk_full, "skill_sin_era5": sk_noe,
                "aporte_era5": sk_full[0] - sk_noe[0],
                "ruido_curva": {str(lv): curve[lv] for lv in NOISE_LEVELS},
            }
            print(f"  +{h}d | ABLACION  con ERA5={sk_full[0]:+.3f} [{sk_full[1]:+.3f},{sk_full[2]:+.3f}]  "
                  f"sin ERA5={sk_noe[0]:+.3f} [{sk_noe[1]:+.3f},{sk_noe[2]:+.3f}]  "
                  f"-> aporte={sk_full[0]-sk_noe[0]:+.3f}")
            curve_str = "  ".join(f"{int(lv*100):>3}%={curve[lv][0]:+.3f}" for lv in NOISE_LEVELS)
            print(f"        ESTRES ruido ERA5 (skill):  {curve_str}")
        print()

    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(f"Reporte -> {OUT}")
    print("Lectura: 'aporte' pequeno y curva de ruido PLANA -> el sistema NO es fragil al cambio")
    print("         reanalisis->pronostico (ERA5 aporta poco / de forma robusta). Curva que se")
    print("         desploma con ruido -> dependencia critica de ERA5 de alta calidad (riesgo operativo).")


if __name__ == "__main__":
    main()
