"""
evaluate_nested.py — VALIDACION ANIDADA + TEST FINAL INTACTO (numeros definitivos, sin sesgo).

Por que existe:
  Las metricas "honestas" previas (evaluate_robust.py) siguen teniendo un sesgo de SELECCION:
  el set de features por horizonte (select_features_per_horizon.py) y el umbral de alerta
  (calibrate_alert.py) se eligieron MAXIMIZANDO skill sobre la MISMA serie OOS con la que
  luego se reporta. Eso infla el numero. Para la defensa hay que medir sobre datos que ninguna
  decision toco.

Que hace (protocolo anidado, por (grupo, horizonte)):
  1. CORTE TEMPORAL: DEV = primer (1-TEST_FRAC) del tiempo; TEST INTACTO = ultimo TEST_FRAC.
     Banda de EMBARGO (config.purge_days) entre DEV y TEST -> corta fuga via features
     autorregresivas (chl_t0/lag/roll usan target <= t0; el embargo evita solape DEV-target /
     TEST-feature).
  2. SELECCION INTERNA (solo DEV): ablacion de familias de features por ventana expansiva OOS
     DENTRO de DEV. El TEST nunca participa de la seleccion.
  3. REENTRENO en TODO DEV con las features elegidas -> se predice el TEST UNA sola vez.
  4. Metricas en TEST agrupadas por (grupo, horizonte) con IC95% bootstrap:
       - intensidad: SKILL de regresion vs persistencia (titular).
       - alerta: PR-AUC del ensamble XGB+Red (sin umbral -> sin fuga de calibracion).
  5. Tabla comparativa OPTIMISTA (OOS global, robust_metrics.json) vs ANIDADO (test intacto).

Reusa familias/_model/_clf de train.py, _fit de la red y _boot del protocolo bootstrap.
"""
from __future__ import annotations
import os, json, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import config as C
from train import SPECTRAL, AUTOREG, DYNAMICS, SEASONAL, ERA5, NUTRIENTS, WATERQUAL, _model, _clf, PAIRS
from train_nn import _fit
from evaluate_robust import _boot

OUT = os.path.join(C.DIR_REPORTS, "nested_metrics.json")
ROBUST = os.path.join(C.DIR_REPORTS, "robust_metrics.json")
PRED_DUMP = os.path.join(C.DIR_REPORTS, "nested_test_predictions.csv")  # para figuras de validacion

TEST_FRAC = 0.25            # ultimo 25% del tiempo = test intacto (~2025-2026)
PURGE_DAYS = C.VALIDATION["purge_days"]   # embargo entre DEV y TEST
N_INNER = 3                 # folds de la ventana expansiva interna (solo DEV)
MIN_TRAIN_FRAC = 0.45       # arranque de la ventana expansiva interna
MIN_TEST = 8                # test minimo por celda para reportar
MIN_DEV = 40

# Familias candidatas para la ablacion interna (AUTOREG = backbone, siempre presente).
# DYNAMICS (dinamica temporal) y SEASONAL (estacionalidad/climatologia) entran como OPCIONALES:
# la seleccion interna decide POR HORIZONTE si suben el skill; si no, se descartan.
FAMILIES = {"ERA5": ERA5, "SPECTRAL": SPECTRAL, "INSITU": NUTRIENTS + WATERQUAL,
            "DYNAMICS": DYNAMICS, "SEASONAL": SEASONAL}
# Default: familias ADOPTADAS. DYNAMICS/SEASONAL se probaron (HABS_NEWFEATS=1) pero el control
# anidado mostro que NO mejoran de forma robusta (lavado) -> fuera del default.
OPTIONAL = ["ERA5", "SPECTRAL", "INSITU"]
if os.environ.get("HABS_NEWFEATS"):
    OPTIONAL = OPTIONAL + ["DYNAMICS", "SEASONAL"]


def _skill(y, yhat, per):
    rp = np.sqrt(mean_squared_error(y, per))
    return 1 - np.sqrt(mean_squared_error(y, yhat)) / rp if rp > 0 else np.nan


def _subsets():
    for r in range(len(OPTIONAL) + 1):
        for c in itertools.combinations(OPTIONAL, r):
            yield list(c)


def _feats_of(combo, avail):
    feats = list(AUTOREG)
    for fam in combo:
        feats += FAMILIES[fam]
    return [f for f in feats if f in avail]


def _inner_oos_skill(dev, feats):
    """Skill OOS de ventana expansiva DENTRO de DEV (por cuerpo, agrupado). Solo XGB."""
    ys, yh, yp = [], [], []
    for _, g in dev.groupby("water_body"):
        g = g.sort_values("fecha_t0").reset_index(drop=True)
        N = len(g); start = int(N * MIN_TRAIN_FRAC)
        if N - start < N_INNER * 4:
            continue
        fold = (N - start) // N_INNER
        for k in range(N_INNER):
            a = start + k * fold
            b = N if k == N_INNER - 1 else a + fold
            tr, te = g.iloc[:a], g.iloc[a:b]
            if len(te) < 3 or len(tr) < 20:
                continue
            m = _model().fit(tr[feats], tr["log_chl_target"])
            ys.append(te["log_chl_target"].values); yh.append(m.predict(te[feats]))
            yp.append(te["log_chl_t0"].values)
    if not ys:
        return np.nan
    y, h, p = np.concatenate(ys), np.concatenate(yh), np.concatenate(yp)
    return _skill(y, h, p)


SELECT_MARGIN = 0.02      # regla de parsimonia (analoga a 1-SE): tolerancia de skill DEV

def inner_select(dev, avail):
    """Elige familias por skill OOS interno (solo DEV) CON PARSIMONIA: entre las combinaciones
    cuyo skill cae dentro de SELECT_MARGIN del mejor, se queda con la de MENOS familias (y a
    igualdad, menos features). Evita que la seleccion agarre familias que apenas mejoran en DEV
    y no transfieren al test (sobreajuste de seleccion)."""
    scored = []
    for combo in _subsets():
        feats = _feats_of(combo, avail)
        sk = _inner_oos_skill(dev, feats)
        scored.append((sk if np.isfinite(sk) else -9.0, combo))
    best = max(s for s, _ in scored)
    cands = [(len(c), len(_feats_of(c, avail)), c) for s, c in scored if s >= best - SELECT_MARGIN]
    cands.sort()                                   # menos familias -> menos features
    best_combo = cands[0][2]
    return ["AUTOREG"] + best_combo, _feats_of(best_combo, avail)


def split_dev_test(d):
    """Corte temporal por (grupo,horizonte): DEV (pasado) / TEST intacto (futuro) con embargo."""
    d = d.sort_values("fecha_t0")
    cut = d["fecha_t0"].quantile(1 - TEST_FRAC)
    embargo = cut + pd.Timedelta(days=PURGE_DAYS)
    dev = d[d["fecha_t0"] <= cut]
    test = d[d["fecha_t0"] > embargo]      # banda de embargo descartada
    return dev, test, cut


def nn_proba(dev, test, feats):
    """Probabilidad de alerta de la red (entrenada en DEV), imputada+escalada sin fuga."""
    # keep_empty_features: cuerpos sin in-situ (Honduras) tienen columnas all-NaN; conservarlas
    # mantiene n_in consistente con la red (las imputa a 0 en vez de eliminarlas).
    imp = SimpleImputer(keep_empty_features=True).fit(dev[feats])
    sc = StandardScaler().fit(imp.transform(dev[feats]))
    Xtr = sc.transform(imp.transform(dev[feats])); Xte = sc.transform(imp.transform(test[feats]))
    net = _fit(Xtr, dev["log_chl_target"].values, dev["hab_target"].values.astype(float), len(feats))
    with torch.no_grad():
        _, pc = net(torch.tensor(Xte, dtype=torch.float32))
    return torch.sigmoid(pc).numpy()


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    avail = set(df.columns)
    optimistic = json.load(open(ROBUST)) if os.path.exists(ROBUST) else {}
    report = {}
    dump_rows = []                          # predicciones del TEST intacto (para figuras)
    print("PROTOCOLO ANIDADO: features elegidas SOLO en DEV; TEST intacto evaluado una vez.")
    print(f"TEST = ultimo {int(TEST_FRAC*100)}% del tiempo | embargo {PURGE_DAYS} d entre DEV y TEST\n")

    for group in ("freshwater", "marine"):
        print(f"############  {group}  —  TEST FINAL INTACTO  ############")
        report[group] = {}
        for h in [x for x in C.HORIZONS if x != 0]:
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            # 1) split temporal POR CUERPO (cada cuerpo tiene su propio calendario), pero
            #    2) seleccion de features y entrenamiento con el DEV AGRUPADO de todos los cuerpos
            #    -> una sola decision por (grupo,horizonte): reduce la varianza de seleccion y
            #    se alinea con produccion (train_final entrena por grupo). El TEST sigue intacto.
            devs, tests, cutdates = [], [], []
            for wb, g in d.groupby("water_body"):
                dev, test, cut = split_dev_test(g)
                if len(dev) < MIN_DEV or len(test) < MIN_TEST:
                    continue
                devs.append(dev); tests.append(test)
                cutdates.append(str(pd.Timestamp(cut).date()))
            if not devs:
                print(f"  +{h}d  (sin celdas con DEV/TEST suficientes)")
                continue
            DEV = pd.concat(devs, ignore_index=True)
            TEST = pd.concat(tests, ignore_index=True)
            fams, feats = inner_select(DEV, avail)      # UNA seleccion sobre DEV agrupado
            chosen = {"_grupo": "+".join(fams)}
            # --- intensidad: XGB entrenado en DEV agrupado -> predice TEST intacto ---
            reg = _model().fit(DEV[feats], DEV["log_chl_target"])
            test_pred = reg.predict(TEST[feats])
            pooled_y = [TEST["log_chl_target"].values]
            pooled_yh = [test_pred]
            pooled_per = [TEST["log_chl_t0"].values]
            # volcado de predicciones del TEST intacto (mismo split/features) para validar visualmente
            td = TEST[["group", "horizon", "water_body", "fecha_t0",
                       "log_chl_target", "log_chl_t0"]].copy()
            td["pred_log"] = test_pred
            td["chl_real"] = np.expm1(td["log_chl_target"])
            td["chl_pred"] = np.clip(np.expm1(td["pred_log"]), 0, None)
            td["chl_persist"] = np.expm1(td["log_chl_t0"])
            dump_rows.append(td)
            # --- alerta: ensamble XGB-clf + Red (DEV agrupado) -> PR-AUC en TEST ---
            pooled_hab, pooled_xgbp, pooled_nnp = [], [], []
            if DEV["hab_target"].nunique() > 1:
                xp = _clf(DEV["hab_target"].values).fit(DEV[feats], DEV["hab_target"]).predict_proba(TEST[feats])[:, 1]
                npb = nn_proba(DEV, TEST, feats)
                pooled_hab.append(TEST["hab_target"].values)
                pooled_xgbp.append(xp); pooled_nnp.append(npb)
            y = np.concatenate(pooled_y); yh = np.concatenate(pooled_yh); per = np.concatenate(pooled_per)
            m = np.isfinite(per)
            skill = _boot(_skill, y[m], yh[m], per[m])

            pra = (np.nan,)*3; n_pos = 0
            if pooled_hab:
                hab = np.concatenate(pooled_hab)
                ens = 0.5 * np.concatenate(pooled_xgbp) + 0.5 * np.concatenate(pooled_nnp)
                n_pos = int(hab.sum())
                if 0 < n_pos < len(hab):
                    pra = _boot(lambda a, b: average_precision_score(a, b) if 0 < a.sum() < len(a) else None,
                                hab, ens)

            opt = optimistic.get(group, {}).get(str(h), {}).get("skill_reg", [None])
            opt_sk = opt[0] if isinstance(opt, list) else None
            report[group][h] = {
                "n_test": int(len(y)), "pos_test": n_pos,
                "skill_nested": skill, "pr_auc_nested": pra,
                "skill_optimistic_oos": opt_sk,
                "features_per_body": chosen, "test_cutoff": cutdates,
            }
            opt_str = f"{opt_sk:+.2f}" if isinstance(opt_sk, (int, float)) else "  n/a"
            print(f"  +{h}d  n_test={len(y):>3} eventos={n_pos:>2} | "
                  f"SKILL anidado={skill[0]:+.2f} [{skill[1]:+.2f},{skill[2]:+.2f}]  "
                  f"(optimista OOS={opt_str}) | PR-AUC alerta={pra[0]:.2f} [{pra[1]:.2f},{pra[2]:.2f}]")
        print()

    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    if dump_rows:
        pd.concat(dump_rows, ignore_index=True).to_csv(PRED_DUMP, index=False)
        print(f"Predicciones TEST intacto -> {PRED_DUMP}")
    print(f"Reporte -> {OUT}")
    print("Lectura: el SKILL anidado es el numero DEFENDIBLE (test nunca tocado). "
          "Si sigue >0 con IC que no cruza 0 -> skill real, no artefacto de seleccion.")


if __name__ == "__main__":
    main()
