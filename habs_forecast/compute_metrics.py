"""
compute_metrics.py — RMSE y R2 del modelo sobre el TEST INTACTO (numeros crudos).

Lee las predicciones de la validacion anidada (artifacts/reports/nested_test_predictions.csv,
generado por evaluate_nested.py) y calcula RMSE y R2 por (grupo, horizonte) y global, en:
  - log-chl (espacio del modelo) y
  - ug/L (escala fisica interpretable).
Tambien el RMSE/R2 de la PERSISTENCIA (baseline) para contexto.

NB: R2 sobre el test intacto puede ser bajo o negativo a horizonte largo (es lo esperable en
pronostico de eventos raros/ruidosos); el SKILL vs persistencia es la metrica titular. Aqui se
dan RMSE/R2 para completar el cuadro estadistico.

Uso:  python compute_metrics.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import config as C

PRED = os.path.join(C.DIR_REPORTS, "nested_test_predictions.csv")
OUT = os.path.join(C.DIR_REPORTS, "rmse_r2_metrics.csv")


def _row(label, y, yhat, yper):
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    r2 = float(r2_score(y, yhat)) if len(y) > 1 and np.var(y) > 0 else float("nan")
    rmse_p = float(np.sqrt(mean_squared_error(y, yper)))
    return {"grupo_horizonte": label, "n": len(y),
            "RMSE": round(rmse, 3), "R2": round(r2, 3), "RMSE_persist": round(rmse_p, 3)}


def main():
    if not os.path.exists(PRED):
        print(f"Falta {PRED}. Corre evaluate_nested.py primero."); return
    d = pd.read_csv(PRED)
    grp_name = {"freshwater": "Lagos", "marine": "Costa"}

    rows_log, rows_ugl = [], []
    for g in ("freshwater", "marine"):
        for h in (1, 3, 5, 7):
            s = d[(d.group == g) & (d.horizon == h)]
            if len(s) < 3:
                continue
            lbl = f"{grp_name[g]} +{h}d"
            rows_log.append(_row(lbl, s.log_chl_target.values, s.pred_log.values, s.log_chl_t0.values))
            rows_ugl.append(_row(lbl, s.chl_real.values, s.chl_pred.values, s.chl_persist.values))
    # global por grupo y total
    for g in ("freshwater", "marine"):
        s = d[d.group == g]
        rows_log.append(_row(f"{grp_name[g]} (todos h)", s.log_chl_target.values, s.pred_log.values, s.log_chl_t0.values))
        rows_ugl.append(_row(f"{grp_name[g]} (todos h)", s.chl_real.values, s.chl_pred.values, s.chl_persist.values))
    rows_log.append(_row("GLOBAL (todo)", d.log_chl_target.values, d.pred_log.values, d.log_chl_t0.values))
    rows_ugl.append(_row("GLOBAL (todo)", d.chl_real.values, d.chl_pred.values, d.chl_persist.values))

    log_df = pd.DataFrame(rows_log); ugl_df = pd.DataFrame(rows_ugl)
    print("=" * 70)
    print("RMSE y R2 sobre el TEST INTACTO  —  espacio LOG-chl (el que modela XGBoost)")
    print("=" * 70)
    print(log_df.to_string(index=False))
    print("\n" + "=" * 70)
    print("RMSE y R2 sobre el TEST INTACTO  —  escala FISICA (ug/L)")
    print("=" * 70)
    print(ugl_df.to_string(index=False))

    out = pd.concat([log_df.assign(espacio="log_chl"), ugl_df.assign(espacio="ug/L")], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    print("\nLectura: el modelo gana cuando RMSE < RMSE_persist. R2 en log es la bondad de ajuste; "
          "en ug/L baja por la asimetria/picos. Metrica titular = SKILL (1 - RMSE/RMSE_persist).")


if __name__ == "__main__":
    main()
