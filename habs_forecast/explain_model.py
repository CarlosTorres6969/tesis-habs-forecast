"""
explain_model.py — EXPLICABILIDAD (SHAP) del modelo de INTENSIDAD en produccion.

Para cada (grupo, horizonte) carga el regresor XGBoost de produccion (train_final.py) y
calcula la importancia SHAP de cada feature sobre TODOS los pares del grupo/horizonte:
  - shap_importance(reg, X) -> DataFrame [feature, mean_abs_shap] (PURO/testeable).
El regresor de intensidad es el backbone del pronostico Y de la ALERTA operativa
(riesgo = chl_pred >= umbral del cuerpo), asi que explicar SU salida responde la pregunta
de tesis "¿que variables mandan en el pronostico?" de forma defendible.

Es ADITIVO: NO reentrena ni sobrescribe modelos; solo LEE artifacts/models/*.pkl y escribe
un reporte + figuras. Si falta shap, un modelo o los pares, degrada con gracia (avisa/omite).

Salida:
  artifacts/reports/shap_importance.csv           (grupo, horizonte, feature, mean_abs_shap, rank)
  artifacts/reports/figs_shap/shap_{grupo}_h{h}.png  (beeswarm por grupo/horizonte)

Uso:  python explain_model.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C
from train import PAIRS

MODELS = C.DIR_MODELS
OUT_CSV = os.path.join(C.DIR_REPORTS, "shap_importance.csv")
FIGDIR = os.path.join(C.DIR_REPORTS, "figs_shap")
HORIZONS = [1, 3, 5, 7]
MAX_SAMPLE = 2000                    # cota de filas para SHAP (TreeExplainer es exacto pero acotamos)


def shap_importance(reg, X):
    """Importancia SHAP global: media de |valor SHAP| por feature. PURO y testeable.
    reg: regresor de arbol (XGBoost). X: DataFrame con exactamente las features del modelo.
    Devuelve DataFrame [feature, mean_abs_shap] ordenado desc."""
    import shap
    expl = shap.TreeExplainer(reg)
    sv = expl.shap_values(X)                        # (n, n_features)
    imp = np.abs(np.asarray(sv)).mean(axis=0)
    return (pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def _beeswarm(reg, X, path, title):
    """Beeswarm SHAP (efecto y direccion por feature). Headless-safe (backend Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    sv = shap.TreeExplainer(reg).shap_values(X)
    shap.summary_plot(sv, X, show=False, max_display=12)
    fig = plt.gcf(); fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("Falta 'shap' (pip install shap). Omito la explicabilidad."); return
    import joblib
    if not os.path.exists(PAIRS):
        print(f"Faltan los pares ({PAIRS})."); return
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0"])
    os.makedirs(FIGDIR, exist_ok=True)
    grp_name = {"freshwater": "Lagos", "marine": "Costa"}

    rows = []
    for group in ("freshwater", "marine"):
        for h in HORIZONS:
            mp = os.path.join(MODELS, f"{group}_h{h}.pkl")
            if not os.path.exists(mp):
                continue
            b = joblib.load(mp)
            reg, feats = b["reg"], b["feats"]
            d = df[(df["group"] == group) & (df["horizon"] == h)]
            d = d[[f for f in feats if f in d.columns]].dropna(how="all")
            if d.empty or len(feats) == 0:
                continue
            X = d.reindex(columns=feats)              # mismo orden/columnas que el modelo
            if len(X) > MAX_SAMPLE:
                X = X.sample(MAX_SAMPLE, random_state=C.RANDOM_STATE)
            imp = shap_importance(reg, X)
            imp.insert(0, "horizon", h); imp.insert(0, "group", group)
            imp["rank"] = np.arange(1, len(imp) + 1)
            rows.append(imp)
            top = ", ".join(f"{r.feature}({r.mean_abs_shap:.3f})"
                            for r in imp.head(5).itertuples())
            print(f"{grp_name[group]:6s} +{h}d (n={len(X)}): {top}")
            try:
                _beeswarm(reg, X, os.path.join(FIGDIR, f"shap_{group}_h{h}.png"),
                          f"SHAP — {grp_name[group]} +{h}d (intensidad log-chl)")
            except Exception as e:                    # una figura que falle no debe abortar el reporte
                print(f"   (aviso: no se pudo graficar {group} h{h}: {e})")

    if not rows:
        print("No se encontraron modelos de produccion para explicar."); return
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(C.DIR_REPORTS, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nImportancia SHAP -> {OUT_CSV}")
    print(f"Figuras beeswarm  -> {FIGDIR}")
    print("\nLectura: mean_abs_shap = cuanto MUEVE cada feature el log-chl pronosticado (magnitud "
          "media del efecto). El beeswarm muestra ademas la DIRECCION (rojo=valor alto). Corto "
          "plazo suele dominar el autorregresivo (chl reciente); largo plazo, meteo/in-situ.")


if __name__ == "__main__":
    main()
