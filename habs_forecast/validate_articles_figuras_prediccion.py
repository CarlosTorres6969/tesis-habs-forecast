"""
validate_articles_figuras_prediccion.py — Figuras donde SE VE la prediccion del modelo sobre la
imagen satelital real de la fecha del articulo: panel 1 (imagen S2 real) + panel 2 (mapa de biomasa
del modelo) + recuadro con el PRONOSTICO de las 23 variables (chl a +3/+7 d, nivel, prob de alerta).

Une build_map_figure (imagen + mapa) con forecast_historical_yojoa.csv (modelo completo 23 var).
Salida: artifacts/validation_articles/figuras_prediccion/pred_yojoa_<fecha>.png
"""
from __future__ import annotations
import os, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from make_maps import build_map_figure

SCENES = os.path.join(C.DIR_OUT, "validation_articles", "s2_historico", "yojoa")
FC = os.path.join(C.DIR_OUT, "validation_articles", "forecast_historical_yojoa.csv")
OUTD = os.path.join(C.DIR_OUT, "validation_articles", "figuras_prediccion")
os.makedirs(OUTD, exist_ok=True)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# fechas clave alineadas a los articulos (las mejores escenas de cada ventana)
KEY = {
    "2020-11-12": "Huracanes Eta/Iota — Fadum 2023 (Sci Reports)",
    "2021-05-26": "Muestreo metatranscriptomica — mSystems 2024",
    "2021-05-31": "Muestreo metatranscriptomica — mSystems 2024",
    "2022-01-16": "Muestreo ene-2022 — mSystems 2024",
}
NIVEL_ES = {"floracion": "FLORACION", "elevada": "biomasa ELEVADA", "normal": "normal"}


def main():
    fc = pd.read_csv(FC)
    fc["fecha"] = fc["fecha"].astype(str)
    for d, label in KEY.items():
        tifs = glob.glob(os.path.join(SCENES, f"*{d}*.tif"))
        if not tifs:
            print(f"  {d}: sin escena"); continue
        path = tifs[0]
        t0 = pd.Timestamp(d)
        try:
            fig, stats = build_map_figure("yojoa", 3, path, t0=None, gradient_focus=True, hq=True)
        except Exception as e:
            print(f"  {d}: {e}"); continue
        # recuadro con el pronostico REAL de 23 variables
        r = fc[fc["fecha"] == d]
        if len(r):
            r = r.iloc[0]
            txt = (f"PRONOSTICO DEL MODELO (23 variables, fecha fuera de entrenamiento)\n"
                   f"  +3 dias:  {r['chl_pred_h3']:.0f} ug/L  ->  {NIVEL_ES.get(r['nivel_h3'], r['nivel_h3'])}"
                   f"   (prob. alerta {r['prob_alerta_h3']:.2f})\n"
                   f"  +7 dias:  {r['chl_pred_h7']:.0f} ug/L  ->  {NIVEL_ES.get(r['nivel_h7'], r['nivel_h7'])}"
                   f"   (prob. alerta {r['prob_alerta_h7']:.2f})\n"
                   f"  clorofila reciente (VIIRS) en t0: {r['chl0_autoreg_ugL']:.0f} ug/L")
            fig.text(0.5, -0.02, txt, ha="center", va="top", fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round", fc="#fff3cd", ec="#8a1c34"))
        fig.suptitle(f"Lago de Yojoa — {d}\n{label}", fontsize=11)
        out = os.path.join(OUTD, f"pred_yojoa_{d}.png")
        fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"  {d}: -> {out}")


if __name__ == "__main__":
    main()
