"""
validate_articles_historical.py — Compara el TARGET satelital histórico (VIIRS 2018-2022,
FUERA de la ventana del modelo) con los hallazgos de la literatura Q1 que muestreó en esa era,
sobre todo en Lago de Yojoa.

Objetivo honesto: los artículos (Fadum 2023 Sci Rep; metatranscriptómica mSystems 2024 muestreo
jun-2021 + ene-2022; Reyes-Avila 2024) NO dan una serie de chl-a diaria comparable píxel a píxel,
pero SÍ establecen (a) estado trófico eu/hipereutrófico, (b) dominancia estacional de cianobacterias,
(c) estratificación tropical con biomasa superficial. Aquí verificamos que el proxy de biomasa
satelital en 2018-2022 sea CONSISTENTE con esas afirmaciones y CONTINUO con la serie 2023-2026 que
alimenta al modelo (misma fuente VIIRS, mismo bbox).

Salida: artifacts/validation_articles/historical_summary.csv + fig_yojoa_serie_2018_2026.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

TDIR = os.path.join(C.DIR_OUT, "targets")
HIST = os.path.join(TDIR, "historical_target_honduras_2018_2022.csv")
CUR = os.path.join(TDIR, "combined_target.csv")
OUTD = os.path.join(C.DIR_OUT, "validation_articles")
os.makedirs(OUTD, exist_ok=True)

# Fechas de muestreo reales de los artículos (para marcar en la serie)
SAMPLINGS = {
    "yojoa": [
        ("2020-11-05", "Huracanes Eta/Iota (Fadum 2023, Sci Rep)"),
        ("2021-06-15", "Muestreo metatranscriptómica (mSystems 2024)"),
        ("2022-01-15", "Muestreo ene-2022 (mSystems 2024)"),
    ],
}


def trophic(chl):
    if chl < 2.6:  return "oligotrófico"
    if chl < 8:    return "mesotrófico"
    if chl < 25:   return "eutrófico"
    return "hipereutrófico"


def main():
    if not os.path.exists(HIST):
        print(f"FALTA {HIST} — corre primero fetch_historical_target.py"); return
    h = pd.read_csv(HIST, parse_dates=["fecha"])
    h["fecha"] = h["fecha"].dt.tz_localize(None)
    h["periodo"] = "historico_2018_2022"
    cur = pd.read_csv(CUR, parse_dates=["fecha"])
    cur["fecha"] = cur["fecha"].dt.tz_localize(None)
    cur = cur[cur["water_body"].isin(h["water_body"].unique())].copy()
    cur["periodo"] = "modelo_2023_2026"
    allser = pd.concat([h[["water_body", "fecha", "chl_ugl", "periodo"]],
                        cur[["water_body", "fecha", "chl_ugl", "periodo"]]], ignore_index=True)

    rows = []
    for wb, g in allser.groupby("water_body"):
        for per, gp in g.groupby("periodo"):
            chl = gp["chl_ugl"]
            rows.append({
                "cuerpo": wb, "periodo": per, "n_dias": len(gp),
                "rango": f"{gp['fecha'].min().date()}..{gp['fecha'].max().date()}",
                "chl_media": round(float(chl.mean()), 1),
                "chl_mediana": round(float(chl.median()), 1),
                "chl_p90": round(float(chl.quantile(0.9)), 1),
                "estado_trofico": trophic(float(chl.median())),
            })
    summ = pd.DataFrame(rows).sort_values(["cuerpo", "periodo"])
    outp = os.path.join(OUTD, "historical_summary.csv")
    summ.to_csv(outp, index=False)
    print(summ.to_string(index=False))
    print(f"\n-> {outp}")

    # continuidad de nivel entre eras (test de que la serie no salta artificialmente)
    print("\n=== Continuidad histórico vs modelo (mediana chl-a) ===")
    for wb, g in summ.groupby("cuerpo"):
        med = g.set_index("periodo")["chl_mediana"].to_dict()
        if len(med) == 2:
            a, b = med.get("historico_2018_2022"), med.get("modelo_2023_2026")
            print(f"  {wb:10s}: 2018-22={a} ug/L | 2023-26={b} ug/L | "
                  f"mismo estado trófico: {trophic(a)==trophic(b)}")

    # Figura Yojoa: serie completa 2018-2026 con muestreos de artículos marcados
    y = allser[allser["water_body"] == "yojoa"].sort_values("fecha")
    if len(y):
        fig, ax = plt.subplots(figsize=(14, 5))
        for per, c in [("historico_2018_2022", "#8d99ae"), ("modelo_2023_2026", "#2b8cbe")]:
            gp = y[y["periodo"] == per]
            ax.plot(gp["fecha"], gp["chl_ugl"], ".", ms=3, color=c, label=per, alpha=0.6)
        # media móvil 30d
        ys = y.set_index("fecha")["chl_ugl"].rolling("30D").median()
        ax.plot(ys.index, ys.values, "-", color="#d1495b", lw=1.5, label="mediana móvil 30d")
        thr = C.alert_threshold_ugl(float(y["chl_ugl"].quantile(0.85)))
        ax.axhline(thr, ls="--", color="k", lw=0.8, label=f"umbral alerta {thr:.0f} ug/L")
        for d, lab in SAMPLINGS.get("yojoa", []):
            ax.axvline(pd.Timestamp(d), color="green", ls=":", lw=1.2)
            ax.annotate(lab, (pd.Timestamp(d), ax.get_ylim()[1]*0.92), fontsize=7,
                        rotation=90, va="top", color="green")
        ax.axvspan(pd.Timestamp("2023-01-01"), y["fecha"].max(), alpha=0.05, color="blue")
        ax.set_title("Lago de Yojoa — target satelital VIIRS 2018-2026\n"
                     "(gris = era de los artículos, fuera de ventana; azul = ventana del modelo)")
        ax.set_ylabel("chl-a (ug/L)"); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        figp = os.path.join(OUTD, "fig_yojoa_serie_2018_2026.png")
        fig.savefig(figp, dpi=130, bbox_inches="tight")
        print(f"-> {figp}")


if __name__ == "__main__":
    main()
