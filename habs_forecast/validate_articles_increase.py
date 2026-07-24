"""
validate_articles_increase.py — Evidencia del AUMENTO de biomasa algal coincidente con las
investigaciones/eventos, para los 5 cuerpos. Para marino (Tampa, Fonseca) esta es la evidencia
principal (el mapa por píxel no refleja la intensidad).

Métrica de aumento: mediana de chl-a en la VENTANA del evento vs mediana en el BASELINE (90 días
previos a la ventana). Combina el target del modelo (2023-2026) con el histórico VIIRS (2018-2022
Honduras / 2019-2022 Florida) cuando existe.

Salida: artifacts/validation_articles/increase_summary.csv + fig_aumento_<cuerpo>.png
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
OUTD = os.path.join(C.DIR_OUT, "validation_articles")
os.makedirs(OUTD, exist_ok=True)

# ventanas de investigación/evento (cuerpo -> lista de (inicio, fin, etiqueta))
WINDOWS = {
    "yojoa": [
        ("2020-10-15", "2020-12-20", "Huracanes Eta/Iota (Fadum 2023)"),
        ("2021-05-15", "2021-07-20", "Metatranscriptómica (mSystems 2024)"),
        ("2024-10-20", "2024-11-15", "Mortandad de peces (prensa HN)"),
    ],
    "okeechobee": [
        ("2024-02-20", "2024-04-15", "Microcistina>EPA (FDEP 2024)"),
        ("2025-06-01", "2025-09-30", "Microcystis anual (Frontiers Water 2025)"),
    ],
    "cajon": [
        ("2024-02-11", "2024-03-15", "Episodio mayor (sin artículo)"),
    ],
    "fonseca": [
        ("2023-03-01", "2023-03-25", "Diatomeas LABTOX-UES 2023"),
    ],
    "tampa_bay": [
        ("2022-12-20", "2023-03-01", "Marea roja K. brevis (FWC)"),
        ("2024-10-15", "2025-02-10", "Marea roja post-Milton (FWC)"),
    ],
}
HIST_FILES = ["historical_target_honduras_2018_2022.csv", "historical_target_florida_2019_2022.csv"]


def _load_all():
    frames = []
    cur = pd.read_csv(os.path.join(TDIR, "combined_target.csv"), parse_dates=["fecha"])
    cur["fecha"] = cur["fecha"].dt.tz_localize(None)
    frames.append(cur[["water_body", "fecha", "chl_ugl"]])
    for f in HIST_FILES:
        p = os.path.join(TDIR, f)
        if os.path.exists(p):
            h = pd.read_csv(p, parse_dates=["fecha"])
            h["fecha"] = h["fecha"].dt.tz_localize(None)
            frames.append(h[["water_body", "fecha", "chl_ugl"]])
    allser = pd.concat(frames, ignore_index=True).drop_duplicates(["water_body", "fecha"])
    return allser.sort_values(["water_body", "fecha"])


def main():
    allser = _load_all()
    names = {"yojoa": "Lago de Yojoa (HN)", "cajon": "Embalse El Cajón (HN)",
             "okeechobee": "Lago Okeechobee (US)", "tampa_bay": "Bahía de Tampa (US)",
             "fonseca": "Golfo de Fonseca (HN)"}
    rows = []
    for wb, wins in WINDOWS.items():
        g = allser[allser["water_body"] == wb].set_index("fecha")["chl_ugl"].sort_index()
        if not len(g):
            continue
        thr = C.alert_threshold_ugl(float(g.quantile(0.85)))
        fig, ax = plt.subplots(figsize=(13, 4.2))
        ax.plot(g.index, g.values, ".", ms=2.5, color="#8d99ae", alpha=0.5)
        roll = g.rolling("30D").median()
        ax.plot(roll.index, roll.values, "-", color="#2b8cbe", lw=1.3, label="mediana móvil 30d")
        ax.axhline(thr, ls="--", color="k", lw=0.8, label=f"umbral {thr:.0f} ug/L")
        for ini, fin, lab in wins:
            ini_t, fin_t = pd.Timestamp(ini), pd.Timestamp(fin)
            win = g[(g.index >= ini_t) & (g.index <= fin_t)]
            base = g[(g.index >= ini_t - pd.Timedelta(days=90)) & (g.index < ini_t)]
            m_win = float(win.median()) if len(win) else np.nan
            m_base = float(base.median()) if len(base) else np.nan
            ratio = (m_win / m_base) if (m_base and m_base > 0 and not np.isnan(m_win)) else np.nan
            ax.axvspan(ini_t, fin_t, alpha=0.18, color="#d1495b")
            ax.annotate(lab, (ini_t, ax.get_ylim()[1]*0.96), fontsize=6.5, rotation=90,
                        va="top", color="#8a1c34")
            rows.append({"cuerpo": wb, "ventana": f"{ini}..{fin}", "evento": lab,
                         "chl_baseline_90d": round(m_base, 1) if m_base==m_base else None,
                         "chl_ventana": round(m_win, 1) if m_win==m_win else None,
                         "aumento_x": round(ratio, 2) if ratio==ratio else None,
                         "supera_umbral": bool(m_win >= thr) if m_win==m_win else None})
        ax.set_title(f"{names.get(wb, wb)} — biomasa satelital y ventanas de investigación")
        ax.set_ylabel("chl-a (ug/L)"); ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        figp = os.path.join(OUTD, f"fig_aumento_{wb}.png")
        fig.savefig(figp, dpi=125, bbox_inches="tight"); plt.close(fig)
        print(f"  {wb}: figura -> {os.path.basename(figp)}")

    df = pd.DataFrame(rows)
    csvp = os.path.join(OUTD, "increase_summary.csv")
    df.to_csv(csvp, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n-> {csvp}")


if __name__ == "__main__":
    main()
