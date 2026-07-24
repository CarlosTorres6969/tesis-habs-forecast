"""
validate_articles_pattern.py — VALIDACIÓN DE PATRÓN contra la literatura revisada por pares.

No compara fecha-exacta (los artículos muestrearon 2018-2022, fuera de la ventana 2023-2026);
compara que el TARGET del modelo (proxy de biomasa satelital) REPRODUZCA la fenomenología que
esos artículos establecen como duradera para cada cuerpo:

  - Yojoa: picos en ÉPOCA SECA (dic-abr) por estratificación estable (Fadum 2023; Reyes-Avila 2024);
           estado trófico EU/HIPEREUTRÓFICO con dominancia de cianobacterias (metatranscriptómica 2024).
  - Okeechobee: floraciones de Microcystis ANUALES en verano (Frontiers in Water 2025).
  - Tampa: mareas rojas K. brevis casi ANUALES (Yao 2023; early-warning 2024).
  - Fonseca: biomasa de fitoplancton presente; sin confirmación de toxicidad (LABTOX 2023).

Salida: artifacts/validation_articles/pattern_summary.csv  + figuras PNG.
Solo lee artifacts/targets/combined_target.csv (serie 2023-2026). No reentrena, no toca modelado.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

TGT = os.path.join(C.DIR_OUT, "targets", "combined_target.csv")
OUTD = os.path.join(C.DIR_OUT, "validation_articles")
os.makedirs(OUTD, exist_ok=True)

# Estacionalidad esperada por la literatura (meses de pico declarados en los artículos)
EXPECTED = {
    "yojoa":      {"season": "seca (dic-abr)", "peak_months": [12, 1, 2, 3, 4]},
    "cajon":      {"season": "seca (dic-abr)", "peak_months": [12, 1, 2, 3, 4]},
    "okeechobee": {"season": "verano (jun-sep)", "peak_months": [6, 7, 8, 9]},
    "tampa_bay":  {"season": "otoño-invierno (sep-feb)", "peak_months": [9, 10, 11, 12, 1, 2]},
    "fonseca":    {"season": "transición seca-lluvia", "peak_months": [3, 4, 5]},
}

# Estado trófico por clorofila-a media (umbrales OCDE/Carlson estándar, ug/L)
def trophic(chl):
    if chl < 2.6:   return "oligotrófico"
    if chl < 8:     return "mesotrófico"
    if chl < 25:    return "eutrófico"
    return "hipereutrófico"


def main():
    df = pd.read_csv(TGT, parse_dates=["fecha"])
    df["month"] = df["fecha"].dt.month
    df["year"] = df["fecha"].dt.year

    rows = []
    for wb, g in df.groupby("water_body"):
        exp = EXPECTED.get(wb, {})
        chl = g["chl_ugl"]
        # climatología mensual (mediana robusta)
        clim = g.groupby("month")["chl_ugl"].median()
        # meses top-3 por mediana
        top3 = list(clim.sort_values(ascending=False).head(3).index)
        peak_hit = sum(m in exp.get("peak_months", []) for m in top3)
        # fracción de la biomasa alta (>p75) que cae en la temporada esperada
        p75 = chl.quantile(0.75)
        hi = g[g["chl_ugl"] >= p75]
        frac_in_season = float(hi["month"].isin(exp.get("peak_months", [])).mean()) if len(hi) else np.nan
        # recurrencia anual: en cuántos años hay al menos un mes con mediana alta
        yrs = sorted(g["year"].unique())
        rows.append({
            "cuerpo": wb,
            "chl_media_ugL": round(float(chl.mean()), 1),
            "chl_mediana_ugL": round(float(chl.median()), 1),
            "chl_p90_ugL": round(float(chl.quantile(0.90)), 1),
            "estado_trofico": trophic(float(chl.median())),
            "temporada_esperada": exp.get("season", "?"),
            "meses_pico_modelo": top3,
            "coincide_pico_(0-3)": peak_hit,
            "frac_biomasa_alta_en_temporada": round(frac_in_season, 2),
            "anios_cubiertos": f"{yrs[0]}-{yrs[-1]}",
        })
    out = pd.DataFrame(rows)
    outp = os.path.join(OUTD, "pattern_summary.csv")
    out.to_csv(outp, index=False)
    print(out.to_string(index=False))
    print(f"\n-> {outp}")

    # Figura: climatología mensual normalizada por cuerpo, resaltando temporada esperada
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    order = ["yojoa", "cajon", "okeechobee", "tampa_bay", "fonseca"]
    names = {"yojoa": "Lago de Yojoa (HN)", "cajon": "Embalse El Cajón (HN)",
             "okeechobee": "Lago Okeechobee (US)", "tampa_bay": "Bahía de Tampa (US)",
             "fonseca": "Golfo de Fonseca (HN)"}
    for ax, wb in zip(axes.ravel(), order):
        g = df[df["water_body"] == wb]
        clim = g.groupby("month")["chl_ugl"].median().reindex(range(1, 13))
        exp = EXPECTED.get(wb, {})
        colors = ["#d1495b" if m in exp.get("peak_months", []) else "#66a5ad"
                  for m in range(1, 13)]
        ax.bar(range(1, 13), clim.values, color=colors)
        ax.set_title(f"{names[wb]}\nesperado: {exp.get('season','?')}", fontsize=10)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["E","F","M","A","M","J","J","A","S","O","N","D"], fontsize=8)
        ax.set_ylabel("chl-a mediana (ug/L)", fontsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Climatología mensual del target del modelo (2023-2026)\n"
                 "rojo = meses de pico según la literatura revisada por pares", fontsize=12)
    fig.tight_layout()
    figp = os.path.join(OUTD, "fig_climatologia_vs_literatura.png")
    fig.savefig(figp, dpi=130, bbox_inches="tight")
    print(f"-> {figp}")


if __name__ == "__main__":
    main()
