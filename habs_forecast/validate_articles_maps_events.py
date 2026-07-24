"""
validate_articles_maps_events.py — Aplica el MODELO a la escena Sentinel-2 real más limpia dentro de
la ventana de cada evento/investigación, para los 5 cuerpos (evidencia espacial de biomasa algal).

Usa escenas 2023-2026 ya presentes en imagenes/ (fechas in-window) => t0 REAL disponible => el mapa
usa el modelo completo (espectral + no-espectral). Reusa build_map_figure (no toca modelado).

Salida: artifacts/validation_articles/mapas_eventos/mapa_<cuerpo>_<fecha>.png + event_maps.csv

Nota honesta por cuerpo:
  - dulce (okeechobee, cajon): el mapa por píxel SÍ refleja la biomasa (cianobacterias -> chl-a alta).
  - marino (tampa_bay, fonseca): el mapa por píxel NO refleja la intensidad de marea roja/dinoflagelados
    (limitación documentada del proyecto); se genera con ADVERTENCIA. La evidencia marina de aumento
    está en la serie temporal del target (ver validate_articles_increase.py).
"""
from __future__ import annotations
import os, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from make_maps import build_map_figure, KEY2FOLDER, _clear_water_score

OUTD = os.path.join(C.DIR_OUT, "validation_articles", "mapas_eventos")
os.makedirs(OUTD, exist_ok=True)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
WIN = 25  # días alrededor del evento para buscar la mejor escena

MARINE = {"tampa_bay", "fonseca"}

# cuerpo, fecha objetivo, etiqueta (investigación/evento alineado)
EVENTS = [
    ("okeechobee", "2024-03-22", "Microcistina > umbral EPA (FDEP); cyanoHABs Frontiers Microbiology 2023"),
    ("okeechobee", "2025-08-13", "Floración anual de Microcystis (Frontiers in Water 2025, Q1)"),
    ("cajon",      "2024-03-04", "Episodio mayor documentado (SIN artículo revisado por pares)"),
    ("fonseca",    "2023-03-14", "Diatomeas abundantes LABTOX-UES (marino: mapa NO=intensidad)"),
    ("tampa_bay",  "2023-01-23", "Marea roja K. brevis (marino: mapa NO=intensidad)"),
    ("tampa_bay",  "2025-01-17", "Marea roja invierno post-Milton (marino: mapa NO=intensidad)"),
]


def _best_in_window(folder, target, win=WIN):
    tifs = glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True)
    t = pd.Timestamp(target)
    cands = []
    for p in tifs:
        m = DATE_RE.search(os.path.basename(p))
        if not m:
            continue
        d = pd.Timestamp(m.group(1))
        if abs((d - t).days) <= win:
            cands.append(p)
    if not cands:
        return None, None
    best, sc = max(((p, _clear_water_score(p)) for p in cands), key=lambda x: x[1])
    m = DATE_RE.search(os.path.basename(best))
    return best, m.group(1)


def main():
    rows = []
    for wb, target, label in EVENTS:
        folder = KEY2FOLDER[wb]
        path, d = _best_in_window(folder, target)
        if path is None:
            print(f"  {wb} {target}: sin escena en +-{WIN}d"); continue
        t0 = pd.Timestamp(d)
        try:
            fig, stats = build_map_figure(wb, 3, path, t0,
                                          gradient_focus=(wb not in MARINE))
        except Exception as e:
            print(f"  {wb} {d}: descartada ({str(e)[:60]})"); continue
        tag = " [MARINO: mapa no refleja intensidad]" if wb in MARINE else ""
        fig.suptitle(f"{wb} — {d}{tag}\n{label}", fontsize=9)
        out = os.path.join(OUTD, f"mapa_{wb}_{d}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        rows.append({"cuerpo": wb, "fecha_escena": d, "fecha_objetivo": target,
                     "tipo": "marino" if wb in MARINE else "dulce",
                     "chl_media_ugL": round(stats["chl_mean"], 1),
                     "area_biomasa_alta_pct": round(stats["pct_elev"], 0),
                     "area_floracion_pct": round(stats["pct_alert"], 0),
                     "evento": label, "png": out})
        print(f"  {wb:11s} {d}: chl media {stats['chl_mean']:.1f} ug/L | "
              f"biomasa alta {stats['pct_elev']:.0f}% -> {os.path.basename(out)}")
    if rows:
        df = pd.DataFrame(rows)
        csvp = os.path.join(C.DIR_OUT, "validation_articles", "event_maps.csv")
        df.to_csv(csvp, index=False)
        print(f"\n-> {csvp}")


if __name__ == "__main__":
    main()
