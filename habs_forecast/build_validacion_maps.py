"""
build_validacion_maps.py — Genera un mapa del modelo por cada fecha VALIDADA con evidencia
externa (ver VALIDACION_EXTERNA_HABS.md). Reusa build_map_figure (no duplica modelado).
Salida: entregables/validacion/mapa_val_<wb>_<fecha>.png
"""
from __future__ import annotations
import os, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from make_maps import build_map_figure, KEY2FOLDER

OUT = os.path.join(C.BASE, "habs_forecast", "entregables", "validacion")
os.makedirs(OUT, exist_ok=True)

# (clave, cuerpo, fecha de escena S2 elegida, horizonte, etiqueta del evento)
# Escenas Sentinel-2 elegidas por MAXIMA cobertura de agua limpia DENTRO del episodio validado
# (para que el mapa muestre el GRADIENTE espacial de clorofila, no parches por nubes).
JOBS = [
    ("fonseca",    "fonseca",    "2023-03-02", 3, "Fonseca E1 — muestreo LABTOX-UES 14-mar-2023 (diatomeas abundantes; sin marea roja toxica); escena despejada a -12d"),
    ("cajon_e4",   "cajon",      "2026-01-25", 3, "Cajon E4 — episodio mayor sostenido (jun-2025 a abr-2026); escena despejada ene-2026; sin monitoreo de campo publico"),
    ("cajon_e1",   "cajon",      "2024-03-06", 3, "Cajon E1 — floracion de epoca seca (pico 4-mar-2024, 57 ug/L); escena despejada, enfocada al embalse; sin monitoreo de campo publico"),
    ("yojoa_e4",   "yojoa",      "2024-03-01", 3, "Yojoa E4 — episodio MAYOR de epoca seca (pico 18-ene-2024, 96 ug/L); escena despejada"),
    ("yojoa_e1",   "yojoa",      "2023-03-02", 3, "Yojoa E1 — 'Intervenido Lago de Yojoa' por exceso de algas (La Tribuna, 20-feb-2023)"),
    ("yojoa_e7",   "yojoa",      "2024-11-06", 3, "Yojoa E7 — mortandad de peces por hipoxia/algas (3-5 nov-2024); unica escena del episodio (algo neblinosa)"),
    ("okeechobee_e3","okeechobee","2024-03-15", 3, "Okeechobee E3 — microcistina > umbral EPA (FDEP, 22-28 mar-2024)"),
    ("okeechobee_e6","okeechobee","2025-08-02", 3, "Okeechobee E6 — floracion de Microcystis verano 2025 (FAU/USF); escena despejada ago-2025"),
    ("tampa_e1",   "tampa_bay",  "2023-01-13", 3, "Tampa Bay E1 — marea roja K. brevis ene-2023 (FWC). NOTA marina: usar la ALERTA, no la intensidad del mapa"),
    ("tampa_e9",   "tampa_bay",  "2025-01-12", 3, "Tampa Bay E9 — marea roja de invierno post-Milton (oct-2024 -> feb-2025, FWC/WUSF). NOTA marina: usar la ALERTA, no la intensidad del mapa"),
]


def pick_tif(wb, date):
    folder = KEY2FOLDER[wb]
    cands = glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", f"*{date}*.tif"), recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getsize)   # escena con mas datos (mayor archivo)


def main():
    for key, wb, date, h, label in JOBS:
        path = pick_tif(wb, date)
        if not path:
            print(f"[SKIP] {key}: sin raster {date}")
            continue
        t0 = pd.Timestamp(date)
        try:
            fw = key in ("cajon_e1",)   # enfocar al cuerpo de agua principal (rio/embalse angosto)
            fig, stats = build_map_figure(wb, h, path, t0, gradient_focus=True, focus_water=fw)
        except Exception as e:
            print(f"[FAIL] {key} ({date}): {e}")
            continue
        # nota de validacion como pie de figura
        fig.text(0.5, 0.005, "VALIDACION EXTERNA: " + label, ha="center", fontsize=9,
                 style="italic", wrap=True)
        out = os.path.join(OUT, f"mapa_val_{key}_{date}.png")
        fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
        print(f"[OK] {key}: {out} | chl media={stats['chl_mean']:.1f} ug/L | "
              f"area floracion={stats['pct_alert']:.0f}% | px_agua={stats['n_water_px']}")


if __name__ == "__main__":
    main()
