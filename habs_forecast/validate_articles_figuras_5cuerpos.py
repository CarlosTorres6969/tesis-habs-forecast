"""
validate_articles_figuras_5cuerpos.py — Figura de PREDICCIÓN para los 5 cuerpos, para validar el
modelo contra los artículos: imagen S2 real + mapa de biomasa del modelo + recuadro con el pronóstico
de las 23 variables (chl a +3/+7 d, nivel, prob de alerta).

- Yojoa: escena histórica (2020-2022) + pronóstico 23-var de forecast_historical_yojoa.csv (fuera de
  ventana de entrenamiento).
- Okeechobee, Cajón, Fonseca, Tampa: escena in-window de imagenes/ + forecast_body real (las 23
  variables ya existen en 2023-2026).
- Marino (Fonseca, Tampa): el MAPA por píxel no refleja intensidad de marea roja (limitación
  declarada); el recuadro muestra igual el pronóstico/alerta del modelo.

Salida: artifacts/validation_articles/figuras_prediccion/pred_<cuerpo>_<fecha>.png
"""
from __future__ import annotations
import os, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from make_maps import build_map_figure, KEY2FOLDER, _clear_water_score
from predict import forecast_body

OUTD = os.path.join(C.DIR_OUT, "validation_articles", "figuras_prediccion")
os.makedirs(OUTD, exist_ok=True)
YHIST = os.path.join(C.DIR_OUT, "validation_articles", "s2_historico", "yojoa")
YFC = os.path.join(C.DIR_OUT, "validation_articles", "forecast_historical_yojoa.csv")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
MARINE = {"tampa_bay", "fonseca"}
NIVEL_ES = {"floracion": "FLORACIÓN", "elevada": "biomasa ELEVADA", "normal": "normal"}

# cuerpo, fecha objetivo, etiqueta (artículo/evento)
EVENTS = [
    ("okeechobee", "2024-03-22", "Microcistina>EPA (FDEP); cyanoHABs Frontiers Microbiology 2023 (Q1)"),
    ("cajon",      "2024-03-04", "Episodio mayor documentado (sin artículo revisado por pares)"),
    ("fonseca",    "2023-03-14", "Diatomeas abundantes LABTOX-UES (marino: mapa NO=intensidad)"),
    ("tampa_bay",  "2025-01-17", "Marea roja post-Milton FWC (marino: mapa NO=intensidad)"),
]


def _best_in_window(folder, target, win=25):
    tifs = glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True)
    t = pd.Timestamp(target); cands = []
    for p in tifs:
        m = DATE_RE.search(os.path.basename(p))
        if m and abs((pd.Timestamp(m.group(1)) - t).days) <= win:
            cands.append(p)
    if not cands:
        return None, None
    best = max(cands, key=_clear_water_score)
    return best, DATE_RE.search(os.path.basename(best)).group(1)


NAMES = {"yojoa": "Lago de Yojoa (Honduras)", "cajon": "Embalse El Cajón (Honduras)",
         "okeechobee": "Lago Okeechobee (Florida, EE.UU.)", "tampa_bay": "Bahía de Tampa (Florida, EE.UU.)",
         "fonseca": "Golfo de Fonseca (Honduras)"}
MES_ES = {1:"ene",2:"feb",3:"mar",4:"abr",5:"may",6:"jun",7:"jul",8:"ago",9:"sep",10:"oct",11:"nov",12:"dic"}
NIVEL_UP = {"floracion": "FLORACIÓN", "elevada": "BIOMASA ELEVADA", "normal": "NORMAL"}


def _fecha_es(d):
    t = pd.Timestamp(d); return f"{t.day} {MES_ES[t.month]} {t.year}"


def _caption(fig, wb, chl3, n3, chl7, n7, chl0):
    """Pie limpio (una línea, sin recuadro) con el pronóstico de las 23 variables."""
    seg = (f"Pronóstico del modelo · 23 variables         "
           f"+3 d:  {chl3:.0f} µg/L  {NIVEL_UP.get(n3, n3)}          "
           f"+7 d:  {chl7:.0f} µg/L  {NIVEL_UP.get(n7, n7)}")
    nota = ("marino: el mapa por píxel no refleja la marea roja; la evidencia es la alerta"
            if wb in MARINE else f"clorofila reciente observada (t0): {chl0:.0f} µg/L")
    # fuera del area de los paneles (bbox_inches='tight' incluye la banda inferior)
    fig.text(0.5, -0.04, seg, ha="center", fontsize=11, color="#1b1f27", fontweight="bold")
    fig.text(0.5, -0.075, nota, ha="center", fontsize=9, color="#6a7280", style="italic")


def _titulo(fig, wb, d, lab):
    # título limpio ARRIBA (Caso Título), sin tapar los títulos de panel de build_map_figure
    fig.suptitle(f"{NAMES[wb]}  ·  {_fecha_es(d)}", fontsize=14, fontweight="bold", y=1.06)
    fig.text(0.5, 1.005, lab, ha="center", fontsize=10, color="#1f6f54", transform=fig.transFigure)


def _fig_yojoa():
    fc = pd.read_csv(YFC); fc["fecha"] = fc["fecha"].astype(str)
    d = "2021-05-26"; lab = "Metatranscriptómica · mSystems 2024  (fecha fuera de entrenamiento)"
    tifs = glob.glob(os.path.join(YHIST, f"*{d}*.tif"))
    if not tifs:
        print("  yojoa: sin escena"); return
    # MISMA llamada que app.py: build_map_figure(..., gradient_focus=True)
    fig, _ = build_map_figure("yojoa", 3, tifs[0], t0=None, gradient_focus=True)
    r = fc[fc["fecha"] == d].iloc[0]
    _caption(fig, "yojoa", r["chl_pred_h3"], r["nivel_h3"], r["chl_pred_h7"], r["nivel_h7"],
             r["chl0_autoreg_ugL"])
    _titulo(fig, "yojoa", d, lab)
    out = os.path.join(OUTD, f"pred_yojoa_{d}.png")
    fig.savefig(out, dpi=175, bbox_inches="tight"); plt.close(fig)
    print(f"  yojoa {d}: -> {os.path.basename(out)}")


def main():
    _fig_yojoa()
    for wb, target, lab in EVENTS:
        folder = KEY2FOLDER[wb]
        path, d = _best_in_window(folder, target)
        if path is None:
            print(f"  {wb}: sin escena en ventana"); continue
        t0 = pd.Timestamp(d)
        try:
            # MISMA llamada que app.py (gradient_focus=True para todos)
            fig, _ = build_map_figure(wb, 3, path, t0, gradient_focus=True)
        except Exception as e:
            print(f"  {wb} {d}: {e}"); continue
        fc = forecast_body(wb, t0)
        if fc is None:
            print(f"  {wb} {d}: sin forecast"); plt.close(fig); continue
        hz = {h["horizon"]: h for h in fc["horizons"]}
        h3, h7 = hz.get(3), hz.get(7)
        _caption(fig, wb, h3["chl_pred"], h3["nivel"], h7["chl_pred"], h7["nivel"], fc["chl0"])
        _titulo(fig, wb, d, lab)
        out = os.path.join(OUTD, f"pred_{wb}_{d}.png")
        fig.savefig(out, dpi=175, bbox_inches="tight"); plt.close(fig)
        print(f"  {wb} {d}: -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
