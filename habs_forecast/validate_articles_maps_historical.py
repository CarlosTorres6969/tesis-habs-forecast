"""
validate_articles_maps_historical.py — VÍA D: aplica el MODELO (cabeza de regresión espectral por
píxel) a escenas Sentinel-2 REALES de Yojoa en las fechas de los artículos (2020-2021), generando el
mapa de biomasa que el modelo deriva de esas imágenes. Es "correr el modelo en la fecha del artículo".

Reusa build_map_figure(path=...) SIN tocar imagenes/ ni el modelado. t0=None: en esa era no hay
features no-espectrales (ERA5/insitu/target del modelo), así que el mapa usa la señal ESPECTRAL de la
escena (h=3, con gradiente espacial). Muestra el CAMPO de biomasa (patrón), no un pronóstico píxel a
píxel anclado a autorregresión.

Salida: artifacts/validation_articles/mapas_historicos/mapa_yojoa_<fecha>.png + historical_maps.csv
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
OUTD = os.path.join(C.DIR_OUT, "validation_articles", "mapas_historicos")
os.makedirs(OUTD, exist_ok=True)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# etiqueta de la ventana del artículo a la que pertenece cada fecha
def window_label(d):
    if "2020-1" in d or "2020-12" in d:  return "Huracanes Eta/Iota nov-2020 (Fadum 2023, Sci Rep)"
    if d.startswith("2021-0") and d[5:7] in ("05","06","07"): return "Metatranscriptómica jun-2021 (mSystems 2024)"
    if d.startswith("2022-0") or d.startswith("2021-12"): return "Muestreo ene-2022 (mSystems 2024)"
    return "otra"


def main():
    tifs = sorted(glob.glob(os.path.join(SCENES, "*.tif")))
    if not tifs:
        print(f"No hay escenas en {SCENES} — corre fetch_s2_historical_yojoa.py"); return
    print(f"{len(tifs)} escenas históricas encontradas")
    rows = []
    for path in tifs:
        m = DATE_RE.search(os.path.basename(path))
        d = m.group(1) if m else "????"
        try:
            fig, stats = build_map_figure("yojoa", 3, path, t0=None, gradient_focus=True)
        except Exception as e:
            print(f"  {d}: descartada ({str(e)[:60]})"); continue
        rows.append({"fecha": d, "ventana_articulo": window_label(d),
                     "chl_media_ugL": round(stats["chl_mean"], 1),
                     "area_biomasa_alta_pct": round(stats["pct_elev"], 0),
                     "area_floracion_pct": round(stats["pct_alert"], 0),
                     "umbral_ugL": round(stats["thr"], 1),
                     "n_water_px": stats["n_water_px"], "path_png": ""})
        out = os.path.join(OUTD, f"mapa_yojoa_{d}.png")
        fig.suptitle(f"Lago de Yojoa — {d}\nmodelo aplicado a escena S2 real | "
                     f"{window_label(d)}", fontsize=10)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        rows[-1]["path_png"] = out
        print(f"  {d}: chl media {stats['chl_mean']:.1f} ug/L | "
              f"agua {stats['n_water_px']} px | biomasa alta {stats['pct_elev']:.0f}% -> {out}")

    if not rows:
        print("Ninguna escena util."); return
    df = pd.DataFrame(rows).sort_values(["ventana_articulo", "n_water_px"], ascending=[True, False])
    csvp = os.path.join(C.DIR_OUT, "validation_articles", "historical_maps.csv")
    df.to_csv(csvp, index=False)
    print(f"\n-> {csvp}")
    print("\n=== mejor escena (más agua) por ventana de artículo ===")
    best = df.groupby("ventana_articulo").first().reset_index()
    print(best[["ventana_articulo", "fecha", "chl_media_ugL", "area_biomasa_alta_pct",
                "n_water_px"]].to_string(index=False))


if __name__ == "__main__":
    main()
