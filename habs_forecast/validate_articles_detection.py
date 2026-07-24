"""
validate_articles_detection.py — VEREDICTO DE DETECCION: consolida el resultado de aplicar el modelo
a las imagenes Sentinel-2 reales de las fechas de las investigaciones (vias D y E) en una tabla clara
de "el modelo DETECTA floracion a partir de la imagen: SI / NO / no-detectable(marino)".

Regla de deteccion (a partir de la imagen, no del target satelital):
  - dulce  : DETECTA FLORACION si chl_media del mapa >= umbral del cuerpo (24 ug/L) O el area en
             floracion >= 50%.  DETECTA BIOMASA ELEVADA si supera el nivel elevado (< floracion).
  - marino : el mapa por pixel NO detecta la marea roja (limitacion documentada) -> "no-detectable
             por imagen; usar alerta".

Lee historical_maps.csv (Yojoa 2020-2022) + event_maps.csv (5 cuerpos). Salida: detection_verdict.csv
"""
from __future__ import annotations
import os
import pandas as pd
import config as C

D = os.path.join(C.DIR_OUT, "validation_articles")
MARINE = {"tampa_bay", "fonseca"}


def verdict(row):
    if row["tipo"] == "marino":
        return "no-detectable por imagen (marino; ver alerta)"
    if row["chl_media_ugL"] >= row["umbral_ugL"] or row["area_floracion_pct"] >= 50:
        return "DETECTA FLORACION"
    if row["area_biomasa_alta_pct"] >= 50:
        return "DETECTA BIOMASA ELEVADA"
    return "no detecta"


def main():
    frames = []
    # Yojoa historico (via D)
    h = pd.read_csv(os.path.join(D, "historical_maps.csv"))
    h["cuerpo"] = "yojoa"; h["tipo"] = "dulce"
    h = h.rename(columns={"fecha": "fecha_escena", "ventana_articulo": "evento"})
    frames.append(h[["cuerpo", "fecha_escena", "chl_media_ugL", "area_floracion_pct",
                     "area_biomasa_alta_pct", "tipo", "evento"]].assign(umbral_ugL=24.0))
    # 5 cuerpos eventos (via E)
    e = pd.read_csv(os.path.join(D, "event_maps.csv"))
    e["umbral_ugL"] = e["cuerpo"].map(lambda w: 5.7 if w == "fonseca" else 6.4 if w == "tampa_bay" else 24.0)
    frames.append(e[["cuerpo", "fecha_escena", "chl_media_ugL", "area_floracion_pct",
                     "area_biomasa_alta_pct", "tipo", "evento", "umbral_ugL"]])
    df = pd.concat(frames, ignore_index=True)
    df["VEREDICTO"] = df.apply(verdict, axis=1)

    out = os.path.join(D, "detection_verdict.csv")
    df.sort_values(["cuerpo", "fecha_escena"]).to_csv(out, index=False)

    # resumen por cuerpo
    print("=== VEREDICTO DE DETECCION (modelo aplicado a la imagen de la fecha de investigacion) ===\n")
    for wb, g in df.groupby("cuerpo"):
        det = g["VEREDICTO"].str.startswith("DETECTA").sum()
        tot = len(g)
        tipo = g["tipo"].iloc[0]
        chl_rng = f"{g['chl_media_ugL'].min():.0f}-{g['chl_media_ugL'].max():.0f}"
        if tipo == "marino":
            print(f"  {wb:11s} (marino): {tot} escenas, chl {chl_rng} ug/L -> "
                  f"NO detectable por imagen (marea roja no eleva chl-a; evidencia = alerta)")
        else:
            print(f"  {wb:11s} (dulce) : DETECTA floracion/biomasa en {det}/{tot} escenas | "
                  f"chl {chl_rng} ug/L")
    print(f"\n-> {out}")
    # detalle compacto dulce
    print("\n=== detalle (agua dulce) ===")
    fw = df[df["tipo"] == "dulce"].copy()
    print(fw[["cuerpo", "fecha_escena", "chl_media_ugL", "area_floracion_pct", "VEREDICTO"]]
          .sort_values(["cuerpo", "fecha_escena"]).to_string(index=False))


if __name__ == "__main__":
    main()
