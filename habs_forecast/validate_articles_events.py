"""
validate_articles_events.py — Cuantifica el CRUCE del estado de alerta del modelo con eventos
DOCUMENTADOS in-window (2023-2026) por agencias / literatura. Complementa (no reemplaza) la tabla
manual de VALIDACION_EXTERNA_HABS.md con una metrica reproducible.

Para cada evento documentado: dentro de +-WIN dias del rango del evento, se reporta el maximo del
target del modelo, si cruzo el umbral de alerta del cuerpo (min(p85,24)) y el nivel de biomasa.
Solo lee combined_target.csv. No reentrena.
"""
from __future__ import annotations
import os
import pandas as pd
import config as C

TGT = os.path.join(C.DIR_OUT, "targets", "combined_target.csv")
OUTD = os.path.join(C.DIR_OUT, "validation_articles")
os.makedirs(OUTD, exist_ok=True)
WIN = 15  # dias de tolerancia alrededor del evento

# Eventos documentados dentro de la ventana del modelo (fuente: agencias/literatura Q1/prensa
# ya citada en VALIDACION_EXTERNA_HABS.md y Articulos.md). type: agency|science|press
EVENTS = [
    # cuerpo, inicio, fin, etiqueta, fuente, tipo
    ("okeechobee", "2023-06-01", "2023-06-30", "Floracion cianobacterias (Landsat-9 ~380 mi2)", "NASA Earth Observatory", "agency"),
    ("okeechobee", "2024-03-22", "2024-03-28", "Microcistina > umbral EPA (17 ppb)", "Florida DEP", "agency"),
    ("okeechobee", "2025-06-01", "2025-09-30", "Floracion anual de Microcystis (verano)", "Frontiers in Water 2025 (Q1)", "science"),
    ("tampa_bay",  "2022-12-15", "2023-03-15", "Marea roja K. brevis (invierno)", "FWC / Tampa Bay Times", "agency"),
    ("tampa_bay",  "2024-10-10", "2025-02-15", "Marea roja invierno post-huracan Milton", "FWC / WUSF", "agency"),
    ("yojoa",      "2023-02-14", "2023-02-26", "Degradacion por exceso de algas ('Intervenido')", "La Tribuna (HN)", "press"),
    ("yojoa",      "2024-10-31", "2024-11-07", "Mortandad de peces por hipoxia + algas", "La Prensa/Tiempo/Proceso (HN)", "press"),
    ("fonseca",    "2023-03-10", "2023-03-20", "Diatomeas abundantes; SIN marea roja toxica", "LABTOX-UES / DIGEPESCA", "agency"),
]


def main():
    df = pd.read_csv(TGT, parse_dates=["fecha"])
    thr = {}
    for wb, g in df.groupby("water_body"):
        p85 = float(g["chl_ugl"].quantile(0.85))
        thr[wb] = C.alert_threshold_ugl(p85)

    rows = []
    for wb, ini, fin, label, src, typ in EVENTS:
        g = df[df["water_body"] == wb].copy()
        ini_w = pd.Timestamp(ini) - pd.Timedelta(days=WIN)
        fin_w = pd.Timestamp(fin) + pd.Timedelta(days=WIN)
        win = g[(g["fecha"] >= ini_w) & (g["fecha"] <= fin_w)]
        t = thr[wb]
        if len(win):
            cmax = float(win["chl_ugl"].max())
            fecha_max = win.loc[win["chl_ugl"].idxmax(), "fecha"].date()
            alerta = cmax >= t
            nivel = C.biomass_level(cmax, t)
            n_alert = int((win["chl_ugl"] >= t).sum())
        else:
            cmax, fecha_max, alerta, nivel, n_alert = float("nan"), None, False, "sin_dato", 0
        rows.append({
            "cuerpo": wb, "evento": label, "rango": f"{ini}..{fin}",
            "umbral_ugL": round(t, 1), "chl_max_modelo_ugL": round(cmax, 1) if cmax == cmax else None,
            "fecha_max": fecha_max, "dias_con_alerta_en_ventana": n_alert,
            "ALERTA": "SI" if alerta else "no", "nivel": nivel,
            "fuente": src, "tipo": typ,
        })
    out = pd.DataFrame(rows)
    outp = os.path.join(OUTD, "event_hits.csv")
    out.to_csv(outp, index=False)
    with pd.option_context("display.max_colwidth", 44, "display.width", 200):
        print(out[["cuerpo", "evento", "umbral_ugL", "chl_max_modelo_ugL",
                   "dias_con_alerta_en_ventana", "ALERTA", "nivel"]].to_string(index=False))
    hit = (out["ALERTA"] == "SI").sum()
    print(f"\nEventos con ALERTA del modelo dentro de +-{WIN} d: {hit}/{len(out)}")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
