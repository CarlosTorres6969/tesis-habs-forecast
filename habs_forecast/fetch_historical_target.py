"""
fetch_historical_target.py — Extiende el TARGET satelital de clorofila hacia ATRÁS (2016-2022),
FUERA de la ventana 2023-2026 del modelo, para VALIDACIÓN EXTERNA contra los artículos revisados
por pares (que muestrearon en 2018-2022, antes de la ventana del modelo).

NO toca el modelado ni los números de validación interna. Salida separada:
    artifacts/targets/historical_target_2016_2022.csv

Fuente: NASA/CoastWatch ERDDAP (VIIRS SNPP chlor_a diario), SIN credenciales — igual patrón que
fetch_satellite_chl.py, solo cambia el rango temporal. VIIRS SNPP existe desde 2012, así que
2016-2022 está disponible.

Uso:
    python fetch_historical_target.py            # todos los cuerpos, 2016-2022
    HIST_T0=2018 HIST_T1=2022 python fetch_historical_target.py
"""
from __future__ import annotations
import os, ssl, io, urllib.request, time
import pandas as pd
import config as C

OUT_DIR = os.path.join(C.DIR_OUT, "targets")
OUT = os.path.join(OUT_DIR, "historical_target_2016_2022.csv")

ERDDAP = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaDaily.csv"
          "?chlor_a%5B({t0}):1:({t1})%5D%5B(0.0)%5D%5B({la_hi}):1:({la_lo})%5D"
          "%5B({lo_lo}):1:({lo_hi})%5D")

# mismos bbox/grupos que fetch_satellite_chl.py (consistencia con el target 2023-2026)
BODIES = {
    "okeechobee": (26.70, 27.20, -81.10, -80.60, "freshwater"),
    "tampa_bay":  (27.50, 27.95, -82.75, -82.40, "marine"),
    "yojoa":      (14.78, 14.95, -88.02, -87.90, "freshwater"),
    "cajon":      (14.70, 14.95, -87.80, -87.58, "freshwater"),
    "fonseca":    (12.90, 13.45, -87.85, -87.35, "marine"),
}
YR0 = int(os.environ.get("HIST_T0", "2016"))
YR1 = int(os.environ.get("HIST_T1", "2022"))
# Filtro opcional de cuerpos: HIST_BODIES="yojoa,fonseca" (default: todos)
_sel = [b.strip() for b in os.environ.get("HIST_BODIES", "").split(",") if b.strip()]
if _sel:
    BODIES = {b: v for b, v in BODIES.items() if b in _sel}
# Sufijo de salida por env para no pisar corridas distintas
_suf = os.environ.get("HIST_SUFFIX", "")
if _suf:
    OUT = os.path.join(OUT_DIR, f"historical_target_{_suf}.csv")


def _get(url, tries=4):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=300, context=ctx).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def _fetch_body(name, la_lo, la_hi, lo_lo, lo_hi):
    chunks = []
    for yr in range(YR0, YR1 + 1):
        t0 = f"{yr}-01-01"; t1 = f"{yr}-12-31"
        url = ERDDAP.format(t0=t0, t1=t1, la_hi=la_hi, la_lo=la_lo, lo_lo=lo_lo, lo_hi=lo_hi)
        try:
            raw = _get(url)
        except Exception as e:
            print(f"      {name} {yr}: {type(e).__name__}")
            continue
        try:
            d = pd.read_csv(io.BytesIO(raw), skiprows=[1])
        except Exception as e:
            print(f"      {name} {yr}: parse {type(e).__name__}")
            continue
        chunks.append(d)
        print(f"      {name} {yr}: {len(d)} filas crudas")
        time.sleep(0.5)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df["chlor_a"] = pd.to_numeric(df["chlor_a"], errors="coerce")
    df = df.dropna(subset=["chlor_a"])
    if not len(df):
        return pd.DataFrame()
    df["fecha"] = pd.to_datetime(df["time"]).dt.normalize()
    daily = df.groupby("fecha").agg(chl_ugl=("chlor_a", "median"),
                                    n_valid_px=("chlor_a", "size")).reset_index()
    daily["water_body"] = name
    return daily


def build():
    frames = []
    for name, (la_lo, la_hi, lo_lo, lo_hi, group) in BODIES.items():
        print(f"  {name} ...")
        try:
            d = _fetch_body(name, la_lo, la_hi, lo_lo, lo_hi)
            if len(d):
                d["group"] = group
                frames.append(d)
                # guardado incremental (sobrevive interrupciones)
                os.makedirs(OUT_DIR, exist_ok=True)
                pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
                print(f"  {name:12s}: {len(d):>4} días | chl mediana={d['chl_ugl'].median():.2f} ug/L "
                      f"| {d['fecha'].min().date()} -> {d['fecha'].max().date()} [guardado]")
            else:
                print(f"  {name:12s}: sin datos validos {YR0}-{YR1}")
        except Exception as e:
            print(f"  {name:12s}: FALLO {type(e).__name__}: {e}")
        time.sleep(1)
    if not frames:
        print("Sin datos."); return
    out = pd.concat(frames, ignore_index=True).sort_values(["water_body", "fecha"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nTarget histórico -> {OUT} ({len(out)} días-cuerpo)")


if __name__ == "__main__":
    build()
