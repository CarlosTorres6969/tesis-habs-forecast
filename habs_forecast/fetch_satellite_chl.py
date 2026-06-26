"""
fetch_satellite_chl.py — TARGET satelital diario de clorofila (2023-2026) via NASA/CoastWatch
ERDDAP (VIIRS SNPP, chlor_a diario, sin credenciales).

Para cada cuerpo de agua agrega los pixeles validos del bbox a una MEDIANA DIARIA -> serie
de target densa e independiente de Sentinel-2 (otro sensor) para construir pares 0-7 d.

Cobertura: optima en costa (Fonseca, Tampa) y Okeechobee (lago grande). Para lagos pequenos
(Yojoa, Cajon) la resolucion ~750 m es marginal -> complementar con Sentinel-3 OLCI (ver
fetch_olci_chl.py). Salida: artifacts/targets/satellite_chl_daily.csv
"""
from __future__ import annotations
import os, ssl, io, urllib.request, time
import numpy as np
import pandas as pd
import config as C

OUT_DIR = os.path.join(C.DIR_OUT, "targets")
OUT = os.path.join(OUT_DIR, "satellite_chl_daily.csv")

ERDDAP = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaDaily.csv"
          "?chlor_a%5B({t0}):1:({t1})%5D%5B(0.0)%5D%5B({la_hi}):1:({la_lo})%5D"
          "%5B({lo_lo}):1:({lo_hi})%5D")

# bbox por cuerpo: (lat_lo, lat_hi, lon_lo, lon_hi) y grupo ecologico
BODIES = {
    "okeechobee":     (26.70, 27.20, -81.10, -80.60, "freshwater"),
    "tampa_bay":      (27.50, 27.95, -82.75, -82.40, "marine"),
    "yojoa":          (14.78, 14.95, -88.02, -87.90, "freshwater"),
    "cajon":          (14.70, 14.95, -87.80, -87.58, "freshwater"),
    "fonseca":        (12.90, 13.45, -87.85, -87.35, "marine"),
}
T0, T1 = "2023-01-01", "2026-04-30"


def _get(url, tries=4):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=300, context=ctx).read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))   # backoff ante 502/timeout transitorio


def _fetch_body(name, la_lo, la_hi, lo_lo, lo_hi):
    # trocear por anio evita 502 del proxy en peticiones largas
    chunks = []
    for yr in (2023, 2024, 2025, 2026):
        t0 = f"{yr}-01-01"; t1 = f"{yr}-12-31" if yr < 2026 else "2026-04-30"
        url = ERDDAP.format(t0=t0, t1=t1, la_hi=la_hi, la_lo=la_lo, lo_lo=lo_lo, lo_hi=lo_hi)
        try:
            raw = _get(url)
        except Exception as e:
            print(f"      {name} {yr}: {type(e).__name__}")
            continue
        d = pd.read_csv(io.BytesIO(raw), skiprows=[1])
        chunks.append(d)
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
        try:
            d = _fetch_body(name, la_lo, la_hi, lo_lo, lo_hi)
            d["group"] = group
            frames.append(d)
            print(f"  {name:12s}: {len(d):>4} dias con dato | "
                  f"chl mediana={d['chl_ugl'].median():.2f} ug/L" if len(d) else
                  f"  {name:12s}: sin datos validos")
        except Exception as e:
            print(f"  {name:12s}: FALLO {type(e).__name__}: {e}")
        time.sleep(1)
    if not frames:
        print("Sin datos."); return
    out = pd.concat(frames, ignore_index=True).sort_values(["water_body", "fecha"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nTarget satelital diario -> {OUT} ({len(out)} dias-cuerpo)")

    print("\n=== potencial 0-7d (gaps consecutivos por cuerpo) ===")
    for name, g in out.groupby("water_body"):
        gaps = g["fecha"].drop_duplicates().sort_values().diff().dropna().dt.days
        n7 = int((gaps <= 7).sum())
        print(f"  {name:12s}: {len(g):>4} dias | pares gap<=7d = {n7}")


if __name__ == "__main__":
    build()
