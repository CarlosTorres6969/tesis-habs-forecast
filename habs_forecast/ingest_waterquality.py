"""
ingest_waterquality.py — CONTEXTO in-situ de calidad de agua (WQP, publico, sin credenciales).

En 2023-2026 Florida tiene densidad alta de: temperatura del agua, OD, pH, turbidez,
conductividad, Secchi, amonio. Honduras sin cobertura WQP (-> NaN, XGBoost lo maneja).

Se ingiere por bbox de cada cuerpo, mediana diaria, formato ancho (una col por variable).
Salida: artifacts/targets/waterquality_daily.csv
    water_body, fecha, water_temp, do_mgl, ph, turbidity_insitu, spec_cond, secchi, ammonia

En match_pairs.py se une como contexto (merge_asof <= t0). Todas medidas en t0 -> sin fuga.
"""
from __future__ import annotations
import os, ssl, io, urllib.request, time
import pandas as pd
import config as C

OUT_DIR = os.path.join(C.DIR_OUT, "targets")
OUT = os.path.join(OUT_DIR, "waterquality_daily.csv")

BODIES = {
    "okeechobee": ("-81.10,26.70,-80.60,27.20", "freshwater"),
    "tampa_bay":  ("-82.75,27.50,-82.40,27.95", "marine"),
    "yojoa":      ("-88.02,14.78,-87.90,14.95", "freshwater"),
    "cajon":      ("-87.80,14.70,-87.58,14.95", "freshwater"),
    "fonseca":    ("-87.85,12.90,-87.35,13.45", "marine"),
}
# characteristicName WQP -> nombre corto de columna
CHARS = {
    "Temperature, water": "water_temp",
    "Dissolved oxygen (DO)": "do_mgl",
    "pH": "ph",
    "Turbidity": "turbidity_insitu",
    "Specific conductance": "spec_cond",
    "Depth, Secchi disk depth": "secchi",
    "Ammonia": "ammonia",
}


def _query(bbox, char):
    url = ("https://www.waterqualitydata.us/data/Result/search?"
           f"bBox={bbox}&characteristicName={urllib.request.quote(char)}"
           "&startDateLo=01-01-2023&startDateHi=12-31-2026&mimeType=csv&zip=no"
           "&dataProfile=narrowResult")
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=180, context=ctx).read()
    try:
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception:
        return pd.DataFrame()


def build():
    body_frames = []
    for name, (bbox, group) in BODIES.items():
        cols = {}
        for char, short in CHARS.items():
            try:
                df = _query(bbox, char)
            except Exception:
                continue
            if not len(df) or "ResultMeasureValue" not in df.columns:
                continue
            v = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")
            fe = pd.to_datetime(df["ActivityStartDate"], errors="coerce").dt.normalize()
            s = pd.DataFrame({"fecha": fe, short: v}).dropna()
            if len(s):
                cols[short] = s.groupby("fecha")[short].median()
            time.sleep(0.3)
        if not cols:
            print(f"  {name:12s}: sin datos in-situ"); continue
        wide = pd.concat(cols.values(), axis=1).reset_index()
        wide["water_body"] = name; wide["group"] = group
        body_frames.append(wide)
        print(f"  {name:12s}: {len(wide):>4} dias | variables: {list(cols.keys())}")

    if not body_frames:
        print("Sin datos."); return
    out = pd.concat(body_frames, ignore_index=True).sort_values(["water_body", "fecha"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nCalidad de agua in-situ -> {OUT} ({len(out)} dias-cuerpo)")


if __name__ == "__main__":
    build()
