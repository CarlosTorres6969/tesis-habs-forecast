"""
ingest_nutrients.py — TARGET/CONTEXTO de nutrientes in-situ (Fosforo) desde WQP (publico).

Hallazgo: en 2023-2026 el Fosforo SI es denso para Okeechobee (4389 medidas / 483 fechas)
y Tampa (3841 / 444). Nitrogeno escaso. Honduras sin cobertura WQP.

El fosforo cambia LENTO -> contexto de susceptibilidad (no driver diario). Se ingiere por
bbox de cada cuerpo (ya filtrado espacialmente) y se agrega a mediana diaria.
Salida: artifacts/targets/nutrients_daily.csv (water_body, fecha, tp_mgl, group)

En match_pairs.py se une como feature de contexto 'tp_context' (carry-forward <= t0, lento).
"""
from __future__ import annotations
import os, ssl, io, urllib.request, time
import pandas as pd
import config as C

OUT_DIR = os.path.join(C.DIR_OUT, "targets")
OUT = os.path.join(OUT_DIR, "nutrients_daily.csv")

# bbox (oeste,sur,este,norte) + grupo  — mismos cuerpos del target
BODIES = {
    "okeechobee": ("-81.10,26.70,-80.60,27.20", "freshwater"),
    "tampa_bay":  ("-82.75,27.50,-82.40,27.95", "marine"),
    "yojoa":      ("-88.02,14.78,-87.90,14.95", "freshwater"),
    "cajon":      ("-87.80,14.70,-87.58,14.95", "freshwater"),
    "fonseca":    ("-87.85,12.90,-87.35,13.45", "marine"),
}
CHAR = "Phosphorus"


def _query(bbox):
    url = ("https://www.waterqualitydata.us/data/Result/search?"
           f"bBox={bbox}&characteristicName={urllib.request.quote(CHAR)}"
           "&startDateLo=01-01-2023&startDateHi=12-31-2026&mimeType=csv&zip=no"
           "&dataProfile=narrowResult")
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=180, context=ctx).read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def build():
    frames = []
    for name, (bbox, group) in BODIES.items():
        try:
            df = _query(bbox)
        except Exception as e:
            print(f"  {name:12s}: FALLO {type(e).__name__}"); continue
        if not len(df) or "ResultMeasureValue" not in df.columns:
            print(f"  {name:12s}: sin datos"); continue
        df["val"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")
        df["fecha"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce").dt.normalize()
        unit = df.get("ResultMeasure/MeasureUnitCode", "").astype(str).str.lower()
        # normalizar a mg/L (ug/L -> /1000); descartar valores no fisicos
        f = pd.Series(1.0, index=df.index); f[unit.str.contains("ug/l", na=False)] = 0.001
        df["tp_mgl"] = df["val"] * f
        df = df.dropna(subset=["fecha", "tp_mgl"])
        df = df[(df["tp_mgl"] > 0) & (df["tp_mgl"] < 50)]
        daily = df.groupby("fecha", as_index=False)["tp_mgl"].median()
        daily["water_body"] = name; daily["group"] = group
        frames.append(daily)
        print(f"  {name:12s}: {len(daily):>4} dias | TP mediana={daily['tp_mgl'].median():.3f} mg/L")
        time.sleep(0.5)

    if not frames:
        print("Sin nutrientes."); return
    out = pd.concat(frames, ignore_index=True).sort_values(["water_body", "fecha"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nNutrientes -> {OUT} ({len(out)} dias-cuerpo)")


if __name__ == "__main__":
    build()
