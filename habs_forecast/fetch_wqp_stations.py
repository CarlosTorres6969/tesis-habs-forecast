"""
fetch_wqp_stations.py — Descarga coordenadas de estaciones WQP (Florida) para geolocalizar
el target in-situ. El archivo de resultados solo trae MonitoringLocationIdentifier; las
coordenadas viven en el servicio Station. Descarga una vez y cachea a datasets/wqp_stations.csv.
"""
from __future__ import annotations
import os, ssl, urllib.request, io
import pandas as pd
import config as C

OUT = os.path.join(C.DIR_DATASETS, "wqp_stations.csv")
RESULT_FILE = os.path.join(C.DIR_DATASETS, "florida", "florida_chlorophyll_wqp_2025_2026.csv")
BASE = "https://www.waterqualitydata.us/data/Station/search?organization={org}&mimeType=csv&zip=no"
COLS = ["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure",
        "MonitoringLocationTypeName"]


def _orgs():
    df = pd.read_csv(RESULT_FILE, low_memory=False, usecols=["OrganizationIdentifier"])
    return df["OrganizationIdentifier"].value_counts().index.tolist()


def fetch():
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    orgs = _orgs()
    print(f"Descargando estaciones de {len(orgs)} organizaciones...")
    frames = []
    for i, org in enumerate(orgs, 1):
        url = BASE.format(org=urllib.request.quote(str(org)))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=120, context=ctx).read()
            d = pd.read_csv(io.BytesIO(raw), low_memory=False)
            have = [c for c in COLS if c in d.columns]
            if "LatitudeMeasure" in have:
                frames.append(d[have])
        except Exception as e:
            print(f"  [{i:>2}/{len(orgs)}] {org}: FALLO {type(e).__name__}")
            continue
        if i % 10 == 0:
            print(f"  [{i:>2}/{len(orgs)}] ...{sum(len(f) for f in frames)} estaciones")
    df = pd.concat(frames, ignore_index=True)
    keep = df.dropna(subset=["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure"])
    keep = keep.drop_duplicates("MonitoringLocationIdentifier")
    keep.to_csv(OUT, index=False)
    print(f"\nEstaciones con coords: {len(keep)} -> {OUT}")
    return keep


if __name__ == "__main__":
    fetch()
