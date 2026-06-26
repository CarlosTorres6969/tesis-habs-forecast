"""
validate_yojoa_insitu.py — VALIDACION del target satelital de Yojoa contra in-situ (FUERA del
modelo). NO entra al entrenamiento; es un cheque de credibilidad de la MEDICION satelital.

Contexto: dentro de la ventana del proyecto (2023-2026) NO existe in-situ publico para Yojoa
(el monitoreo de campo de Fadum/Ross, CSU, termina en 2022; Zenodo 8139922). Pero ese in-situ
2018-2022 (Secchi, transparencia) sirve como VARA DE MEDIR independiente: si el VIIRS que usamos
como target sigue la transparencia real del lago, gana credibilidad para 2023-2026.

Metodo:
  1. In-situ: Secchi 2018-2022 (Zenodo, 808 medidas 1979-2022), media diaria sobre los puntos.
  2. Satelital: VIIRS chlor_a diario para el bbox de Yojoa 2018-2022 (mismo ERDDAP que el target).
  3. Emparejar por fecha cercana (<=4 d) y correlacionar.
Lectura: clorofila ALTA -> agua turbia -> Secchi BAJO, asi que se ESPERA correlacion NEGATIVA.
Es indirecta (Secchi != clorofila), pero es la unica verdad de campo disponible para Yojoa.

Salida: artifacts/validation_yojoa/yojoa_target_validation.json (+ csv de matchups)
"""
from __future__ import annotations
import os, ssl, io, urllib.request, time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import config as C

VDIR = os.path.join(C.DIR_OUT, "validation_yojoa")
SECCHI = os.path.join(VDIR, "secchi_yojoa.csv")
OUT_JSON = os.path.join(VDIR, "yojoa_target_validation.json")
OUT_CSV = os.path.join(VDIR, "yojoa_matchups.csv")

# bbox Yojoa (igual que fetch_satellite_chl) y ventana de validacion (fuera del modelo)
LA_LO, LA_HI, LO_LO, LO_HI = 14.78, 14.95, -88.02, -87.90
YEARS = [2018, 2019, 2020, 2021, 2022]
ERDDAP = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaDaily.csv"
          "?chlor_a%5B({t0}):1:({t1})%5D%5B(0.0)%5D%5B({la_hi}):1:({la_lo})%5D"
          "%5B({lo_lo}):1:({lo_hi})%5D")
MATCH_TOL_DAYS = 4


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


def fetch_viirs():
    chunks = []
    for yr in YEARS:
        url = ERDDAP.format(t0=f"{yr}-01-01", t1=f"{yr}-12-31",
                            la_hi=LA_HI, la_lo=LA_LO, lo_lo=LO_LO, lo_hi=LO_HI)
        try:
            raw = _get(url)
        except Exception as e:
            print(f"  VIIRS {yr}: {type(e).__name__}"); continue
        d = pd.read_csv(io.BytesIO(raw), skiprows=[1])
        chunks.append(d); time.sleep(0.5)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df["chlor_a"] = pd.to_numeric(df["chlor_a"], errors="coerce")
    df = df.dropna(subset=["chlor_a"])
    f = pd.to_datetime(df["time"])
    if getattr(f.dt, "tz", None) is not None:
        f = f.dt.tz_localize(None)
    df["fecha"] = f.dt.normalize()
    return (df.groupby("fecha").agg(chl_sat=("chlor_a", "median"),
                                    n_px=("chlor_a", "size")).reset_index())


def load_secchi():
    s = pd.read_csv(SECCHI)
    s["secchi"] = pd.to_numeric(s["secchi"].astype(str).str.replace("..", ".", regex=False),
                                errors="coerce")
    s["fecha"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["fecha", "secchi"])
    s = s[(s["fecha"].dt.year >= YEARS[0]) & (s["fecha"].dt.year <= YEARS[-1])]
    # media diaria sobre los puntos de muestreo
    return s.groupby("fecha").agg(secchi=("secchi", "mean"), n_pts=("secchi", "size")).reset_index()


def main():
    os.makedirs(VDIR, exist_ok=True)
    print("Trayendo VIIRS chlor_a Yojoa 2018-2022 (solo validacion, NO entra al modelo)...")
    sat = fetch_viirs()
    ins = load_secchi()
    print(f"  VIIRS: {len(sat)} dias con dato | in-situ Secchi: {len(ins)} dias (2018-2022)")
    if sat.empty or ins.empty:
        print("Sin datos suficientes para validar."); return

    # emparejar por fecha mas cercana dentro de la tolerancia
    sat = sat.sort_values("fecha"); ins = ins.sort_values("fecha")
    m = pd.merge_asof(ins, sat, on="fecha", direction="nearest",
                      tolerance=pd.Timedelta(days=MATCH_TOL_DAYS)).dropna(subset=["chl_sat"])
    m.to_csv(OUT_CSV, index=False)
    n = len(m)
    print(f"  matchups (<= {MATCH_TOL_DAYS} d): {n}")
    if n < 10:
        print("Pocos matchups para una correlacion fiable; reportar como exploratorio.");
    res = {"n_matchups": int(n), "tol_days": MATCH_TOL_DAYS,
           "viirs_dias": int(len(sat)), "insitu_dias": int(len(ins))}
    if n >= 5:
        chl = m["chl_sat"].values; sec = m["secchi"].values
        pr, pp = pearsonr(chl, sec)
        lr, lp = pearsonr(np.log1p(chl), sec)
        sr, sp = spearmanr(chl, sec)
        res.update({
            "pearson_chl_secchi": [float(pr), float(pp)],
            "pearson_logchl_secchi": [float(lr), float(lp)],
            "spearman_chl_secchi": [float(sr), float(sp)],
            "chl_sat_mediana": float(np.median(chl)), "secchi_mediana": float(np.median(sec)),
        })
        print(f"\n  Correlacion VIIRS-chl  vs  Secchi in-situ (n={n}):")
        print(f"    Pearson (chl, secchi)     r={pr:+.3f}  p={pp:.3f}")
        print(f"    Pearson (log-chl, secchi) r={lr:+.3f}  p={lp:.3f}")
        print(f"    Spearman (rango)          r={sr:+.3f}  p={sp:.3f}")
        veredicto = ("NEGATIVA y significativa -> el VIIRS SIGUE la transparencia real del lago: "
                     "target de Yojoa CREIBLE" if (sr < 0 and sp < 0.05) else
                     "no concluyente -> Yojoa permanece exploratorio (VIIRS 750 m es marginal "
                     "para un lago pequeno)")
        res["veredicto"] = veredicto
        print(f"\n  VEREDICTO: {veredicto}")
    import json
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print(f"\nReporte -> {OUT_JSON}")


if __name__ == "__main__":
    main()
