"""
run_pipeline.py — Orquesta el pipeline de modelado en orden (reproducibilidad).

Ejecuta los pasos que parten de los artefactos ya descargados (rasters Sentinel-2 en imagenes/,
serie de target satelital y ERA5 en artifacts/). La INGESTA de datos (descarga) requiere
credenciales y se corre por separado ANTES de esto:
    fetch_satellite_chl.py   (VIIRS, sin credenciales)
    fetch_olci_chl.py        (Sentinel-3 OLCI, requiere cuenta Copernicus)
    fetch_s2_scenes.py       (Sentinel-2, requiere cuenta Google Earth Engine + EE_PROJECT)
    ingest_insitu.py / ingest_nutrients.py / ingest_waterquality.py   (WQP, sin credenciales)
    build_era5_daily.py      (desde los NetCDF de ERA5)

Uso:  python run_pipeline.py
"""
from __future__ import annotations
import subprocess, sys, time, os

STEPS = [
    ("Estado por escena (rasters S2 + Landsat -> predictores)", "build_scene_state.py"),
    ("Harmonizacion cross-sensor Landsat -> escala S2",       "harmonize_landsat.py"),
    ("Correccion de sesgo del target (Okeechobee vs in-situ)", "bias_correct_target.py"),
    ("Target combinado por cuerpo (OLCI costa + VIIRS lagos)", "build_combined_target.py"),
    ("Pares causales (predictor t0 -> target t0+h, sin fuga)", "match_pairs.py"),
    ("Seleccion de features por horizonte",                   "select_features_per_horizon.py"),
    ("Modelos de produccion (XGB + Red + cuantiles CQR)",     "train_final.py"),
    ("Calibracion de alerta (isotonica + umbral F2)",         "calibrate_alert.py"),
    ("Validacion anidada (test temporal intacto)",            "evaluate_nested.py"),
    ("Intervalos de incertidumbre (cobertura CQR)",           "evaluate_intervals.py"),
    ("Sensibilidad ERA5 (reanalisis vs pronostico)",          "era5_sensitivity.py"),
    ("Reporte de defensa consolidado",                        "build_final_report.py"),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print(f"Pipeline de modelado HABs — {len(STEPS)} pasos\n")
    for i, (desc, script) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {desc}  ->  {script}")
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)])
        if r.returncode != 0:
            print(f"  FALLO en {script} (exit {r.returncode}). Deteniendo.")
            sys.exit(r.returncode)
        print(f"  OK ({time.time() - t:.0f}s)\n")
    print("Pipeline completo. Resultados en artifacts/ y REPORTE_DEFENSA.md")


if __name__ == "__main__":
    main()
