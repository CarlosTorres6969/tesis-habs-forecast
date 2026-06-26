# habs_forecast — Predicción temprana de riesgo de biomasa algal (0–7 días)

![integridad](https://github.com/CarlosTorres6969/tesis-habs-forecast/actions/workflows/ci.yml/badge.svg)

Pipeline de la tesis sobre **pronóstico temprano de floraciones algales**. El sistema es una
**herramienta de alerta temprana de condiciones de riesgo** (clorofila-a / biomasa algal elevada),
**no** un detector certero de toxicidad: confirmar nocividad requiere verificación de campo
(cianobacterias, toxinas). Sentinel-2 estima biomasa, no identifica cianobacterias.

Reconstruido desde cero para **pronóstico causal** X(≤t₀) → clorofila-a(t₀+h), corrigiendo el
sistema anterior (en `../legacy/`), que era **detección con fuga** (target circular + validación con
shuffle → AUC≈1.0 falso).

---

## Qué hace

- **Intensidad**: pronostica clorofila-a (µg/L) a +1/+3/+5/+7 días, con **banda de incertidumbre
  P10–P90 calibrada** (regresión cuantil conformalizada, CQR; cobertura ~0.80 verificada).
- **Alerta de riesgo**: probabilidad calibrada de biomasa anómala (P85 de la climatología local),
  ensamble **XGBoost + red neuronal**, con umbral operativo F2 (prioriza recall).
- **Mapas** espaciales de clorofila-a prevista por cuerpo.

## Datos (solo 2023–2026)

| Fuente | Rol |
|---|---|
| Sentinel-2 (multiespectral) + **Landsat 8/9** | predictor espectral en t₀ (Landsat densifica Cajón) |
| VIIRS / OLCI (clorofila satelital diaria) | **target** en t₀+h (sensor independiente → sin circularidad) |
| ERA5-Land | drivers meteorológicos |
| WQP in-situ (fósforo, calidad de agua) | contexto + validación |

Cuerpos: lagos **Okeechobee, Yojoa, Cajón**; costa **Tampa Bay, Golfo de Fonseca**.

---

## Reproducir

```bash
pip install -r ../requirements.txt
python run_pipeline.py        # build_scene_state → harmonize → match_pairs → train → eval → reporte
python check_integrity.py     # 11 aserciones de honestidad (sin fuga / causal / consistente)
```

La descarga de datos crudos (requiere credenciales) va aparte: `fetch_satellite_chl.py` (VIIRS, sin
credenciales), `fetch_s2_scenes.py` / `fetch_landsat_scenes.py` (GEE), `fetch_olci_chl.py`
(Copernicus), `ingest_*.py` (WQP), `build_era5_daily.py`. Ver `run_pipeline.py`.

## Validación honesta

- **`evaluate_nested.py`** — validación anidada con **test temporal intacto** (último ~25% nunca
  tocado; selección de features solo en DEV). Es el número defendible.
- **`evaluate_intervals.py`** — cobertura de los intervalos de incertidumbre (CQR).
- **`era5_sensitivity.py`** — robustez reanálisis vs pronóstico (ablación + ruido).
- **`check_integrity.py`** — test ejecutable: sin fuga, causalidad (target>t₀), features limpias,
  modelos con intervalos. Corre en CI (modo estático) y en local (11/11).

Resultados definitivos auto-generados en **`REPORTE_DEFENSA.md`** (`build_final_report.py`).

### Lectura de resultados
- **Lagos** (Okeechobee, Yojoa, Cajón): skill de regresión significativo vs persistencia en la
  mayoría de horizontes; la ventaja **crece a horizonte largo** (estacionalidad/inercia).
- **Costa**: la **alerta** funciona (PR-AUC); la intensidad es significativa a horizontes medios.
- **h=0** es detección (persistencia perfecta) → se reporta aparte, no es pronóstico.

## Niveles de confianza
- **ALTA**: Okeechobee (target VIIRS validado con in-situ), Tampa/Fonseca (target satelital validado).
- **Validado fuera de ventana**: Yojoa (VIIRS sigue el Secchi de campo 2018–2022).
- **Exploratorio**: Cajón (evaluable gracias a Landsat; sin verdad de campo).

## Limitaciones declaradas
- Clorofila-a ≠ floración nociva (proxy de biomasa; toxicidad requiere campo).
- Sentinel-2 no distingue cianobacterias (sin banda de ficocianina ~620 nm).
- Nitrógeno representado por **amonio** (una sola forma del N; sin nitrato/nitrito/N total).
- In-situ 2023–2026 escaso (Honduras sin verdad de campo en ventana).
- Landsat 8/9 sin red-edge → NDCI/CI_red/FAI quedan NaN en sus escenas (offset cross-sensor
  harmonizado; el pronóstico es robusto a él, ver `ESTADO_PROYECTO.md`).

---

Estado detallado y bitácora: **`ESTADO_PROYECTO.md`**. Anexo reproducible: **`Modelo_HABs_limpio.ipynb`**.
