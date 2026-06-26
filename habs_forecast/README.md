# habs_forecast — Pronóstico temprano de HABs (0–7 días)

Pipeline de la tesis *"Sistema Inteligente de Predicción Temprana de Floraciones Algales
Nocivas a 0–7 días mediante Análisis Multiespectral y Redes Neuronales"*.

Reconstruido desde cero para **pronóstico causal real** (X disponible ≤ t₀ → estado en t₀+h),
corrigiendo el sistema anterior, que en realidad era **detección** con métricas infladas por
fuga de información (target derivado de las mismas bandas, validación con shuffle, AUC≈1.0).

---

## Decisiones de diseño (Fase 1, cerradas)

| Tema | Decisión | Justificación |
|---|---|---|
| Problema | Pronóstico causal X(≤t₀)→y(t₀+h) | el título exige anticipación, no detección |
| Horizontes | Modelos **separados** h=0,1,3,5,7 | N pequeño; validación más simple |
| Ventana | **Solo 2023–2026** | cobertura Sentinel-2 |
| Grupos | Lago (cianobacterias) vs Costa (dinoflagelados) | biogeoquímica distinta, no óptica |
| Predictores | S2 espectral + autorregresivo chl + ERA5 | ver importancia de features |
| Target | **VIIRS chl diario** (sensor independiente de S2) | rompe la circularidad |
| Validación | Walk-forward (operativo) + **LOWBO** lago↔lago, costa↔costa | generalización honesta |
| Métricas | Skill vs persistencia, RMSE(log-chl), Recall, PR-AUC | eventos raros |

---

## Flujo del pipeline

```
imagenes/*.tif (888 S2) ──► build_scene_state.py ──► scene_state.csv      (PREDICTOR X en t0)
era5_temp_nc/*.nc ─────────► build_era5_daily.py ──► era5_daily.csv        (drivers meteo)
ERDDAP VIIRS ──────────────► fetch_satellite_chl.py ► satellite_chl_daily.csv (TARGET y en t0+h)
WQP ───────► fetch_wqp_stations.py + ingest_insitu.py ► insitu_chl.csv     (VALIDACIÓN in-situ)
                                   │
                                   ▼
                         match_pairs.py  (une X(t0) con y(t0+h) + autorregresivo + ERA5, sin fuga)
                                   ▼
                         pairs_forecast.csv
                                   ▼
                    train.py  (XGBoost por horizonte; walk-forward + LOWBO)
                    analyze_importance.py  (qué impulsa el pronóstico)
```

### Orden de ejecución
```bash
python fetch_satellite_chl.py     # target VIIRS diario (ERDDAP, sin credenciales)
python fetch_wqp_stations.py      # coords de estaciones WQP (1 vez)
python ingest_insitu.py           # set de validación in-situ 2023-2026
python build_scene_state.py       # predictores desde rasters S2 (lento, ~40GB)
python build_era5_daily.py        # drivers ERA5 diarios
python match_pairs.py             # pares causales X(t0)->y(t0+h)
python train.py                   # entrenamiento + validación
python analyze_importance.py      # diagnóstico de features
```

Salidas en `artifacts/` (state_series, targets, pairs, models, reports).

---

## Resultados honestos (no inflados)

**Skill de pronóstico vs persistencia (walk-forward, Okeechobee):**

| Horizonte | RMSE_log modelo | RMSE_log persistencia | Skill |
|---|---|---|---|
| +1 d | 0.459 | 0.505 | +9% |
| +3 d | 0.576 | 0.599 | +4% |
| +5 d | 0.555 | 0.641 | +13% |
| +7 d | 0.557 | 0.719 | **+23%** |

La ventaja sobre la persistencia **crece con el horizonte** → justifica el sistema de IA.

**Importancia de features por familia (sin fuga óptica):**

| Horizonte | ERA5 | Autorregresivo | Espectral S2 |
|---|---|---|---|
| +1 d | 1.05 | 0.75 | 0.20 |
| +3 d | 0.87 | 0.94 | 0.19 |
| +7 d | 0.84 | 0.78 | 0.38 |

- Lagos: dominan clorofila reciente + radiación/lluvia (cianobacterias, limitación nutrientes/luz).
- Costa: dominan presión y viento (dinoflagelados, forzamiento físico).
- Espectral S2 pesa poco → el modelo **no** lee el bloom del target en la imagen (no hay fuga).

---

## Sistema híbrido (regresión + clasificación)

- **Cabeza de regresión**: predice log(chl) → intensidad. Resultado estable y citable (skill vs
  persistencia, arriba). Es el producto principal.
- **Cabeza de clasificación directa**: XGBoost con `scale_pos_weight` sobre el evento de alerta
  (umbral **relativo por ecosistema**, P85 de la climatología local). Mejora el Recall en costa
  donde "regresión→umbral" fallaba (p.ej. Tampa Bay LOWBO: 0.00 → 0.33–1.00). Producto
  complementario, de **menor confianza** (alta varianza por folds pequeños).

## Hallazgos / advertencias para la defensa

1. **h=0 es detección, no pronóstico** (persistencia perfecta por construcción). Reportar aparte.
2. **h≥1 dentro del cuerpo = skill real** de regresión sobre persistencia (producto principal).
3. **Transferencia inter-ecosistema (LOWBO) es débil** — límite genuino de generalización.
4. **La alerta de clasificación es inestable a horizonte largo** (folds de test pequeños, eventos
   raros). Reportar con incertidumbre; mejora pendiente: métricas agregadas/bootstrap, no fold único.

## Limitaciones

- Target VIIRS 750 m es grueso para lagos pequeños (Yojoa, Cajón) → densidad/ruido. Mejora
  futura: **Sentinel-3 OLCI 300 m** (requiere credenciales Copernicus Data Space).
- ERA5: serie ~574 días/cuerpo (no estrictamente diaria); Fonseca (13.2 N) al borde sur de la
  grilla → punto más cercano (aproximación).
- In-situ 2023–2026 escaso (389 pts) → rol de validación, no de entrenamiento.
- Nubosidad limita la cadencia de S2 (predictor).
