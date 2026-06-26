# ESTADO DEL PROYECTO — Sistema de predicción temprana de HABs

> Documento de continuación. Si retomas el trabajo (o una nueva sesión de IA no carga
> contexto), **lee este archivo primero**: resume qué se hizo, en qué estado quedó, y qué sigue.
> Última actualización: fin de jornada 2026-06-24.

---

## CÓMO RETOMAR
Si la IA no carga contexto, dile: **"Lee `habs_forecast/ESTADO_PROYECTO.md` y seguimos"**.
Todo el trabajo (código, modelos, resultados) está en disco, en la carpeta `habs_forecast/`.

---

## QUÉ ES EL PROYECTO
Tesis: *"Sistema inteligente de predicción temprana de floración de algas nocivas (HABs) a
0–7 días mediante análisis multiespectral aplicando redes neuronales."*
Cuerpos: lagos/embalse (Okeechobee, Yojoa, Cajón) y costa (Tampa Bay, Golfo de Fonseca).
Ventana de datos: **2023–2026** (solo).

## DIAGNÓSTICO INICIAL (corregido)
El sistema viejo (notebooks `Modelo_tesis.ipynb`) era **detección con fuga de datos**
(AUC=1.0 falso: target derivado de las mismas bandas, validación con shuffle). Se rediseñó
desde cero a **pronóstico causal real** X(≤t0) → clorofila(t0+h).
**Los notebooks viejos NO se deben usar** (reflejan el sistema defectuoso).

## QUÉ LLEVAMOS (funciona, en habs_forecast/)
- **Pipeline sin fuga:** rasters Sentinel-2 → estado por escena → target satelital
  (combinado: OLCI en costa, VIIRS corregido a in-situ en Okeechobee, VIIRS en Honduras)
  → pares causales con variables espectrales + clorofila reciente + ERA5 + nutrientes
  (fósforo) + calidad de agua in-situ (temp, OD, pH, turbidez, conductividad, Secchi, amonio).
- **Modelo:** features seleccionadas por horizonte; **XGBoost** (intensidad) + **red neuronal**
  (HABNet); el **ensamble Red+XGBoost mejora la ALERTA**. La red cumple el título.
- **Sistema desplegable:**
  - `predict.py` → pronóstico 0–7d (clorofila esperada + probabilidad de alerta).
  - `calibrate_alert.py` → alerta calibrada (recall 0.67 lagos / 1.00 costa).
  - `make_maps.py` → mapas espaciales de clorofila (buenos en lagos).
  - `train_final.py` → 8 modelos de producción guardados en `artifacts/models/`.

## RESULTADOS HONESTOS
- **Lagos:** skill de pronóstico **significativo** a 1–7 días (skill +0.16 a +0.24 vs persistencia).
- **Costa:** la ALERTA funciona; la **intensidad NO es pronosticable** (límite físico, agua de
  baja varianza dominada por persistencia).
- **Honduras (Yojoa, Cajón):** **exploratorio** — sin datos in-situ 2023–2026 para validar.
- OLCI se descartó como target para lagos (validado contra in-situ: correlación 0 en lago somero).

## VALIDACIÓN DEFINITIVA (hecho 2026-06-25)
- **#4 — Validación anidada + test final intacto** (`evaluate_nested.py`): se reserva el último
  ~25% del tiempo por (grupo,horizonte) como TEST INTACTO (con embargo de 8 d); la selección de
  features se hace SOLO en DEV; el TEST se evalúa una vez. **RESULTADO: el skill de lagos
  SOBREVIVE** sobre datos nunca tocados, significativo en TODOS los horizontes (IC95% excluye 0).
  No era artefacto de selección de features.
- **#5 — Sensibilidad ERA5** (`era5_sensitivity.py`): ablación (con vs sin ERA5) + estrés de ruido
  (proxy reanálisis→pronóstico). **RESULTADO: el sistema NO es frágil a ERA5.** En lagos ERA5
  aporta casi nada (ablación −0.01 a +0.04) y la curva de ruido es plana (skill estable aun con
  ruido al 100% de la variabilidad de cada driver) → **se puede operar con ERA5 de pronóstico sin
  perder skill** (el backbone es la clorofila autorregresiva + espectral).
  Reporte: `artifacts/reports/era5_sensitivity.json`.

## DENSIFICACIÓN SENTINEL-2 + RE-VALIDACIÓN (hecho 2026-06-25)
Diagnóstico: el cuello de botella de los cuerpos débiles era el **nº de escenas S2** (predictor),
no el target. Con `fetch_s2_scenes.py` (Google Earth Engine, `COPERNICUS/S2_SR_HARMONIZED`, máscara
de nubes SCL, 2023–2026, descarga local incremental) se bajaron **+487 escenas**:
Yojoa 78→196, Cajón 79→179, Fonseca 139→272, Tampa 83→219 (Okeechobee intacto). Pares **2890→4327**.
Re-corrida la validación anidada, el antes/después (skill de regresión, IC95%):

| Horiz | LAGOS antes | LAGOS después | COSTA antes | COSTA después |
|------|-------------|---------------|-------------|---------------|
| +1d | +0.24 [.16,.33] | **+0.25 [.16,.34]** | +0.04 [ns] | +0.21 [−.00,.44] |
| +3d | +0.15 [.03,.24] | **+0.16 [.05,.26]** | +0.15 [.02,.37] | **+0.21 [.07,.36]** |
| +5d | +0.23 [.15,.31] | **+0.18 [.09,.25]** | +0.18 [ns] | **+0.23 [.02,.43]** |
| +7d | +0.20 [.12,.29] | **+0.20 [.13,.27]** | +0.34 [ns] | +0.19 [ns] |

- **Lagos:** siguen significativos en TODOS los horizontes; el test ahora incluye **Okeechobee + Yojoa**
  (antes solo Okeechobee) → conjunto defendible más amplio.
- **Costa (cambio de conclusión):** pasa de "intensidad NO pronosticable" a **skill significativo a
  +3d y +5d** (eventos 5–11 → 12–16). ⇒ la costa estaba **limitada por datos del predictor**, NO por
  un límite físico. +1d borderline, +7d aún incierto.
- Niveles de confianza: **ALTA** = Okeechobee (validado in-situ) y costa (target satelital validado);
  **validación interna robusta sin verdad de campo** = Yojoa (target VIIRS, sin in-situ 2023–2026);
  **aún exploratorio** = Cajón (174 pares, no alcanza el test anidado).

## CIERRE FINO 2026-06-25 (Cajón, validación Yojoa, honestidad, reporte único)
- **Cajón (tope nubes 85%)**: +54 escenas (179→233), pares 174→221. Mejoró pero **sigue sin entrar
  al test anidado**: tras el split 75/25 su DEV cae a ~32–38 pares (< mínimo 40). Decisión: no bajar
  el umbral solo para colarlo. Queda exploratorio (borderline). `fetch_s2_scenes.py` ahora acepta
  `S2_MAXCLOUD` por env y cuerpos por argumento.
- **In-situ Honduras 2023–2026**: NO existe público (confirmado). El único dataset de Yojoa
  (Fadum/Ross, CSU; Zenodo 8139922) es Secchi y **termina en 2022**. Cajón sin programa de monitoreo.
- **Validación target Yojoa** (`validate_yojoa_insitu.py`, FUERA del modelo): VIIRS-chl vs Secchi
  in-situ 2018–2022, 85 matchups → **correlación NEGATIVA y significativa** (Pearson −0.31 p=0.004,
  Spearman −0.28 p=0.009). El VIIRS sigue la transparencia real ⇒ **target de Yojoa CREÍBLE** (sube
  de "exploratorio" a "validado fuera de ventana"). NO entra al entrenamiento (regla 2023–2026 intacta).
- **Chequeo de honestidad**: 0 features contaminadas, 0 pares con fuga temporal, NDVI no-predictor,
  h=0 separado; `predict.py`/`make_maps.py` solo usan datos ≤t0. Limpio.
- **Reporte único de defensa**: `REPORTE_DEFENSA.md` (generado por `build_final_report.py`) consolida
  inventario, validación anidada, sensibilidad ERA5, alerta calibrada, validación Yojoa, confianza y
  honestidad. **Este es el documento de números definitivos.**

## MEJORA DE PROTOCOLO + EXPERIMENTO DE FEATURES (2026-06-25)
- **ADOPTADO — selección agrupada + parsimonia** en `evaluate_nested.py`: la selección de features
  y el entrenamiento ahora usan el **DEV agrupado de todos los cuerpos del grupo** (una decisión por
  grupo-horizonte, alineado con producción que entrena por grupo), y entre combinaciones casi
  empatadas en DEV se elige la de menos familias (parsimonia, regla anti-sobreajuste de selección).
  Esto **redujo la varianza de selección y subió los números honestos**. Números oficiales (test
  intacto, IC95%, `*`=significativo):

  | Horiz | LAGOS | COSTA |
  |------|-------|-------|
  | +1d | +0.25 [.16,.34]* | +0.10 [−.14,.31] |
  | +3d | +0.17 [.08,.24]* | +0.22 [.07,.39]* |
  | +5d | +0.19 [.10,.28]* | +0.28 [.08,.49]* |
  | +7d | **+0.32 [.23,.38]*** | +0.28 [−.05,.49] |

  Lagos significativos en TODOS los horizontes (antes +7d 0.20 → ahora **0.32**); costa significativa
  a +3d y +5d (antes +5d 0.23 → **0.28**). Alerta lagos PR-AUC +1d 0.38.
- **NO ADOPTADO — features de dinámica temporal + estacionalidad** (`DYNAMICS`, `SEASONAL`, causales,
  construidas en `match_pairs.py`): probadas con `HABS_NEWFEATS=1` en validación anidada CONTROLADA
  (con vs sin, mismo protocolo y datos). Veredicto: **lavado** (mejoran +5d lagos pero empeoran +3d
  y +7d) → no aportan skill robusto, se descartan. Las "ganancias" iniciales eran artefacto de
  comparar contra otro protocolo. **Resultado negativo legítimo** (demuestra rigor); columnas se
  conservan para re-test futuro pero NO entran al modelo.

## INTERVALOS DE INCERTIDUMBRE + INTERPRETACIÓN BIOLÓGICA (2026-06-25)
- **Intervalos de incertidumbre (CQR)** (`evaluate_intervals.py`): cada pronóstico de intensidad
  lleva una banda **P10–P90** (regresión cuantil XGBoost + conformalización split). Validada en el
  test intacto: la banda cruda quedaba sobreconfiada (cobertura 0.45–0.61); **CQR la calibra a
  0.71–0.82** (objetivo 0.80) en los 8 cuerpos-horizonte. Integrada en `train_final.py` (guarda
  P10/P90 + offset conformal) y `predict.py` (muestra la banda en µg/L). Anchos coherentes: lagos
  ~29–49 µg/L (eutróficos, alta incertidumbre), costa ~3 µg/L (baja varianza).
- **Interpretación biológica (revisión asesora)**: el sistema predice **clorofila-a = proxy de
  biomasa**, NO confirma floración NOCIVA (toxicidad). Salidas reetiquetadas en `predict.py`/
  `make_maps.py` ("riesgo / biomasa elevada" en vez de "floración"); S2 no distingue cianobacterias;
  amonio = una forma del N (limitación declarada). En `REPORTE_DEFENSA.md` (sección interpretación).
  Redacción del título/abstract la maneja el usuario en su documento (no en código).
- **Landsat 8/9**: NO construido. Limitación: OLI carece de banda red-edge (~705 nm) que alimenta
  NDCI/CI_red/FAI → aportaría pares pero con señal espectral degradada. Queda como experimento
  opcional, se prefirió la mejora segura (intervalos).

## QUÉ SIGUE (pendiente)
Modelos en estado de defensa. Quedan, cuando el usuario lo indique (EN PAUSA): figuras, notebook
limpio y redacción de tesis.

## EN PAUSA (no hacer hasta que el usuario lo pida)
Figuras, notebook limpio que reemplace los viejos, redacción de tesis.

---

## MAPA DE ARCHIVOS CLAVE (habs_forecast/)
- `config.py` — configuración central (cuerpos, horizontes, umbrales, índices).
- `build_scene_state.py` — lee rasters S2 → predictores por escena.
- `fetch_satellite_chl.py` / `fetch_olci_chl.py` — target satelital (VIIRS / OLCI).
- `fetch_s2_scenes.py` — **descarga incremental de más escenas Sentinel-2 (GEE)** para densificar
  cuerpos débiles (requiere auth GEE + EE_PROJECT; `S2_MAXCLOUD` y cuerpos por argumento). Local.
- `validate_yojoa_insitu.py` — valida el target VIIRS de Yojoa vs Secchi in-situ 2018–2022 (fuera
  del modelo, no entrena). Salida: `artifacts/validation_yojoa/`.
- `build_final_report.py` — consolida todos los números definitivos en `REPORTE_DEFENSA.md`.
- `ingest_insitu.py` / `ingest_nutrients.py` / `ingest_waterquality.py` — datos in-situ WQP.
- `build_era5_daily.py` — drivers ERA5.
- `bias_correct_target.py` — corrección de escala del target a in-situ.
- `build_combined_target.py` — target óptimo por cuerpo.
- `match_pairs.py` — empareja predictores(t0) con target(t0+h), sin fuga.
- `select_features_per_horizon.py` — features por horizonte.
- `train.py` / `train_nn.py` / `train_stack.py` — XGBoost / red / ensamble.
- `evaluate_robust.py` — evaluación OOS de ventana expansiva con bootstrap IC95%.
- `evaluate_nested.py` — **validación anidada + test final intacto** (números definitivos #4).
- `evaluate_intervals.py` — valida cobertura de los **intervalos de incertidumbre P10–P90 (CQR)**.
- `era5_sensitivity.py` — **sensibilidad ERA5 reanálisis vs pronóstico** (ablación + ruido #5).
- `train_final.py` / `predict.py` / `calibrate_alert.py` / `make_maps.py` — sistema desplegable.
- `variables_modelo.txt` — lista de variables del modelo.
- `README.md` — metodología y resultados.
- `artifacts/` — datos procesados, modelos y reportes generados.
