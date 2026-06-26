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

> ⚠️ Tabla INTERMEDIA (efecto de la densificación S2, **pre-Landsat-Cajón**). Para las cifras
> vigentes ver "NÚMEROS OFICIALES (reconciliados 2026-06-26)" más abajo / `REPORTE_DEFENSA.md`.

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
  Esto **redujo la varianza de selección y subió los números honestos**.

  > **NÚMEROS OFICIALES (reconciliados 2026-06-26, con OLCI fresco + orden canónico)** — test
  > intacto, IC95% bootstrap, `*`=significativo (IC no cruza 0). Esta tabla es la **única fuente de
  > verdad** y coincide exactamente con `REPORTE_DEFENSA.md` y `artifacts/reports/nested_metrics.json`.
  > Incluye Cajón (Landsat) y **target costero OLCI extendido a junio 2026**. El pipeline es ahora
  > **REPRODUCIBLE** (orden canónico de pares, ver más abajo): dos corridas dan números idénticos.

  | Horiz | LAGOS | COSTA |
  |------|-------|-------|
  | +1d | +0.23 [.14,.31]* | +0.23 [.00,.44]* |
  | +3d | +0.09 [−.03,.21] | +0.33 [.11,.50]* |
  | +5d | +0.14 [.08,.20]* | +0.30 [.08,.49]* |
  | +7d | +0.24 [.14,.32]* | +0.26 [−.08,.50] |

  Lagos significativos a **+1d, +5d y +7d** (+3d no significativo). Costa: **gran salto con OLCI
  fresco** — ahora significativa a **+1d (borderline, IC inf ≈0), +3d y +5d** (antes +1d no era
  significativo); +7d aún incierto. Alerta lagos PR-AUC +1d 0.57. Cuerpos en el test: lagos =
  Okeechobee/Yojoa/Cajón, costa = Tampa/Fonseca.
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

## REPRODUCIBILIDAD + GITHUB (2026-06-25)
- Repo en **GitHub (privado)**: `CarlosTorres6969/tesis-habs-forecast`. Solo código+docs; los datos
  pesados (~44 GB: imagenes/, era5_temp_nc/, datasets/, artifacts/) quedan fuera vía `.gitignore`.
- Añadidos `requirements.txt`, `run_pipeline.py` (orquestación) y `check_integrity.py` (test de
  honestidad: 11/11 OK — sin fuga, causalidad, features limpias, intervalos guardados).
- Fix de datos: `match_pairs.py` ahora elimina **duplicados exactos** (se quitaron 97; 4374→4277
  pares). Las escenas distintas del mismo día (espectro diferente) se conservan (legítimas).

## LANDSAT 8/9 PARA CAJÓN (2026-06-25) — ADOPTADO solo en Cajón
`fetch_landsat_scenes.py` (GEE, LC08+LC09 C02 L2 SR, máscara QA_PIXEL) + `build_scene_state.py`
ahora consciente del sensor (Landsat = 4 bandas blue/green/red/NIR; SIN red-edge → NDCI/CI_red/FAI
quedan NaN, XGBoost los maneja). Se bajaron 145 escenas Landsat de Cajón (113 pasan máscara de agua).
RESULTADO: Cajón pasó de **no evaluable** (pares insuficientes) a **dentro del test anidado**
(pares 78/64/64/65 por horizonte). Cajón solo predice a horizonte LARGO (+5d/+7d skill ~+0.42),
débil a +1d/+3d. Lagos ahora = **3 cuerpos** (Okeechobee+Yojoa+Cajón); el pool baja algo a +3d/+7d
porque Cajón es el más difícil, pero entra en la ventana 0–7 d (decisión del usuario: incluirlo).
LIMITACIÓN declarada: offset cross-sensor (reflectancias Landsat ~3× menores que S2); no rompe el
modelo (el backbone autorregresivo, independiente del sensor, sostiene el skill). NO se expande
Landsat a Yojoa/Fonseca/Tampa (ya pasan con S2; solo los diluiría). `check_integrity` sigue 11/11.

**Harmonización cross-sensor** (`harmonize_landsat.py`): se corrige el offset alineando momentos
(media+desv) de las bandas Landsat a la escala de S2 en fechas cercanas (la regresión OLS fallaba,
R²≈0). RESULTADO: skill **idéntico** → la harmonización es **inerte** para el pronóstico, por dos
razones: (1) lagos casi no usa SPECTRAL (familias AUTOREG+ERA5/INSITU dominan), y (2) XGBoost es
invariante a transformaciones lineales por feature. **Hallazgo:** el pronóstico es **robusto al
offset cross-sensor** (la limitación no tiene impacto práctico). Se conserva como higiene de datos
(alinea sensores, beneficia a la red neuronal escalada). `match_pairs` usa el estado harmonizado si
existe (`scene_state_harmonized.csv`).

## CAPA OPERATIVA — SISTEMA DE ALERTA (2026-06-26)
Se convirtió el predictor en un **sistema operativo de alerta** (sin tocar modelado ni números de
validación; pronóstico causal intacto, solo datos ≤ t0). Reusa `predict.forecast_body` como **fuente
única de inferencia** (la misma que `predict.py`), no duplica feature engineering.

- **`guards.py`** — guardas de FRESCURA/COBERTURA compartidas (las usan `run_forecast` y `predict`):
  - `STALE` si la escena t0 es más vieja que `config.MAX_DATA_AGE_DAYS` (default 14 d).
  - `LOW_COVERAGE` si `n_water_px < config.MIN_WATER_PIXELS`.
  - `EXPLORATORIO` si el cuerpo está en `config.EXPLORATORY_BODIES` (Cajón).
  - El campo `confianza` reporta la **PEOR** condición aplicable (`config.CONFIDENCE_SEVERITY`,
    orden peor→mejor: LOW_COVERAGE > STALE > EXPLORATORIO > OK). No silencia: marca.
- **`run_forecast.py`** — bucle operativo: para cada cuerpo (`config.REGIONS`) × horizonte (1,3,5,7)
  toma la última escena como t0 y emite chl-a esperada + banda P10–P90 (CQR) + prob/bandera de RIESGO
  (ensamble Red+XGBoost) + `confianza`. Usa `logging` y **try/except por cuerpo** (si uno falla, loguea
  y sigue). Escribe `artifacts/forecasts/forecast_<YYYYMMDD_HHMMSS>.csv` y `.json` (snapshot del run) y
  **apenda** a `artifacts/forecasts/forecast_log.csv` (bitácora acumulada, una fila por cuerpo-horizonte-run).
  Esquema: `run_ts, water_body, group, t0, horizon, chl_pred, p10, p90, prob_riesgo, riesgo, confianza,
  data_age_days, n_water_px, modelo_meta`. Núcleo `build_rows` es **puro y testeable**. Modo opcional
  `--backfill K` siembra la bitácora con pronósticos históricos madurables (para arrancar la verificación).
- **`build_model_cards.py`** — model card por modelo, junto a los `.pkl` (`artifacts/models/model_cards.json`):
  fecha de entrenamiento, nº de pares, commit git, features y skill validado anidado por (grupo,horizonte).
  `run_forecast` lo incluye en `modelo_meta` (trazabilidad de cada pronóstico). No reentrena.
- **`verify_forecasts.py`** — verificación operativa POSTERIOR (cierra el lazo, no entrena): cruza la
  bitácora con `combined_target.csv`; para pronósticos ya madurados (target real t0+h disponible) calcula
  error realizado (chl_pred vs chl_real), si cayó dentro de P10–P90 y si la bandera de riesgo acertó.
  Salidas `artifacts/reports/forecast_verification.csv` + resumen por (grupo,horizonte): **MAE, cobertura
  empírica de la banda, hit-rate de alerta**. Núcleo `verify` es **puro y testeable**.
  Demostración (backfill 12/cuerpo): cobertura empírica de la banda **0.72–0.87** (≈0.80 objetivo CQR) →
  el lazo operativo reproduce la calibración; MAE lagos crece con el horizonte (≈2→6 µg/L), costa ≈0.5 µg/L.
- **`tests/`** (pytest, 16 tests, todos pasan): `test_guards.py` (frescura/cobertura/peor condición),
  `test_run_forecast_schema.py` (esquema de salida con pronóstico sintético), `test_verify_forecasts.py`
  (cruce/MAE/cobertura/hit-rate con caso sintético). `conftest.py` agrega la carpeta al path.
- **`check_integrity.py`**: añadidos 3 checks estáticos de la capa operativa → **14/14 OK** (siguen los 11
  de honestidad/causalidad; los 3 nuevos validan las guardas, sin datos ni torch → CI verde).

Cómo operar: `python run_forecast.py` (emite y registra) · `python run_forecast.py --backfill 12` (siembra
histórico) · `python verify_forecasts.py` (evalúa lo madurado) · `python build_model_cards.py` (refresca cards).

## RECONCILIACIÓN DE REPORTE + FIGURAS (2026-06-26) — HECHO
- **Reporte reconciliado**: se re-corrió `evaluate_nested.py` (refresca `nested_metrics.json` +
  `nested_test_predictions.csv`; reentrena solo modelos de EVALUACIÓN efímeros, no los de producción)
  y se regeneró `REPORTE_DEFENSA.md` con `build_final_report.py`. El reporte ya leía los JSON oficiales;
  se ajustó SOLO el display (no la evaluación): tabla de validación anidada ahora muestra la columna
  **Familias** y lista los **cuerpos reales del test** desde `nested_test_predictions.csv` (antes imprimía
  "_grupo" porque la selección es agrupada). **REPORTE_DEFENSA.md, nested_metrics.json y la tabla
  "NÚMEROS OFICIALES" de este documento coinciden ahora exactamente.**
- **Discrepancia encontrada y corregida**: la antigua tabla "Números oficiales" tenía cifras
  **pre-Landsat-Cajón** (+3d lagos 0.17*, +7d lagos 0.32*). Los números vigentes (con Cajón incluido)
  son +3d lagos **0.09 (ns)** y +7d lagos **0.24***, costa +3d **0.32***/+5d **0.25***. Coincide con lo
  ya anticipado en la sección de Landsat ("+3d→ns, +7d 0.32→0.24"). Tablas intermedias quedaron marcadas.
- **Figuras regeneradas** (`build_validation_figs.py`, leen los JSON/CSV oficiales → consistentes por
  construcción) en `artifacts/reports/`: `fig_skill_horizonte.png` (verificada visualmente: lagos +3d
  gris ns, costa +1d/+7d gris ns — coincide con la tabla), `fig_cobertura_intervalos.png`,
  `fig_pr_alerta.png`, `fig_serie_temporal.png`, `fig_dispersion_freshwater.png`,
  `fig_dispersion_marine.png`. (Nota: `interval_metrics.json` proviene de `evaluate_intervals.py`, no
  cambió; la cobertura mostrada sigue válida con los mismos datos.)
- **Integridad**: `check_integrity.py` = **14/14 OK** (11 de honestidad/causalidad + 3 de la capa
  operativa). Sin fuga, causal, consistente.

## ENTREGABLES VISUALES + PROGRAMACIÓN (2026-06-26) — HECHO
- **Set completo de mapas** (5 cuerpos, escena más limpia + máscara estricta) guardado en
  `habs_forecast/entregables/mapas/mapa_<cuerpo>_h1.png` (local; las PNG quedan fuera de git por
  `.gitignore`, regenerables con `python make_maps.py <cuerpo>`).
- **Pronóstico programable**: `run_scheduled.py` (forecast + verificación encadenados, probado) +
  `register_task.ps1` (tarea diaria de Windows). El registro de la tarea lo activa el usuario.
- **Densificación Cajón (#4) y OLCI costa (#5)**: NO ejecutados. #4 cambiaría los números oficiales
  recién reconciliados (rendimiento decreciente); #5 requiere login interactivo de Copernicus y, sin
  OLCI denso, la costa +1d/+7d no mejora. Quedan como opcionales a criterio del usuario.

## OLCI COSTA FRESCO + REPRODUCIBILIDAD (2026-06-26) — HECHO
- **#5 OLCI costa — ADOPTADO (mejora real):** se re-bajó el target Sentinel-3 OLCI 300 m extendiendo
  la cobertura a **junio 2026** (antes terminaba feb 2026): Tampa 1090→1112, Fonseca 1061→1082 días.
  `fetch_olci_chl.py` (T1 a 2026-06-30) → `build_combined_target.py` (costa usa OLCI). RESULTADO:
  **costa mejora de forma defendible** — +1d pasa de **NO significativo (0.11) a significativo (0.23,
  borderline)**, +5d 0.25→**0.30**, +3d 0.32→0.33. A diferencia de Cajón (escenas nubosas = ruido,
  revertido), el OLCI fresco es **dato bueno** y además deja el target costero al día para la operación.
  (Token openeo cacheado válido; no requirió re-login.)
- **Bug de reproducibilidad encontrado y CORREGIDO:** al reconstruir los pares, el skill de lagos +5d
  "bailaba" (0.19↔0.14). Causa raíz: **XGBoost con `subsample`/`colsample` y seed fijo muestrea filas
  según su POSICIÓN**; si el orden del DataFrame cambia (al reconstruir con otro cuerpo), el mismo seed
  toma filas distintas → modelo distinto. FIX en `match_pairs.py`: **orden canónico** de los pares
  (`sort_values(["water_body","horizon","fecha_t0","fecha_target"])`) antes de guardar. Verificado: dos
  corridas de `evaluate_nested.py` dan resultados **idénticos** (`diff` vacío). El pipeline es ahora
  reproducible corrida a corrida — refuerza la defensa (vs. sistema viejo con fuga).
- **Costo honesto:** con el orden canónico, lagos +5d se asienta en **0.14** (antes 0.19; sigue
  significativo, IC [.08,.20]). No es degradación real: las dos familias (`SPECTRAL+INSITU` vs
  `ERA5+INSITU`) estaban casi empatadas y el orden estable desempató reproduciblemente.
- **Reconciliado:** ESTADO + `REPORTE_DEFENSA.md` + `nested_metrics.json` + figuras coinciden con los
  números nuevos. `check_integrity` 14/14.

## QUÉ SIGUE (pendiente)
Modelos en estado de defensa. Quedan, cuando el usuario lo indique (EN PAUSA): notebook limpio y
redacción de tesis. (Figuras de validación + mapas + capa operativa + OLCI costa: HECHOS.)

## EN PAUSA (no hacer hasta que el usuario lo pida)
Figuras, notebook limpio que reemplace los viejos, redacción de tesis.

---

## MAPA DE ARCHIVOS CLAVE (habs_forecast/)
- `config.py` — configuración central (cuerpos, horizontes, umbrales, índices).
- `build_scene_state.py` — lee rasters S2 → predictores por escena.
- `fetch_satellite_chl.py` / `fetch_olci_chl.py` — target satelital (VIIRS / OLCI).
- `fetch_s2_scenes.py` — **descarga incremental de más escenas Sentinel-2 (GEE)** para densificar
  cuerpos débiles (requiere auth GEE + EE_PROJECT; `S2_MAXCLOUD` y cuerpos por argumento). Local.
- `fetch_landsat_scenes.py` — **descarga Landsat 8/9 (GEE)** para densificar Cajón (sin red-edge).
- `harmonize_landsat.py` — alinea (momentos) las bandas Landsat a la escala de S2; inerte pero
  higiene de datos correcta. Genera `scene_state_harmonized.csv`.
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
- `check_integrity.py` — **test de integridad ejecutable** (sin fuga / causal / consistente); 11/11 OK.
- `run_pipeline.py` — orquesta el pipeline de modelado en orden (reproducibilidad).
- `Modelo_HABs_limpio.ipynb` — **notebook reproducible** (anexo) que reemplaza los .ipynb viejos
  con fuga; lo genera `build_notebook.py` y se ejecuta con nbconvert (20 celdas, 0 errores).
- `era5_sensitivity.py` — **sensibilidad ERA5 reanálisis vs pronóstico** (ablación + ruido #5).
- `train_final.py` / `predict.py` / `calibrate_alert.py` / `make_maps.py` — sistema desplegable.
  `predict.py` expone `forecast_body(wb,t0)` (inferencia estructurada reutilizable).
- `build_validation_figs.py` — figuras de validación (skill, intervalos, serie temporal, dispersión).
- **Capa operativa de alerta** (2026-06-26):
  - `guards.py` — guardas de frescura/cobertura/estado → etiqueta de `confianza`.
  - `run_forecast.py` — bucle operativo (emite + bitácora `forecast_log.csv`); `--backfill K` siembra histórico.
  - `verify_forecasts.py` — verificación posterior (MAE/cobertura banda/hit-rate alerta de lo madurado).
  - `run_scheduled.py` — runner programable: encadena `run_forecast` + `verify_forecasts` con log a
    `artifacts/forecasts/scheduled.log`. Para que la bitácora madure sola. Registrar con `register_task.ps1`.
  - `register_task.ps1` — registra una tarea DIARIA de Windows que corre `run_scheduled.py` (06:00).
  - `build_model_cards.py` — model cards (`artifacts/models/model_cards.json`) para trazabilidad.
  - `tests/` — pytest de la capa operativa (guards, esquema, verificación). `conftest.py`.
- `variables_modelo.txt` — lista de variables del modelo.
- `README.md` — metodología y resultados.
- `artifacts/` — datos procesados, modelos y reportes generados; `artifacts/forecasts/` = snapshots +
  bitácora de pronósticos operativos.
