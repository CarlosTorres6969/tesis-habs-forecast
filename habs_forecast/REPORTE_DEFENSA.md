# Reporte de defensa — Sistema de predicción temprana de HABs (0–7 d)

> Números definitivos, generado por `build_final_report.py`. Pronóstico causal X(≤t0)→chl(t0+h), ventana 2023–2026. Validación anidada con bloque temporal reservado. Skill = mejora de RMSE(log-chl) vs persistencia; `*` se aplica solo al skill cuando su IC95% no cruza 0.

## 1. Inventario de datos

| Cuerpo | Grupo | Escenas S2 | Pares causales |
|---|---|---|---|
| cajon | freshwater | 343 | 288 |
| fonseca | marine | 270 | 1038 |
| okeechobee | freshwater | 539 | 1700 |
| tampa_bay | marine | 219 | 914 |
| yojoa | freshwater | 188 | 391 |

Total: **1559 escenas**, **4331 pares**.

## 2. Validación anidada (TEST FINAL TEMPORAL) — el número defendible

Test = último ~25% de las fechas, con **un corte común para todos los cuerpos** de cada grupo-horizonte. Toda etiqueta de DEV queda antes del primer predictor de TEST; la selección usa folds internos purgados. Los IC95% usan bootstrap por bloques de 14 días y cuerpo.

### Lagos
| Horizonte | Skill regresión (test intacto) | PR-AUC alerta | n_test | eventos | Familias |
|---|---|---|---|---|---|
| +1d | +0.05 [-0.07,+0.18]  | 0.07 [0.01,0.24] | 128 | 10 | AUTOREG+INSITU |
| +3d | +0.05 [-0.05,+0.12]  | 0.08 [0.02,0.19] | 113 | 9 | AUTOREG+INSITU |
| +5d | +0.05 [-0.12,+0.12]  | 0.09 [0.01,0.53] | 117 | 6 | AUTOREG+INSITU |
| +7d | +0.09 [+0.02,+0.26]* | 0.04 [0.02,0.09] | 124 | 7 | AUTOREG+SPECTRAL+INSITU |

Cuerpos en el test: cajon, okeechobee, yojoa.

### Costa
| Horizonte | Skill regresión (test intacto) | PR-AUC alerta | n_test | eventos | Familias |
|---|---|---|---|---|---|
| +1d | +0.24 [+0.02,+0.44]* | 0.35 [0.10,0.68] | 94 | 12 | AUTOREG |
| +3d | +0.32 [+0.12,+0.52]* | 0.14 [0.07,0.29] | 93 | 10 | AUTOREG+SPECTRAL+INSITU |
| +5d | +0.28 [+0.07,+0.45]* | 0.11 [0.07,0.23] | 94 | 11 | AUTOREG+SPECTRAL |
| +7d | +0.29 [-0.05,+0.52]  | 0.18 [0.08,0.34] | 95 | 13 | AUTOREG |

Cuerpos en el test: fonseca, tampa_bay.

### Intervalos de incertidumbre (regresión cuantil conformalizada, CQR)

Cada pronóstico de intensidad lleva una banda **P10–P90** calibrada exclusivamente en CALIB, dentro de DEV, y evaluada después en TEST (cobertura objetivo 0.80). Cobertura empírica:

| Grupo | +1d | +3d | +5d | +7d |
|---|---|---|---|---|
| Lagos | 0.79 | 0.73 | 0.85 | 0.68 |
| Costa | 0.84 | 0.83 | 0.81 | 0.79 |

Cobertura cercana a 0.80 ⇒ intervalos razonablemente calibrados. La banda cruda sin conformalizar quedó en 0.32–0.69; CQR mejoró su cobertura.

## 3. Sensibilidad ERA5 (reanálisis vs pronóstico — honestidad operativa)

Ablación y estrés simulado de ruido (hasta 100% de la variabilidad de cada driver). Una curva estable sugiere baja sensibilidad al error meteorológico, pero no sustituye validar un producto ERA5 de pronóstico real.

| Grupo | Horiz | Skill con ERA5 | Aporte ERA5 | Skill con ruido 100% |
|---|---|---|---|---|
| Lagos | +1d | +0.04 [-0.10,+0.16]  | +0.008 | +0.011 |
| Lagos | +3d | +0.01 [-0.10,+0.09]  | -0.001 | -0.103 |
| Lagos | +5d | +0.00 [-0.16,+0.08]  | -0.050 | -0.052 |
| Lagos | +7d | +0.13 [+0.03,+0.21]* | +0.004 | +0.090 |
| Costa | +1d | +0.00 [-0.20,+0.18]  | -0.007 | -0.010 |
| Costa | +3d | +0.17 [+0.03,+0.28]* | -0.012 | +0.138 |
| Costa | +5d | +0.11 [-0.05,+0.25]  | +0.007 | +0.101 |
| Costa | +7d | +0.19 [+0.00,+0.38]* | +0.045 | +0.187 |

## 4. Validación del target de Yojoa contra in-situ (fuera de ventana, NO entra al modelo)

In-situ Secchi 2018–2022 (Fadum/Ross, CSU; Zenodo 8139922) vs VIIRS-chl, 85 matchups (≤4 d).

- Pearson (chl, Secchi): **r=-0.311** (p=0.004)
- Spearman (rango): **r=-0.283** (p=0.009)

**NEGATIVA y significativa -> el VIIRS SIGUE la transparencia real del lago: target de Yojoa CREIBLE**
(Esperado: correlación NEGATIVA, más clorofila ⇒ menos transparencia. Indirecto pero significativo ⇒ el target satelital de Yojoa es creíble.)

## 5. Alerta calibrada (operativa)

Ensamble Red+XGBoost, calibración isotónica + umbral F2 (prioriza recall: perder un bloom cuesta más que una falsa alarma).

| Grupo | Umbral elegido en DEV | Recall TEST (IC95%) | Precisión TEST (IC95%) | F2 TEST |
|---|---|---|---|---|
| Lagos | 0.05 | 0.56 [0.21,0.81] | 0.06 [0.01,0.12] | 0.22 [0.05,0.34] |
| Costa | 0.10 | 1.00 [1.00,1.00] | 0.12 [0.08,0.17] | 0.42 [0.30,0.51] |

*Isotónica y umbral se ajustan con predicciones OOS purgadas de DEV. Recall, precisión y F2 se calculan una sola vez en TEST; no son métricas de reajuste de producción.*

## 6. Niveles de confianza por cuerpo

- **Okeechobee**: la escala VIIRS se calibró solo con campo de 2023. La validación causal posterior tiene cobertura muy limitada (2/1/1/0 fechas en +1/+3/+5/+7 d), por lo que no permite afirmar desempeño de campo concluyente.
- **Tampa Bay y Fonseca**: cuentan con validación temporal del target satelital a nivel de grupo; la alerta prioriza sensibilidad y presenta baja precisión, por lo que exige confirmación de campo.
- **Validado fuera de ventana**: Yojoa (target VIIRS sigue el Secchi de campo 2018–2022; sin in-situ 2023–2026, limitación documentada).
- **Exploratorio**: Cajón (sí participa en el test temporal, pero sigue sin validación in-situ independiente y presenta alta nubosidad).

## 7. Interpretación biológica y limitaciones (revisión asesora limnológica)

- El modelo predice **clorofila-a (µg/L) = proxy de BIOMASA algal**. Clorofila-a alta indica más biomasa, **no confirma por sí sola floración NOCIVA** (toxicidad).
- Distinción a mantener: **↑ clorofila-a → ↑ biomasa algal → floración nociva** son conceptos relacionados pero distintos. La alerta señala **condiciones de RIESGO** que ameritan **verificación de campo** (identificación de cianobacterias, toxinas, ficocianina).
- **Sentinel-2 no distingue cianobacterias** (carece de banda de ficocianina ~620 nm, sí en OLCI): detecta biomasa, no el grupo tóxico.
- **Nutrientes**: fósforo total = adecuado (clave en eutrofización). **Amonio** usado como N disponible es una **limitación declarada**: es solo una forma del N (lo ideal sería nitrato/nitrito/N total, sin datos en la ventana). Contexto in-situ (temp, OD, pH, turbidez, conductividad, Secchi) ayuda a interpretar el estado trófico.

## 8. Chequeo de honestidad (sin fuga)

- 0 features contaminadas (sin delta_*, sin target, sin NDVI como predictor; backbone autorregresivo usa el último valor ≤t0).
- 0 pares con fuga temporal (todo target a +1…+8 d estrictamente futuro).
- h=0 (detección) se reporta aparte del titular de pronóstico.
- Corte cronológico común entre cuerpos y embargo verificado en cada frontera temporal.
- La corrección de escala de Okeechobee está congelada al 2023-12-31; ningún dato del TEST interviene en su ajuste.
- Selección de features y umbral dentro de DEV; el modelo de producción reutiliza esas familias.
- IC95% por bloques cuerpo-tiempo, sin tratar escenas repetidas como observaciones iid.
- `predict.py` y `make_maps.py` construyen features solo con datos ≤t0.
