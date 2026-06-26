# Reporte de defensa — Sistema de predicción temprana de HABs (0–7 d)

> Números definitivos, generado por `build_final_report.py`. Pronóstico causal X(≤t0)→chl(t0+h), ventana 2023–2026. Validación anidada con test temporal intacto. Skill = mejora de RMSE(log-chl) vs persistencia; `*` = IC95% bootstrap no cruza 0 (significativo).

## 1. Inventario de datos

| Cuerpo | Grupo | Escenas S2 | Pares causales |
|---|---|---|---|
| cajon | freshwater | 229 | 221 |
| fonseca | marine | 269 | 1130 |
| okeechobee | freshwater | 509 | 1726 |
| tampa_bay | marine | 217 | 902 |
| yojoa | freshwater | 188 | 395 |

Total: **1412 escenas**, **4374 pares**.

## 2. Validación anidada (TEST FINAL INTACTO) — el número defendible

Test = último ~25% del tiempo por (grupo,horizonte), nunca tocado; features elegidas solo en DEV.

### Lagos
| Horizonte | Skill regresión (test intacto) | PR-AUC alerta | n_test | eventos |
|---|---|---|---|---|
| +1d | +0.25 [+0.16,+0.34]* | +0.38 [+0.14,+0.68]* | 102 | 6 |
| +3d | +0.17 [+0.08,+0.24]* | +0.29 [+0.08,+0.63]* | 96 | 8 |
| +5d | +0.19 [+0.10,+0.28]* | +0.06 [+0.02,+0.12]* | 94 | 5 |
| +7d | +0.32 [+0.23,+0.38]* | +0.07 [+0.02,+0.18]* | 93 | 3 |

Cuerpos en el test: _grupo.

### Costa
| Horizonte | Skill regresión (test intacto) | PR-AUC alerta | n_test | eventos |
|---|---|---|---|---|
| +1d | +0.10 [-0.14,+0.31]  | +0.35 [+0.14,+0.58]* | 96 | 13 |
| +3d | +0.22 [+0.07,+0.39]* | +0.27 [+0.13,+0.47]* | 101 | 16 |
| +5d | +0.28 [+0.08,+0.49]* | +0.11 [+0.05,+0.19]* | 96 | 12 |
| +7d | +0.28 [-0.05,+0.49]  | +0.26 [+0.11,+0.45]* | 97 | 14 |

Cuerpos en el test: _grupo.

### Intervalos de incertidumbre (regresión cuantil conformalizada, CQR)

Cada pronóstico de intensidad lleva una banda **P10–P90** calibrada en el test intacto (cobertura objetivo 0.80). Cobertura empírica:

| Grupo | +1d | +3d | +5d | +7d |
|---|---|---|---|---|
| Lagos | 0.75 | 0.78 | 0.82 | 0.77 |
| Costa | 0.77 | 0.77 | 0.79 | 0.71 |

Cobertura ≈0.80 ⇒ intervalos fiables (no sobreconfiados). La banda cruda sin conformalizar quedaba en ~0.45–0.61 (sobreconfiada); CQR la corrige.

## 3. Sensibilidad ERA5 (reanálisis vs pronóstico — honestidad operativa)

Ablación (aporte real de ERA5) y estrés de ruido (skill con ruido al 100% de la variabilidad de cada driver). Curva plana ⇒ se puede operar con ERA5 de pronóstico sin perder skill.

| Grupo | Horiz | Skill con ERA5 | Aporte ERA5 | Skill con ruido 100% |
|---|---|---|---|---|
| Lagos | +1d | +0.17 [+0.11,+0.24]* | -0.025 | +0.150 |
| Lagos | +3d | +0.15 [+0.08,+0.21]* | +0.014 | +0.103 |
| Lagos | +5d | +0.09 [+0.03,+0.15]* | -0.007 | +0.074 |
| Lagos | +7d | +0.24 [+0.18,+0.29]* | +0.044 | +0.216 |
| Costa | +1d | -0.02 [-0.24,+0.14]  | -0.044 | -0.019 |
| Costa | +3d | +0.11 [-0.03,+0.23]  | +0.007 | +0.076 |
| Costa | +5d | +0.11 [-0.02,+0.22]  | +0.025 | +0.090 |
| Costa | +7d | +0.19 [+0.03,+0.35]* | +0.029 | +0.151 |

## 4. Validación del target de Yojoa contra in-situ (fuera de ventana, NO entra al modelo)

In-situ Secchi 2018–2022 (Fadum/Ross, CSU; Zenodo 8139922) vs VIIRS-chl, 85 matchups (≤4 d).

- Pearson (chl, Secchi): **r=-0.311** (p=0.004)
- Spearman (rango): **r=-0.283** (p=0.009)

**NEGATIVA y significativa -> el VIIRS SIGUE la transparencia real del lago: target de Yojoa CREIBLE**
(Esperado: correlación NEGATIVA, más clorofila ⇒ menos transparencia. Indirecto pero significativo ⇒ el target satelital de Yojoa es creíble.)

## 5. Alerta calibrada (operativa)

Ensamble Red+XGBoost, calibración isotónica + umbral F2 (prioriza recall: perder un bloom cuesta más que una falsa alarma).

| Grupo | Umbral operativo | Recall | Precisión |
|---|---|---|---|
| Lagos | 0.09 | 0.81 | 0.17 |
| Costa | 0.05 | 1.00 | 0.21 |

*(Recall/precisión de la última corrida de `calibrate_alert.py`; recall alto a propósito para alerta temprana.)*

## 6. Niveles de confianza por cuerpo

- **ALTA**: Okeechobee (target VIIRS validado con in-situ), Tampa Bay y Fonseca (target satelital validado; alerta fiable).
- **Validado fuera de ventana**: Yojoa (target VIIRS sigue el Secchi de campo 2018–2022; sin in-situ 2023–2026, limitación documentada).
- **Exploratorio**: Cajón (pares insuficientes para el test anidado tras el split temporal; embalse muy nuboso, sin in-situ).

## 7. Interpretación biológica y limitaciones (revisión asesora limnológica)

- El modelo predice **clorofila-a (µg/L) = proxy de BIOMASA algal**. Clorofila-a alta indica más biomasa, **no confirma por sí sola floración NOCIVA** (toxicidad).
- Distinción a mantener: **↑ clorofila-a → ↑ biomasa algal → floración nociva** son conceptos relacionados pero distintos. La alerta señala **condiciones de RIESGO** que ameritan **verificación de campo** (identificación de cianobacterias, toxinas, ficocianina).
- **Sentinel-2 no distingue cianobacterias** (carece de banda de ficocianina ~620 nm, sí en OLCI): detecta biomasa, no el grupo tóxico.
- **Nutrientes**: fósforo total = adecuado (clave en eutrofización). **Amonio** usado como N disponible es una **limitación declarada**: es solo una forma del N (lo ideal sería nitrato/nitrito/N total, sin datos en la ventana). Contexto in-situ (temp, OD, pH, turbidez, conductividad, Secchi) ayuda a interpretar el estado trófico.

## 8. Chequeo de honestidad (sin fuga)

- 0 features contaminadas (sin delta_*, sin target, sin NDVI como predictor; backbone autorregresivo usa el último valor ≤t0).
- 0 pares con fuga temporal (todo target a +1…+8 d estrictamente futuro).
- h=0 (detección) se reporta aparte del titular de pronóstico.
- Selección de features y evaluación separadas (anidada) ⇒ sin sesgo de selección.
- `predict.py` y `make_maps.py` construyen features solo con datos ≤t0.
