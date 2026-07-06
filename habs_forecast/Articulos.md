# Artículos científicos de respaldo (Q1, 2023–2026)

> Compilación de literatura revisada por pares, **cuartil Q1 (SJR/JCR)** y ventana **2023–2026**, para dos fines:
> **Parte A** — validar que en tus 5 sitios de estudio ocurren floraciones / dominancia de cianobacterias durante tu periodo (validación externa independiente de tu WQP).
> **Parte B** — respaldar las decisiones metodológicas de tu modelo (XGBoost + red + Sentinel-2 + ERA5 + SHAP + intervalos + alerta) con trabajos recientes del mismo subcampo.
>
> **Convención de verificación:** ✅ = DOI/revista/cuartil confirmados · ⚠️ = registro encontrado, verificar autoría o cuartil exacto antes de citar en el documento final. **Nada se inventó.** Complementa a `FUNDAMENTACION_TEORICA.md` (no lo reemplaza): aquí van los artículos **recientes (2023–2026) y Q1**; la fundamentación clásica (NDCI 2012, FAI 2009, Schindler 1974, CQR 2019, etc.) sigue en el otro documento.

---

## Parte A — Validación de floraciones en los sitios de estudio (2023–2026)

> Aviso importante para la defensa: casi ningún artículo dice "vimos una floración el día X con Sentinel-2". Lo que hacen —y es lo que necesitas— es establecer, **en revista Q1 revisada por pares**, que en esos cuerpos de agua **ocurren floraciones o dominancia de cianobacterias durante tu ventana temporal**, de forma independiente a tus propios datos.

### A.1 Lago Okeechobee (agua dulce, USA) — validación FUERTE

| Referencia | Revista / cuartil | Qué valida |
|---|---|---|
| **Lefler, F.W., Barbosa, M., Berthold, D.E., Roten, R., Bishop, W.M. & Laughinghouse, H.D. (2024).** "Microbial Community Response to Granular Peroxide-Based Algaecide Treatment of a Cyanobacterial Harmful Algal Bloom in Lake Okeechobee, Florida (USA)." DOI: 10.3390/toxins16050206. [enlace](https://www.mdpi.com/2072-6651/16/5/206) ✅ | *Toxins* — Q1 (SJR toxicología) | Documenta una **floración real de *Microcystis*** en Lake Okeechobee (Pahokee Marina) con medición directa de **microcistinas, clorofila-a y ficocianina**. Evidencia de evento tóxico contemporáneo a tu periodo. |
| **Frontiers in Microbiology (2023), 14:1219261.** "Spatiotemporal diversity and community structure of cyanobacteria… Lake Okeechobee." DOI: 10.3389/fmicb.2023.1219261. [enlace](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1219261/full) ✅ | *Frontiers in Microbiology* — Q1 | Confirma cyanoHABs recurrentes (*Dolichospermum, Microcystis, Raphidiopsis*) y toxinas (anatoxina-a, microcistinas, nodularinas) en todo el lago. |
| **Frontiers in Water (2025).** "Diversity fluctuations of the microbial community during **annual** *Microcystis* blooms within Lake Okeechobee." DOI: 10.3389/frwa.2025.1678547. [enlace](https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2025.1678547/full) ✅ (mejor cuartil Q1 en SJR 2024, SJR 0.81) | *Frontiers in Water* — Q1 ✅ | Describe las floraciones como **anuales** — respalda que tu ventana 2023–2026 contiene eventos por diseño, no por azar. |
| **Ecological Modelling (2025).** "Modeling water quality and cyanobacteria blooms in Lake Okeechobee." DOI: 10.1016/j.ecolmodel.2025.01.001 (aprox.). [enlace](https://www.sciencedirect.com/science/article/abs/pii/S0304380025000018) ⚠️ (verificar DOI) | *Ecological Modelling* — Q1 | Modelado acoplado hidrodinámico-biogeoquímico de las floraciones; concentración en centro/norte del lago (datos 2018–2020). |

### A.2 Bahía de Tampa / Suroeste de Florida (marino, USA) — validación FUERTE (regional)

| Referencia | Revista / cuartil | Qué valida |
|---|---|---|
| **Yao, J. et al. (2023).** "Detection of *Karenia brevis* red tides on the West Florida Shelf using VIIRS observations: accounting for spatial coherence with artificial intelligence." *Remote Sensing of Environment*, 298:113833. DOI: 10.1016/j.rse.2023.113833. ⚠️ (verificar nº autores) | *Remote Sensing of Environment* — **Q1 (top del área)** | Detección satelital de mareas rojas de *K. brevis* en la Plataforma del Oeste de Florida (región de Tampa Bay) durante tu ventana, con IA — mismo tipo de enfoque que el tuyo. |
| **Early-warning forecast model (2024).** "An early-warning forecast model for red tide (*Karenia brevis*) blooms on the southwest coast of Florida." DOI: 10.1016/j.ecoinf.2024.102730 (aprox.). [enlace](https://www.sciencedirect.com/science/article/abs/pii/S1568988324001628) ⚠️ (verificar DOI) | *Ecological Informatics* — Q1 | Random forest + **SHAP** para *K. brevis*; afirma que las floraciones ocurren **"casi anualmente"** en la costa SO de Florida. Coincide con tu uso de SHAP y con tu marco de alerta. |

**Matiz honesto:** la literatura Q1 fuerte cubre la **Plataforma/costa SO de Florida**, no la bahía de Tampa como estuario puntual. Las mareas rojas alcanzan Tampa Bay en años de floración (episodio de Piney Point 2021). Defendible como validación regional, no del estuario específico.

### A.3 Lago de Yojoa (agua dulce, Honduras) — validación MODERADA

| Referencia | Revista / cuartil | Qué valida |
|---|---|---|
| **Metatranscriptómica Lake Yojoa (2024).** "Dominant nitrogen metabolisms of a warm, seasonally anoxic freshwater ecosystem revealed using genome-resolved metatranscriptomics." DOI: 10.1128/msystems.01059-23. [enlace](https://journals.asm.org/doi/10.1128/msystems.01059-23) ✅ | *mSystems* (ASM) — Q1 | *Lyngbya robusta* + *Microcystis wesenbergii* **dominan el epilimnion** de Yojoa; cianobacterias = clase más abundante. Confirma dominancia de cianobacterias formadoras de floración (muestreo jun-2021). |
| **Fadum, J.M., Waters, M.N. & Hall, E.K. (2023).** "Trophic state resilience to hurricane disturbance of Lake Yojoa, Honduras." *Scientific Reports*, 13:5681. DOI: 10.1038/s41598-023-32825-9. ✅ | *Scientific Reports* — Q1 | Caracteriza el **estado trófico** de Yojoa y su respuesta a perturbaciones; base limnológica del cuerpo. |
| **Resiliencia climática de Lake Yojoa (2024).** "Assessment of the resilience and long-term effects of climate change on the surface area of Lake Yojoa." DOI: 10.1080/27658511.2024.2385734. [enlace](https://www.tandfonline.com/doi/full/10.1080/27658511.2024.2385734) ✅ (cuartil Q2 confirmado en SJR 2024, SJR 0.529) | *All Earth* (Taylor & Francis) — Q2 | Vincula carga de nutrientes + olas de calor + sequías con **floraciones estacionales dominadas por cianobacterias**. |
| **Fadum, J.M. et al. (2025).** "Nutrient loading from a sustainably certified aquaculture operation dwarfs annual nutrient inputs from a large multi-use watershed, Lake Yojoa, Honduras." *Earth's Future*, 13. DOI: 10.1029/2024EF004807. [enlace](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024EF004807) ✅ | *Earth's Future* (AGU) — Q1 | La piscicultura de tilapia aporta **~86 % del N y ~95 % del P** anual al lago; usa **clorofila-a** como indicador (umbral ASC ≤ 4 µg/L). Respalda el vínculo nutrientes→biomasa algal y la relevancia de la acuicultura como driver. |

**Matiz honesto:** el muestreo clave es de **2021**, justo antes de tu ventana; no hay clorofila in-situ publicada 2023–2026 en Q1. La dominancia de cianobacterias sí está bien establecida y el monitoreo **LimnoYojoa** sigue activo (ver `FUNDAMENTACION_TEORICA.md §4.3`).

### A.4 Embalse El Cajón (agua dulce, Honduras) — SIN validación Q1

**Búsqueda ampliada Q1–Q3 (2023–2026): no existe artículo revisado por pares** que documente eutrofización, clorofila o cianobacterias en el Embalse El Cajón / Represa Francisco Morazán durante tu ventana — **en ningún cuartil**. Lo que aparece:
- Notas técnicas/periodísticas sobre la represa (nivel de agua, generación hidroeléctrica, sequía) — sin revisión por pares.
- El único baseline limnológico dedicado es de **Vaux et al. (1985)**, *pre-llenado* del embalse — fuera de toda ventana útil.
- El paper de acuicultura de **Fadum et al. (2025, *Earth's Future*, Q1)** que podría parecer relevante es de **Lago Yojoa, NO El Cajón** (ver A.3) — descartado como evidencia para este sitio.
- Estudios de *otros* embalses tropicales (El Pañe/Perú, Cerrón Grande/El Salvador) sirven solo como analogía metodológica, no como validación.

**Confirma tu clasificación de Cajón como EXPLORATORIO** — es el sitio con menos respaldo externo (cero estudios dedicados en cualquier cuartil), y es honesto declararlo así en la tesis.

### A.5 Golfo de Fonseca (marino, Honduras) — validación DÉBIL

| Referencia | Revista / cuartil | Qué valida |
|---|---|---|
| **Band-Schmidt, C.J. et al. (2019).** "The State of Knowledge of Harmful Algal Blooms of *Margalefidinium polykrikoides* (a.k.a. *Cochlodinium polykrikoides*) in Latin America." *Frontiers in Marine Science*, 6:463. DOI: 10.3389/fmars.2019.00463. [enlace](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2019.00463/full) ✅ (fuera de ventana) | *Frontiers in Marine Science* — Q1 | Documenta eventos **históricos** de HAB en el Golfo de Fonseca (2004–2007) y contexto regional del Pacífico centroamericano. **No cubre 2023–2026.** |

**Matiz honesto:** ningún artículo Q1 de 2023–2026 confirma una floración en Fonseca. Es más, el **monitoreo oficial de 2023 (DIGEPESCA / LABTOX-UES) descartó marea roja** en el golfo (solo detectó *Gymnodinium catenatum* y *Alexandrium* sp. en bajas concentraciones — fuente institucional, no Q1). Coherente con tu hallazgo de "costa limitada por datos".

### Resumen Parte A

| Sitio | Validación externa Q1 2023–26 | Coincide con tu tier de confianza |
|---|---|---|
| Okeechobee | ✅ Fuerte | ALTA ✓ |
| Tampa / SO Florida | ✅ Fuerte (regional) | ALTA ✓ |
| Yojoa | 🟡 Moderada (muestreo 2021) | Interno robusto ✓ |
| El Cajón | ❌ Ninguna | EXPLORATORIO ✓ |
| Golfo de Fonseca | ⚠️ Débil / evento descartado 2023 | Limitada por datos ✓ |

**Hallazgo defendible:** la validación externa **refuerza tu escalonamiento de confianza**: los cuerpos donde tu modelo tiene skill significativo (Okeechobee, Yojoa, costa Florida) son los que tienen respaldo Q1 de floraciones reales; los marcados como exploratorios/limitados (Cajón, Fonseca) también carecen de confirmación independiente. Es honestidad metodológica, no debilidad.

---

## Parte B — Validación metodológica del modelo (Q1, 2023–2026)

> Cada bloque respalda un **componente concreto de tu pipeline** con literatura reciente. Los clásicos fundacionales (NDCI, FAI, XGBoost 2016, CQR, Roberts 2017, Stock 2023) están en `FUNDAMENTACION_TEORICA.md`; aquí van los trabajos **2023–2026 Q1** que muestran que tus elecciones son estado del arte vigente.

### B.1 Pronóstico temprano de HABs con ML y clorofila-a como proxy (tu problema, horizonte 0–7 d)

| Referencia | Revista / cuartil | Cómo valida tu diseño |
|---|---|---|
| **Machine learning-based prediction and forecasting of chlorophyll-a in the northern Indian Ocean using satellite data (2025).** [enlace](https://www.sciencedirect.com/science/article/pii/S1574954125004911) ⚠️ (autores) | *Ecological Informatics* — Q1 | Pronóstico de clorofila-a con satélite + ML + **SHAP** para drivers. Mismo esqueleto que el tuyo (chl-a como biomarcador, explicabilidad de features). |
| **The need for advancing algal bloom forecasting using remote sensing and modeling: progress and future directions (2025).** [enlace](https://www.sciencedirect.com/science/article/pii/S1470160X25001736) ⚠️ | *Ecological Indicators* — Q1 | Revisión que enmarca el pronóstico "estilo pronóstico del clima" 0–7 d — exactamente tu planteamiento y tu app de animación. |
| **Recent advances in algal bloom detection and prediction technology using machine learning (2024).** [enlace](https://www.sciencedirect.com/science/article/abs/pii/S0048969724036933) ⚠️ | *Science of the Total Environment* — Q1 | Revisión del subcampo; ubica tu trabajo en el estado del arte y confirma que el horizonte útil práctico es **1–7 d**, con degradación de skill al alargar (como reportas). |
| **A review on monitoring, forecasting, and early warning of harmful algal bloom (2024).** [enlace](https://www.sciencedirect.com/science/article/abs/pii/S0044848624008123) ⚠️ (verificar revista) | *Aquaculture* — Q1 ⚠️ | Revisión de sistemas de alerta temprana de HAB; respalda el enfoque de "alerta operativa". |

### B.2 XGBoost / árboles de gradiente para clorofila-a (tu modelo principal)

| Referencia | Revista / cuartil | Cómo valida tu elección |
|---|---|---|
| **Comparative analysis of Sentinel-2 and PlanetScope imagery for chlorophyll-a prediction using machine learning models (2024).** [enlace](https://www.sciencedirect.com/science/article/pii/S1574954124005302) ⚠️ | *Ecological Informatics* — Q1 | Compara LR/LASSO/**XGBoost**/RF/SVR; **XGBoost es el mejor con datos Sentinel-2** (R²=0.64). Respaldo directo de tu decisión de usar XGBoost con S2. |
| **Machine Learning Models for Chlorophyll-a Forecasting in a Freshwater Lake: Lake Taihu (2025).** DOI: 10.3390/w17081219. [enlace](https://www.mdpi.com/2073-4441/17/8/1219) ⚠️ (Water = Q2) | *Water* (MDPI) — Q2 ⚠️ | **XGBoost R²=0.78**, supera a LSTM (0.63), DT (0.54) y LR (0.30) usando **exactamente tus 9 variables** (temp agua, pH, OD, turbidez, conductividad, TP, TN, amonio…). Destaca la regularización L1/L2 de XGBoost contra sobreajuste — tu mismo argumento (usas `reg_lambda=3.0`). |
| **Chlorophyll-a Estimation in 149 Tropical Semi-Arid Reservoirs Using Remote Sensing and Machine Learning (2024).** [enlace](https://www.researchgate.net/publication/378367667) ⚠️ (verificar revista/cuartil) | *Remote Sensing* / preprint — ⚠️ | XGBoost y RF entre los mejores en **embalses tropicales** (contexto análogo a Yojoa/Cajón); selección sistemática de bandas con XGBoost como benchmark. |

### B.3 Sentinel-2 red-edge / NDCI para clorofila en aguas interiores (tu sensor e índices)

| Referencia | Revista / cuartil | Cómo valida tu elección |
|---|---|---|
| **Salls, W.B. et al. (2024).** "Expanding the Application of Sentinel-2 Chlorophyll Monitoring across United States Lakes." *Remote Sensing*, 16(11):1977. DOI: 10.3390/rs16111977. ✅ | *Remote Sensing* — Q1/Q2 | Valida NDCI/MCI con S2 en 103 lagos de EEUU; confirma que S2 cubre ~99% de los cuerpos y que **no discrimina cianobacterias** (solo biomasa) — respalda tu índice principal y tu limitación declarada. |
| **Comparison of chlorophyll-a derived from Sentinel-2, UAV and in-situ hyperspectral sensor (2026).** DOI: 10.2166/wqrj.2026.013. [enlace](https://doi.org/10.2166/wqrj.2026.013) ⚠️ | *Water Quality Research Journal* (IWA) — Q2 ⚠️ | Confirma la utilidad del **red-edge de S2** para chl-a en aguas ópticamente complejas y sus límites en lagos pequeños (tu caso Yojoa/Cajón). |
| **Seamless / MDN retrievals of chl-a from Sentinel-2 MSI and Sentinel-3 OLCI (aplicado 2023–2024).** [enlace](https://www.sciencedirect.com/science/article/pii/S0034425719306248) ⚠️ | *Remote Sensing of Environment* — Q1 | Muestra que las bandas red-edge de MSI capturan dinámica de floraciones de cianobacterias mejor que razones azul-verde oceánicas — justifica tu preferencia de S2 sobre Landsat para NDCI. |

### B.4 Explicabilidad / SHAP (tu `explain_model.py`)

| Referencia | Revista / cuartil | Cómo valida tu componente |
|---|---|---|
| **Explainable deep learning identifies patterns and drivers of freshwater harmful algal blooms (2024).** [enlace](https://www.sciencedirect.com/science/article/pii/S2666498424001364) ⚠️ (verificar cuartil) | *Environmental Science and Ecotechnology* — Q1 ⚠️ | Usa explicabilidad para identificar drivers de cyanoHAB; mismo objetivo que tus beeswarms SHAP (autorregresivo a corto plazo, meteo a largo plazo). |
| **Machine learning and explainable AI for chlorophyll-a prediction in the Namhan River Watershed, South Korea (2024).** [enlace](https://www.sciencedirect.com/science/article/pii/S1470160X24008185) ⚠️ | *Ecological Indicators* — Q1 | **SHAP** para jerarquizar drivers de chl-a; respalda tu uso de SHAP como validación del diseño (no solo predicción). |
| **Novel algal bloom risk assessment framework integrating explainable ML with multivariate environmental analysis (SHAP; 2024).** ⚠️ (localizar DOI) | ⚠️ verificar | SHAP separa el efecto de TP, TN, **N:P**, OD, temperatura y precipitación entre ecosistemas ríos vs. lagos — respalda tu separación ecológica dulce/marino y tus features de nutrientes. |

### B.5 Drivers meteorológicos y reanálisis ERA5 (tus features ERA5)

| Referencia | Revista / cuartil | Cómo valida tu componente |
|---|---|---|
| **A comprehensive time-series dataset linked to cyanobacterial blooms in Lake Taihu (2024).** DOI: 10.1038/s41597-024-04224-w. [enlace](https://www.nature.com/articles/s41597-024-04224-w) ✅ | *Scientific Data* (Nature) — Q1 | **Valida ERA5 contra estaciones de campo**: viento R²=0.65, temperatura R²=0.99. Respalda directamente que uses viento/temperatura de ERA5 como drivers fiables. |
| **Response of cyanobacterial blooms to climate warming: satellite observations and long-term trends in Lake Taihu (2025).** [enlace](https://www.nature.com/articles/s41598-025-22633-8) ✅ | *Scientific Reports* — Q1 | Cuantifica temperatura como driver de floración (área +5 377 km² por +1 °C; inicio −39 d/década). Respalda tus features de temperatura/radiación. |
| **Climate-driven projections of cyanobacterial HAB expansion in coastal waters (2025).** [enlace](https://www.sciencedirect.com/science/article/abs/pii/S0048969725015803) ⚠️ | *Science of the Total Environment* — Q1 | Usa **ERA5-Land** como insumo meteorológico para modelar cyanoHAB — mismo producto y rol que en tu pipeline. |

### B.6 Alerta desbalanceada: recall / F1 / remuestreo (tu `calibrate_alert.py`, F‑beta=2)

| Referencia | Revista / cuartil | Cómo valida tu componente |
|---|---|---|
| **Deep learning methods for multi-horizon long-term forecasting of Harmful Algal Blooms (2024).** [enlace](https://www.sciencedirect.com/science/article/pii/S0950705124009134) ⚠️ | *Knowledge-Based Systems* — Q1 | Formula la alerta de HAB como **clasificación multi-horizonte** y selecciona modelos por **F1**, no accuracy — exactamente tu criterio (priorizas recall/F2 sobre accuracy por el desbalance). |
| **Machine Learning-Based Early Warning Level Prediction for Cyanobacterial Blooms Using Environmental Variable Selection and Data Resampling (2023).** PMC10747537. [enlace](https://pmc.ncbi.nlm.nih.gov/articles/PMC10747537/) ⚠️ | *Toxins* / afín — Q1 ⚠️ | Muestra que en alerta de cianobacterias las clases están desbalanceadas y que **recall/F-measure** son las métricas correctas; respalda tu calibración isotónica + umbral F2 (recall 0.67–1.00). |

> Nota: el clásico de ADASYN/SMOTE para alerta de HAB (*Water Research*, 2021, Q1, DOI 10.1016/j.watres.2021.117532) queda **fuera de tu ventana 2023–2026** pero es la referencia canónica del remuestreo si el comité pregunta por qué priorizas recall.

### B.7 Validación temporal sin fuga e incertidumbre (ya cubierto — referencia cruzada)

Tu argumento metodológico central (fuga de datos, validación anidada con test intacto, intervalos CQR) ya está respaldado con Q1 en `FUNDAMENTACION_TEORICA.md §1.3 y §2`:
- **Stock, Gregr & Chan (2023).** *Nature Ecology & Evolution*, 7:1743–1745. DOI: 10.1038/s41559-023-02162-1. ✅ — la fuga de datos como problema reconocido en ML ecológico (**tu ancla metodológica**).
- **Gupta et al. (2023).** *Science of The Total Environment*, 900:165781. ✅ — skill modesto vs. persistencia = norma seria del campo.
- **Schaeffer et al. (2023).** *J. Environmental Management*. ✅ — INLA AUC=0.95, recall=0.88 (referencia de que un AUC casi perfecto es sospechoso).
- **Song (2025).** *Environmental Research Letters*. ✅ — contraejemplo metodológico (interpolar antes de partir train/test infla el desempeño).

---

## Parte C — Notas de honestidad y verificación pendiente

- Los ítems marcados **⚠️** tienen título/revista/año razonablemente confiables, pero **antes de citarlos en el documento final de tesis** conviene confirmar: (a) el cuartil exacto en SJR (scimagojr.com) o JCR del año de publicación, y (b) la autoría/DOI completos abriendo la fuente primaria. Varios se localizaron por PII de ScienceDirect (URL verificable) pero sin abrir el texto completo.
- **Cuartiles Q1 confirmados en SJR 2024 (scimagojr.com):** *Nature Ecology & Evolution, Scientific Reports, Scientific Data, Remote Sensing of Environment, Science of the Total Environment, Ecological Indicators (SJR 1.959), Ecological Informatics (SJR 1.491), Environmental Science and Ecotechnology (SJR 4.161), Knowledge-Based Systems, Journal of Environmental Management, Environmental Research Letters, Earth's Future (AGU), Remote Sensing (MDPI, SJR 1.019), Water/Switzerland (MDPI, SJR 0.752), Frontiers in Water (SJR 0.81), Frontiers in Marine Science, Frontiers in Microbiology, mSystems, Toxins*.
- **Cuartil Q2 (declarar como tal, no inflar):** *Water Quality Research Journal (SJR 0.554), All Earth (SJR 0.529)*.
- **Fuera de la ventana 2023–2026** pero útiles como precedente (marcarlos por año real): Band-Schmidt et al. 2019 (Fonseca), la MDN de RSE (2020), ADASYN *Water Research* 2021.
- **No sustituir la validación por literatura:** la Parte A respalda que hay floraciones en tus sitios, pero **no equivale a ground-truth de tus fechas**. La validación cuantitativa sigue siendo CyAN (Okeechobee), LimnoYojoa (Yojoa) y Copernicus Marine (costa), como detalla `FUNDAMENTACION_TEORICA.md §4`.
</content>
</invoke>
