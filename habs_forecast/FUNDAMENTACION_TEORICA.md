# Fundamentación teórica y validación de la investigación

> Documento de trabajo (no es el documento de tesis). Compila literatura científica y fuentes institucionales reales para respaldar las decisiones de diseño del sistema y para abrir vías de validación externa. Cada referencia indica su nivel de verificación: **✅ verificada** (confirmada con DOI/fuente primaria) o **⚠️ verificar antes de citar** (registro bibliográfico encontrado pero algún detalle —autores, cifra exacta— no se pudo confirmar con certeza; revisar la fuente original antes de incluirla en el documento final). Ningún dato se inventó: donde no hubo certeza, se marcó explícitamente en vez de rellenar el hueco.

---

## 0. Cómo usar este documento

Cuatro frentes de fundamentación, cada uno respondiendo una pregunta distinta del comité:

1. **Respaldo metodológico** — "¿por qué elegiste estas técnicas?"
2. **Estado del arte** — "¿cómo se compara esto con lo que ya existe?"
3. **Respaldo biológico/limnológico** — "¿por qué interpretas los resultados así, y no como 'detección de floraciones tóxicas'?"
4. **Validación externa** — "¿hay alguna forma de comprobar los resultados fuera de tus propios datos?"

---

## 1. Respaldo metodológico (decisiones técnicas)

### 1.1 Índices espectrales para clorofila-a / biomasa algal

| Referencia | Aporta |
|---|---|
| **Mishra, S. & Mishra, D.R. (2012).** "Normalized difference chlorophyll index: A novel model for remote estimation of chlorophyll-a concentration in turbid productive waters." *Remote Sensing of Environment*, 117, 394–406. DOI: 10.1016/j.rse.2011.10.016. ✅ | Paper fundacional del **NDCI** = (NIR−Rojo)/(NIR+Rojo) usando red-edge (~708 nm) y rojo (~665 nm). Validado en aguas caso-2 turbias/productivas (R²=0.9). Justifica usar B4 (665 nm) y B5 (705 nm, red-edge) de Sentinel-2 — Landsat no tiene red-edge, por eso S2 es preferible para este índice. |
| **Hu, C. (2009).** "A novel ocean color index to detect floating algae in the global oceans." *Remote Sensing of Environment*, 113(10), 2118–2129. DOI: 10.1016/j.rse.2009.05.012. ✅ | Paper original del **FAI** (Floating Algae Index). Justifica el uso de B8 (NIR) y B4 (rojo) para detectar acumulaciones superficiales/scum. |
| **Wynne, T.T. et al. (2008).** "Relating spectral shape to cyanobacterial blooms in the Laurentian Great Lakes." *Int. J. Remote Sensing*, 29(12), 3665–3672, **y** Wynne, T.T. et al. (2010), *Limnology and Oceanography*, 55(5), 2025–2036. ✅ (citar ambos juntos) | Algoritmo y validación del **Cyanobacteria Index (CI)** (forma espectral 620/665/681/709 nm, heredado de MERIS→OLCI). El de 2008 define el algoritmo; el de 2010 lo valida contra el bloom de *Microcystis* del Lago Erie 2008. |
| Revisión "Sentinel-2 for chlorophyll-a water quality monitoring" (*Int. J. Remote Sensing*, 2026) reportando NDCI con S2 R²≈0.82 en lagos. ⚠️ | No se pudo confirmar autoría exacta — verificar en la fuente antes de citar, o sustituir por Salls et al. 2024 (sección 2). |

### 1.2 XGBoost vs. redes neuronales profundas en datos tabulares moderados (~2 000–4 500 filas)

| Referencia | Aporta |
|---|---|
| **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** "Why do tree-based models still outperform deep learning on tabular data?" *NeurIPS 35*, Datasets & Benchmarks. arXiv:2207.08815. ✅ | Benchmark extenso: los árboles de gradiente siguen siendo estado del arte en datos tabulares de tamaño medio (~10 000 filas), superando a deep learning. Referencia central para justificar XGBoost sobre redes profundas en el rango de tamaño de este dataset. |
| **Shwartz-Ziv, R. & Armon, A. (2022).** "Tabular data: Deep learning is not all you need." *Information Fusion*, 81, 84–90. DOI: 10.1016/j.inffus.2021.11.011. ✅ | Ningún modelo de deep learning evaluado supera de forma confiable a XGBoost en datos tabulares heterogéneos (justo el caso: bandas espectrales + meteorología + in-situ mezclados). |
| **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD '16*, 785–794. DOI: 10.1145/2939672.2939785. ✅ | Paper original de XGBoost; su manejo nativo de **datos faltantes** (sparsity-aware split finding) es directamente relevante porque las variables in-situ de WQP tienen huecos. |

### 1.3 Validación temporal y cuantificación de incertidumbre

| Referencia | Aporta |
|---|---|
| **Romano, Y., Patterson, E. & Candès, E.J. (2019).** "Conformalized Quantile Regression." *NeurIPS 32*, 3538–3548. ✅ | Paper original de **CQR**: cobertura garantizada en muestra finita, adaptativa a heterocedasticidad — exactamente lo usado en `evaluate_intervals.py` para las bandas P10–P90. |
| **Roberts, D.R. et al. (2017).** "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography*, 40(8), 913–929. DOI: 10.1111/ecog.02881. ✅ | La referencia más citada en ecología para justificar **block/walk-forward CV** en vez de k-fold aleatorio cuando hay estructura temporal/espacial — respalda directamente la validación anidada con test final intacto. |

### 1.4 Validación de productos satelitales de clorofila (VIIRS, OLCI) vs. in-situ

| Referencia | Aporta |
|---|---|
| **Kravitz, J., Matthews, M., Bernard, S. & Griffith, D. (2020).** "Application of Sentinel-3 OLCI for chl-a retrieval over small inland water targets." *Remote Sensing of Environment*, 237, 111562. DOI: 10.1016/j.rse.2019.111562. ✅ | Documenta directamente las limitaciones de OLCI (300 m) en **lagos pequeños** por contaminación de píxel mixto en orillas — respalda por qué OLCI se descartó para Yojoa/Cajón en este proyecto. |
| **MDPI *Water* (2021), 13(14), 1903.** "Comparison of in-situ chlorophyll-a and Sentinel-3 OLCI data, Gulf of Trieste." DOI: 10.3390/w13141903. ✅ | R=0.56, RMSE=0.4 mg/m³; concluye utilidad limitada de OLCI en **costas de baja productividad/turbidez moderada** — análogo a Tampa Bay/Fonseca. |
| **Salls, W.B. et al. (2024).** "Expanding the Application of Sentinel-2 Chlorophyll Monitoring across United States Lakes." *Remote Sensing*, 16(11), 1977. DOI: 10.3390/rs16111977. ✅ | Valida NDCI/MCI en **103 lagos de EEUU** (incluye lagos pequeños); confirma que S2 cubre ~99% de los cuerpos de agua de la base NHDPlus, pero **no discrimina cianobacterias** (solo biomasa general) — exactamente la limitación señalada por la asesora. |

---

## 2. Estado del arte: sistemas similares de pronóstico de HABs

> Resultados propios para comparar: skill de regresión (RMSE log-clorofila vs. persistencia) lagos +0.09 a +0.24 (significativo en 1, 5, 7 d), costa +0.23 a +0.33 (significativo en 1, 3, 5 d); alerta binaria recall operativo 0.67–1.00. Validación anidada con bloque de test final nunca tocado.

| Referencia | Resultado reportado | Comparación |
|---|---|---|
| **Gupta, A., Hantush, M.M. & Govindaraju, R.S. (2023).** "Sub-monthly time scale forecasting of HAB intensity in Lake Erie using remote sensing and ML." *Science of The Total Environment*, 900, 165781. DOI: 10.1016/j.scitotenv.2023.165781. ✅ | Horizontes 10/20/30 d; RF R²(LOOCV) 0.61→0.55. Lag-1 solo da R²=0.49 vs. 0.55 con todos los predictores a 10 d. | Mejora marginal sobre persistencia, del mismo orden de magnitud que el skill modesto reportado aquí — confirma que ganancias pequeñas y honestas son la norma en la literatura seria, no la excepción. |
| **Schaeffer, B.A. et al. (2023).** "Forecasting freshwater cyanobacterial HABs for Sentinel-3 satellite resolved U.S. lakes and reservoirs." *J. Environmental Management*. DOI: 10.1016/j.jenvman.2023.119518. ✅ | Modelo bayesiano INLA, horizonte 7 d: AUC=0.95, recall=0.88, supera a SVC/RF/DNN/LSTM/RNN (accuracy 0.84-0.85). | Recall=0.88 es del mismo orden que el recall operativo 0.67–1.00 de este trabajo. **Dato clave**: su AUC=0.95 (modelo serio) contrasta con el AUC=0.98–1.00 del sistema legado de esta tesis (diagnosticado con fuga) — evidencia externa de que un AUC casi perfecto en este dominio es sospechoso, no un éxito. |
| **Song, Y. (2025).** "Forecasting short-term chlorophyll a in Lake Erie using XGBoost." *Environmental Research Letters*, 20(6). DOI: 10.1088/1748-9326/add6b7. ✅ | R²=0.99 (1 d) a 0.90 (7 d), con datos in-situ semanales **interpolados a diario** antes de partir train/test. | **Usar como contraejemplo metodológico, no como benchmark**: interpolar antes de partir train/test puede inflar el desempeño por autocorrelación de corto plazo — es justo el tipo de circularidad que esta tesis evitó deliberadamente con su validación anidada. |
| **Molares-Ulloa, A. et al. (2024).** "Harmful algal bloom forecasting. A comparison between stream and batch learning." arXiv:2402.13304. ✅ | A 3 d: mejor modelo R²=0.77, RF R²=0.67; sin comparación contra persistencia ni discusión de fuga. | Establece un techo informal de fiabilidad operativa ~3 d en la literatura comparable; no resuelve el problema de fuga que sí resuelve esta tesis. |
| **Vermont/Wshah et al. (2025/2026).** "Transformers Model for CyanoHAB Intensity in Lake Champlain." arXiv:2512.06598; IEEE JSTARS (2026). ✅ (bibliografía confirmada) | Transformer+BiLSTM, horizonte hasta 14 d, solo con sensores remotos. | Horizonte máximo mayor (14 d vs. 7 d aquí), pero no reportan skill vs. persistencia ni discuten fuga — útil para mencionar como "trabajo futuro posible" (extender horizonte), no como benchmark de rigor. |
| Comparaciones XGBoost/RF vs. LSTM/DNN en Río Fuchun, Lago Taihu y otros (ScienceDirect, varios). ✅ (bibliografía) | Sin ganador universal: XGBoost gana en tramos tipo-embalse, LSTM en tramos de río natural; RF gana a XGBoost en Taihu. | Confirma que **no hay superioridad consistente de deep learning** sobre árboles para clorofila/cianobacterias — alineado con la decisión de usar XGBoost como modelo principal y reservar la red neuronal para la capa de alerta (ensamble). |

### Hallazgo transversal más importante de esta sección

**Stock, A., Gregr, E.J. & Chan, K.M.A. (2023).** "Data leakage jeopardizes ecological applications of machine learning." *Nature Ecology & Evolution*, 7, 1743–1745. DOI: 10.1038/s41559-023-02162-1. ✅

Este es la referencia ancla del argumento metodológico central de la tesis: documenta que la fuga de datos es un problema reconocido y extendido en aplicaciones ecológicas de ML (publicado en una revista de alto impacto), y propone declarar explícitamente el esquema de partición de datos. Úsese para enmarcar el diagnóstico de fuga del sistema legado (AUC=0.98–1.00 espurio) como un caso concreto de un problema ya documentado en la literatura — no como una anécdota aislada.

Complementaria: **Albelali, S. & Ahmed, M. (2024/2025).** "Hidden Leaks in Time Series Forecasting." arXiv:2512.06932. ✅ (bibliografía) — cuantifica que construir secuencias antes de particionar train/test infla el RMSE reportado hasta 20%, mecanismo análogo al de este caso.

**Nota honesta:** no se encontró un paper de HABs que diagnostique exactamente el mismo patrón de circularidad (target derivado de las mismas bandas que los predictores). Esto es bueno para la tesis: enmárquese el diagnóstico de fuga como una contribución metodológica propia dentro del subcampo, respaldada por el problema general ya documentado por Stock et al. (2023).

---

## 3. Respaldo biológico/limnológico

### 3.1 Clorofila-a ≠ floración nociva confirmada

| Referencia | Aporta |
|---|---|
| **US EPA — Cyanobacterial HABs Forecasting Research (CyAN).** epa.gov/water-research/cyanobacterial-harmful-algal-blooms-forecasting-research ✅ | Distingue clorofila-a (biomasa general) de ficocianina (pigmento específico de cianobacterias); confirmar nocividad requiere además análisis de toxinas (microcistinas). |
| **Graham, J.L. et al. (USGS/EPA, 2016).** "Associations between chlorophyll a and various microcystin health advisory concentrations." PMC4830210. ✅ | Muestra estadísticamente que la clorofila-a es un predictor **débil/probabilístico** de toxicidad, no una medida directa — respalda el reencuadre "riesgo de biomasa" en vez de "floración nociva confirmada". |
| USGS — "Challenges for mapping cyanotoxin patterns from remote sensing of cyanobacteria." pubs.usgs.gov/publication/70170957 ✅ | Las cianotoxinas no son detectables directamente por sensores remotos; solo hay proxies indirectos imperfectos (clorofila, ficocianina). |

### 3.2 Fósforo como nutriente limitante en agua dulce

| Referencia | Aporta |
|---|---|
| **Schindler, D.W. (1974).** "Eutrophication and Recovery in Experimental Lakes." *Science*, 184(4139), 897–899. DOI: 10.1126/science.184.4139.897. ✅ | Experimento de lago completo (Experimental Lakes Area): el fósforo, no el N ni el C, controla la eutrofización en agua dulce. Referencia clásica y fundacional. |
| **Schindler, D.W. et al. (2008).** "Eutrophication of lakes cannot be controlled by reducing nitrogen input." *PNAS*, 105(32), 11254–11258. DOI: 10.1073/pnas.0805108105. ✅ | Experimento de 37 años (Lake 227): reducir solo el N favorece cianobacterias fijadoras de N₂, que compensan el déficit — confirma el fósforo como control a largo plazo. |
| **Vollenweider, R.A. & Kerekes, J. (1982).** "Eutrophication of Waters." OECD, Paris. ⚠️ (esquema confirmado; tabla numérica exacta de cortes de clorofila-a sin verificar) | Esquema de clasificación trófica OECD (oligo/meso/eutrófico) basado en fósforo total y clorofila-a; verificar cifras exactas antes de citarlas con números. |

### 3.3 Limitación de Sentinel-2 para distinguir cianobacterias (sin banda de ficocianina ~620 nm)

| Referencia | Aporta |
|---|---|
| **Simis, S.G.H., Peters, S.W.M. & Gons, H.J. (2005).** "Remote sensing of the cyanobacterial pigment phycocyanin in turbid inland water." *Limnology and Oceanography*, 50(1), 237–245. DOI: 10.4319/lo.2005.50.1.0237. ✅ | Algoritmo R709/R620 para ficocianina (compatible con MERIS/OLCI). Establece por qué se necesita una banda ~620 nm —que Sentinel-2 MSI no tiene— para aislar la señal de cianobacterias. |
| Page, B.P., Olmanson, L.G. & Mishra, D.R. — evaluación de algoritmos MERIS/OLCI para ficocianina en el este de EEUU. *Remote Sensing of Environment*. ✅ (bibliografía) | Confirma que la detección de cianobacterias depende de bandas 620+709 nm presentes en MERIS/OLCI, ausentes en S2 MSI. |
| "Phycocyanin Monitoring in Some Spanish Water Bodies with Sentinel-2 Imagery." *Water*, 13(20), 2866, MDPI (2021). ⚠️ (autores sin confirmar) | El hecho de que se necesiten aproximaciones indirectas con bandas estándar de S2 (sin banda nativa de ficocianina) es evidencia en sí misma de la limitación. |

### 3.4 Umbrales de referencia (OMS/EPA) para riesgo de cianobacterias

| Referencia | Aporta |
|---|---|
| **WHO (1999), resumido por EPA:** "WHO 1999 Guideline Values for Cyanobacteria in Freshwater." epa.gov/habs/world-health-organization-who-1999-guideline-values-cyanobacteria-freshwater ✅ | 3 niveles de alerta: Nivel 1 ~20 000 cél/mL o clorofila-a 10 µg/L; Nivel 2 ~100 000 cél/mL o 50 µg/L; Nivel 3 = natas visibles. Marco internacional para justificar niveles escalonados de alerta. |
| **US EPA (2019).** "Recommendations for Public Water Systems to Manage Cyanotoxins in Drinking Water." EPA 823-R-19-001. ✅ | Valores de referencia regulatorios en EEUU, análogos al esquema OMS. |

**Aclaración importante a incluir en la tesis:** el umbral propio (percentil 85 de la climatología local de cada cuerpo de agua) es una métrica de **anomalía relativa**, no un umbral de riesgo sanitario absoluto como los de OMS/EPA. Son lógicas complementarias — cítese OMS/EPA como precedente de "alerta escalonada", aclarando que no son intercambiables con el umbral propio.

---

## 4. Validación externa más allá de la literatura

### 4.1 NOAA CyAN (Cyanobacteria Assessment Network) — la vía más fuerte para Okeechobee

- Programa EPA+NASA+NOAA+USGS, **Sentinel-3 OLCI**, >2 000 lagos de EEUU, 300 m, boletines semanales. **Cobertura de Lake Okeechobee confirmada** (NOAA NCCOS documenta un despliegue dedicado para Okeechobee).
- Acceso: visor web (CyAN App, sin programar) — epa.gov/water-research/cyanobacteria-assessment-network-application-cyan-app; API REST en `cyan.epa.gov/cyan/cyano/` (sin autenticación documentada); paquete R `USGS-R/CyAN` en GitHub.
- **Por qué es fuerte como validación:** usa Sentinel-3 OLCI con un algoritmo y pipeline completamente independientes del propio (no hay circularidad con WQP).
- **Acción concreta:** instalar `USGS-R/CyAN` (o usar la API REST), descargar la serie de CI para el polígono de Okeechobee 2023–01–01 a 2026–06–28, y comparar fechas/intensidad de picos contra las alertas propias. Reportar si el sistema anticipa el pico con 0–7 días de antelación.

### 4.2 Copernicus Marine Service — para Tampa Bay y Golfo de Fonseca

- Copernicus Marine Data Store (data.marine.copernicus.eu): productos globales de clorofila-a "ocean colour" (4 km global, hasta 300 m regional), registro gratuito.
- **Negativo a documentar:** "Copernicus Land — Lake Water Quality v2.0" (100 m, vía S2) parecía ideal pero **solo cubre Europa/África** — no aplica a ninguno de los 5 cuerpos de esta tesis.
- **Acción concreta:** registrarse gratis; descargar `OCEANCOLOUR_GLO_BGC_L3_NRT_009_101` (o equivalente regional del Golfo de México) para Tampa Bay 2023–2026; para Fonseca, verificar si la resolución 4 km resuelve la geometría del golfo (cuerpo semicerrado, riesgo de contaminación de píxel costero — documentar como limitación si aplica).

### 4.3 Monitoreo institucional en Honduras (hallazgo nuevo, no explorado antes)

- **LimnoYojoa** (limnoyojoa.com): alianza UNAH + Universidad de Sevilla + AMUPROLAGO, monitoreo activo 2024–2026 en 12 puntos del Lago de Yojoa (fisicoquímica, sedimentos, Secchi, color de agua). **Esto llena exactamente el vacío 2023–2026 que dejó el dataset de Zenodo.** Acción: contactar vía el formulario del sitio o fabioesmar@gmail.com pidiendo acceso a clorofila-a/Secchi/nutrientes 2024–2026 para uso de tesis.
- **AMUPROLAGO** (amuprolago.org): gestora del lago, muestreo quincenal reportado. Acción: solicitar boletines de monitoreo 2023–2026, aunque sea agregados.
- **El Cajón:** sin programa de monitoreo de calidad de agua público identificado (lo que existe es manejo de caudal/seguridad de presa por ENEE, no calidad de agua). **No inventar una fuente aquí.** Acción alternativa: contactar a ENEE para preguntar por monitoreo interno no publicado; si no hay respuesta, declarar en la tesis que El Cajón se valida solo indirectamente (misma cuenca del río Ulúa, aguas arriba de Yojoa) — limitación honesta, no vacío oculto.
- **Tampa Bay Estuary Program (TBEP)** (tbep.org): publica un "Water Quality Report Card" anual y dashboard, producto de síntesis independiente del propio pipeline (aunque la fuente primaria sea la misma WQP). Acción: comparar segmentos marcados en rojo/naranja del report card 2023–2025 contra las alertas propias en Tampa Bay.

### 4.4 Estrategias metodológicas para ground-truth escaso (marco teórico para justificar todo lo anterior)

- **Validación cruzada entre satélites independientes** ("cross-satellite assessment") — exactamente lo que propone 4.1/4.2.
- **Validación por proxy** (relación física conocida con una variable relacionada) — ya aplicado en el proyecto (VIIRS vs. Secchi en Yojoa, correlación negativa esperada y confirmada).
- **Modelos geoestadísticos espacio-temporales** que combinan monitoreos discontinuos de múltiples fuentes.
- **Ciencia participativa/ciudadana**: "Participatory science methods to ground truth remote sensing of the Chesapeake Bay" (PMC11527148) — modelo replicable a pequeña escala con voluntarios (Secchi + fotos) en Yojoa, coordinable con LimnoYojoa/AMUPROLAGO si hay tiempo.
- **Elicitación estructurada de expertos** (método SHELF/JRC) cuando no hay datos cuantitativos suficientes — opción para El Cajón si no se consigue monitoreo institucional.

---

## 5. Resumen de próximos pasos concretos (por prioridad/esfuerzo)

| Acción | Esfuerzo | Qué aporta |
|---|---|---|
| Descargar CI de CyAN para Okeechobee 2023–2026 y comparar con alertas propias | Bajo (API/paquete R, sin permiso de terceros) | Validación externa independiente más fuerte disponible |
| Contactar LimnoYojoa/AMUPROLAGO pidiendo datos 2024–2026 | Bajo-medio (un correo) | Cierra el hueco de ground-truth de Yojoa 2023–2026 |
| Descargar Copernicus Marine para Tampa Bay/Fonseca | Bajo (registro gratuito) | Segunda fuente de clorofila independiente en costa |
| Comparar contra TBEP Water Quality Report Card | Bajo | Verificación cualitativa adicional en Tampa Bay |
| Verificar las citas marcadas ⚠️ antes de incluirlas en el documento final | Bajo | Evita riesgo de cita incorrecta en la defensa |
| Contactar ENEE por monitoreo interno de El Cajón (o declarar limitación) | Bajo | Cierra honestamente el caso más débil de validación |

---

## 6. Notas de honestidad académica

- Cada referencia fue verificada por al menos una fuente independiente (DOI oficial, sitio institucional, o registro cruzado en 2+ buscadores). Las marcadas ⚠️ tienen el registro bibliográfico (título/revista/año) razonablemente confiable pero algún detalle (autoría exacta o una cifra puntual) sin confirmar — **revisar la fuente primaria antes de citarlas en el documento final de tesis.**
- No se incluyó ninguna referencia que los agentes de investigación no pudieran respaldar con al menos un indicio verificable; donde la evidencia era débil, se dijo explícitamente en vez de rellenar el hueco.
- El hallazgo más valioso para la defensa es indirecto: **Stock et al. (2023, Nature Ecology & Evolution)** y el contraste con **Song (2025)** y **Schaeffer et al. (2023)** permiten argumentar que los resultados modestos pero honestos de esta tesis son *más* defendibles que reportar números altísimos — la literatura reciente muestra que la fuga de datos es un problema reconocido en el campo, y que sistemas serios (INLA, AUC=0.95) obtienen desempeños del mismo orden que el propio, no cercanos a 1.0.
