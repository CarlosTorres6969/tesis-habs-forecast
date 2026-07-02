# Validación externa de las detecciones del modelo de HABs (2023–2026)

> Documento de apoyo para contrastar las **fechas de floración / biomasa elevada que detecta el
> modelo** con **evidencia independiente** (artículos, informes, noticias, comunicados oficiales,
> monitoreos satelitales). Las tablas vienen **pre-cargadas con las fechas del modelo**; las
> columnas de evidencia se completan con la investigación documental.
>
> Generado: 2026-06-29. Fuente de las fechas: `artifacts/targets/combined_target.csv` (serie target
> del modelo, ventana 2023–2026).

---

## 0. Qué significa "fecha detectada por el modelo" (leer antes de validar)

Es importante ser honesto sobre **qué** estamos validando, para no afirmar coincidencias que no
correspondan:

1. **El modelo pronostica clorofila-a (proxy de BIOMASA algal), no toxicidad.** Una detección indica
   biomasa elevada / floración probable, **no confirma una floración NOCIVA** (cianotoxinas). La
   evidencia externa de "cambio de color del agua", "mancha verde", "exceso de clorofila" o
   "eutrofización" respalda la señal de biomasa; la evidencia de "toxinas", "mortandad de peces" o
   "alerta sanitaria" sería un respaldo más fuerte pero es un fenómeno distinto.

2. **Las fechas de este documento son la serie OBJETIVO (target satelital)**, es decir, los días en
   que la clorofila estimada por satélite superó el **umbral de alerta del cuerpo**. Es el fenómeno
   que el modelo aprende a anticipar a 0–7 días. (No son las "alertas operativas" de `run_forecast`,
   que solo existen para 2026; se eligió la serie 2023–2026 completa por decisión del usuario.)

3. **Umbral de alerta = relativo por ecosistema.** Se define como el **percentil 85 (p85)** de la
   climatología de cada cuerpo, **acotado a 24 µg/L** (nivel de alerta sanitaria recreativa). Así un
   lago hipereutrófico no exige niveles absurdos y un agua costera oligotrófica mantiene sensibilidad.
   Ver `config.py` (`USE_RELATIVE_THRESHOLD`, `RELATIVE_PERCENTILE=85`, `alert_threshold_ugl`).

4. **El estudio cubre 5 cuerpos de agua, no 5 puntos dentro de Yojoa.** Dos son anclas de validación
   en Florida (Okeechobee y Tampa Bay, con abundante literatura/monitoreo) y tres son los cuerpos de
   interés en Honduras (Yojoa, El Cajón, Golfo de Fonseca). El modelo pronostica a nivel de **cuerpo
   de agua completo** (promedio espacial), no por estación individual.

5. **Agrupación en episodios.** Las cientos de fechas individuales se consolidaron en **episodios de
   floración** (rachas de detecciones separadas por > 21 días sin alerta). Para cada episodio se da el
   rango, la fecha del pico, la clorofila máxima y el nº de detecciones. La validación documental se
   hace por episodio (buscar evidencia dentro de ±15 días del rango/pico).

---

## 1. Resumen de los 5 cuerpos

| Cuerpo | Ubicación | Grupo | Centroide aprox. | Umbral alerta (µg/L) | Nº detecciones 2023–26 | Nº episodios |
|--------|-----------|-------|------------------|----------------------|------------------------|--------------|
| Lago Okeechobee | Florida, EE.UU. | Dulce | 26.95 N, 80.85 O | 24.0 (p85=42.8) | 206 | 6 |
| Bahía de Tampa | Florida, EE.UU. | Marino | 27.73 N, 82.58 O | 6.4 (p85=6.4) | 175 | 14 |
| **Lago de Yojoa** | **Honduras** | Dulce | 14.87 N, 87.96 O | 24.0 (p85=34.3) | 74 | 13 |
| **Embalse El Cajón** | **Honduras** | Dulce | 14.83 N, 87.69 O | 24.0 (p85=63.8) | 72 | 4 |
| **Golfo de Fonseca** | **Honduras** | Marino | 13.18 N, 87.60 O | 5.7 (p85=5.7) | 165 | 12 |

> Nivel de confianza del propio modelo (de `ESTADO_PROYECTO.md` / `REPORTE_DEFENSA.md`):
> **ALTA** = Okeechobee (validado con clorofila in-situ) y costa (target OLCI validado);
> **validación interna robusta sin verdad de campo 2023–26** = Yojoa (target VIIRS validado fuera de
> ventana contra Secchi 2018–22, correlación significativa);
> **exploratorio** = El Cajón.

---

## 2. Tablas de validación por cuerpo (fechas del modelo pre-cargadas)

Leyenda de **Coincidencia**: `Directa` (evento documentado dentro de ±15 d) · `Parcial` (evidencia
cercana o indirecta) · `Sin evidencia` (no se encontró) · `Contradice` (la fuente indica ausencia de
floración).

### 2.1 Lago Okeechobee — Florida, EE.UU. (ancla de validación)

| # | Episodio (rango) | Pico | Chl-a pico (µg/L) | Nº det. | Evidencia encontrada | Fecha evidencia | Coincidencia | Resumen | Fuente | Confiabilidad |
|---|------------------|------|-------------------|---------|----------------------|-----------------|--------------|---------|--------|---------------|
| 1 | 2023-10-31 | 2023-10-31 | 29.9 | 1 | Floración mayor de cianobacterias documentada por NASA: Landsat-9 captó el 12-jun-2023 ~380 mi² (>½ del lago); el lago tiene floraciones anuales recurrentes | 2023 (jun) | Parcial | Episodio puntual fuera de la cobertura mediática; el lago estaba en estado de floración crónica durante 2023 | NASA Earth Observatory; NOAA NCCOS | ALTA (NASA/NOAA) |
| 2 | 2023-12-23 | 2023-12-23 | 66.9 | 1 | — | | Sin evidencia | No se halló reporte fechado en dic-2023 | — | — |
| 3 | 2024-01-30 → 2024-10-05 | 2024-03-08 | 101.7 | 75 | FDEP: reporte de monitoreo 22–28 mar-2024 con microcistina por encima del nivel EPA (17 ppb, St. Lucie); las HAB comenzaron en primavera 2024 en el lago y estuarios | 2024 (mar+) | **Directa** | El gran episodio del modelo (pico 8-mar) coincide con el inicio documentado de la floración tóxica de 2024 | FDEP weekly updates; petición EPA (Center for Biological Diversity); CBS12 | ALTA (gobierno) |
| 4 | 2024-12-14 | 2024-12-14 | 40.4 | 1 | — | | Sin evidencia | — | — | — |
| 5 | 2025-02-05 → 2025-02-11 | 2025-02-11 | 36.5 | 4 | — | | Sin evidencia | — | — | — |
| 6 | 2025-05-01 → 2026-04-10 | 2025-08-13 | 101.7 | 124 | Floraciones anuales de *Microcystis aeruginosa* (tóxica) en verano, confirmadas por estudios 2025 (FAU Harbor Branch + USF; Frontiers in Water, "annual Microcystis blooms") | 2025 (verano) | Parcial | Recurrencia anual de floración estival documentada; respalda el pico de ago-2025 aunque sin fecha exacta única | Frontiers in Water (2025); FAU Newsdesk; ScienceDaily | ALTA (científica) |

### 2.2 Bahía de Tampa — Florida, EE.UU. (ancla de validación)

| # | Episodio (rango) | Pico | Chl-a pico (µg/L) | Nº det. | Evidencia encontrada | Fecha evidencia | Coincidencia | Resumen | Fuente | Confiabilidad |
|---|------------------|------|-------------------|---------|----------------------|-----------------|--------------|---------|--------|---------------|
| 1 | 2023-01-07 → 2023-01-26 | 2023-01-23 | 25.2 | 3 | Marea roja (*Karenia brevis*) activa en Tampa Bay/Pinellas desde dic-2022; concentraciones altas reportadas en ene-2023 | 2023-01 (ene) | **Directa** | El episodio de ene-2023 del modelo coincide con la marea roja documentada por FWC y prensa | FWC Red Tide Status; Tampa Bay Times (18-ene-2023); Axios | ALTA (gobierno/medios) |
| 2 | 2023-02-19 | 2023-02-19 | 9.1 | 1 | Continuación de la marea roja de invierno 2023 (alto riesgo en playas de Pinellas hasta marzo) | 2023-02/03 | Parcial | Mismo evento prolongado de inicios de 2023 | FWC; WTSP | ALTA |
| 3 | 2023-03-29 → 2023-04-09 | 2023-04-09 | 15.7 | 2 | *Karenia brevis* en alta concentración dentro y frente a Pinellas en abr-2023 | 2023-04 | **Directa** | Coincide con la persistencia de la marea roja en primavera 2023 | FWC; WMNF | ALTA |
| 4 | 2023-07-24 | 2023-07-24 | 8.8 | 1 | — | | Sin evidencia | — | — | — |
| 5 | 2023-09-19 → 2023-10-12 | 2023-10-12 | 64.7 | 5 | — | | Sin evidencia | No se halló reporte fechado de otoño 2023 | — | — |
| 6 | 2023-11-16 → 2024-01-29 | 2023-11-16 | 60.2 | 11 | — | | Sin evidencia | — | — | — |
| 7 | 2024-03-18 → 2024-03-28 | 2024-03-18 | 7.9 | 2 | — | | Sin evidencia | — | — | — |
| 8 | 2024-07-09 | 2024-07-09 | 13.6 | 1 | — | | Sin evidencia | — | — | — |
| 9 | 2024-08-06 → 2025-03-18 | 2025-01-17 | 45.0 | 133 | Marea roja inusual de invierno tras el huracán Milton (oct-2024); se mantuvo y el 2-feb-2025 se extendió de Tampa Bay a Key West; concentraciones de millones de cél/L | 2024-10 → 2025-02 | **Directa** | El mayor episodio del modelo (pico 17-ene-2025) coincide plenamente con la marea roja de invierno 2024–25 | FWC; WUSF (2-feb-2025); WMNF | ALTA (gobierno/medios) |
| 10 | 2025-06-29 | 2025-06-29 | 6.8 | 1 | — | | Sin evidencia | — | — | — |
| 11 | 2025-08-10 → 2025-10-26 | 2025-08-10 | 31.3 | 10 | Costo creciente de las mareas rojas recurrentes en Florida reportado en sep-2025 (contexto, no evento puntual confirmado para Tampa) | 2025-09 | Parcial | Temporada típica de fin de verano/otoño; sin confirmación de fecha exacta | The Invading Sea (15-sep-2025) | MEDIA (medio) |
| 12 | 2025-12-06 → 2025-12-18 | 2025-12-06 | 9.5 | 2 | — | | Sin evidencia | — | — | — |
| 13 | 2026-01-18 → 2026-01-26 | 2026-01-26 | 24.3 | 2 | — | | Sin evidencia | Requiere consulta a FWC (2026 reciente) | — | — |
| 14 | 2026-02-22 | 2026-02-22 | 39.7 | 1 | — | | Sin evidencia | — | — | — |

### 2.3 Lago de Yojoa — Honduras ⭐ (cuerpo de interés principal)

| # | Episodio (rango) | Pico | Chl-a pico (µg/L) | Nº det. | Evidencia encontrada | Fecha evidencia | Coincidencia | Resumen | Fuente | Confiabilidad |
|---|------------------|------|-------------------|---------|----------------------|-----------------|--------------|---------|--------|---------------|
| 1 | 2023-02-14 | 2023-02-14 | 24.6 | 1 | "Intervenido Lago de Yojoa para evitar su muerte en diez años" — degradación por exceso de algas y pérdida de transparencia | 2023-02-20 | Parcial | Reporte de degradación/algas contemporáneo (±6 d); no es una medición de floración pero confirma el estado del lago | La Tribuna (20-feb-2023) | MEDIA (medio) |
| 2 | 2023-09-26 | 2023-09-26 | 34.3 | 1 | — | | Sin evidencia | — | — | — |
| 3 | 2023-10-27 | 2023-10-27 | 54.8 | 1 | — | | Sin evidencia | — | — | — |
| 4 | 2023-12-25 → 2024-05-04 | 2024-01-18 | 95.9 | 42 | Contexto 2023–24: superficie "llena de algas", pérdida de 2–3 m de claridad por mes por sobreproducción de algas; suspensión de licencias de acuicultura (30-may-2023) | 2023 (may–jun) | Parcial | El gran episodio de época seca (dic–abr) coincide con el periodo de crisis de algas/eutrofización ampliamente reportado; sin medición de fecha exacta | La Prensa; Contra Corriente (30-jun-2023); Infobae | MEDIA-ALTA (medios) |
| 5 | 2024-06-02 → 2024-06-04 | 2024-06-02 | 60.7 | 2 | — | | Sin evidencia | — | — | — |
| 6 | 2024-08-05 → 2024-10-07 | 2024-08-25 | 75.4 | 14 | — | | Sin evidencia | Antecede en ~1 mes a la mortandad de peces de nov-2024 (ver E7) | — | — |
| 7 | 2024-10-31 | 2024-10-31 | 28.8 | 1 | Mortandad masiva de peces (~500) en el canal del Lago de Yojoa; La Prensa: "Falta de oxígeno ocasionó muerte de peces" — alta densidad de algas y bajo oxígeno | 2024-11-03/05 | **Directa** | El evento de hipoxia/mortandad (±4 d) es consistente con biomasa algal elevada detectada por el modelo a fin de oct-2024 | La Prensa; Tiempo; Proceso Digital (nov-2024) | ALTA (medios, múltiples) |
| 8 | 2024-12-28 | 2024-12-28 | 31.3 | 1 | — | | Sin evidencia | — | — | — |
| 9 | 2025-02-03 → 2025-03-06 | 2025-03-01 | 45.6 | 4 | — | | Sin evidencia | — | — | — |
| 10 | 2025-06-02 → 2025-06-07 | 2025-06-07 | 37.2 | 2 | — | | Sin evidencia | — | — | — |
| 11 | 2025-07-09 → 2025-07-25 | 2025-07-09 | 27.0 | 2 | — | | Sin evidencia | — | — | — |
| 12 | 2025-08-21 → 2025-09-06 | 2025-08-21 | 26.5 | 2 | — | | Sin evidencia | — | — | — |
| 13 | 2025-11-09 | 2025-11-09 | 66.5 | 1 | — | | Sin evidencia | — | — | — |
| — | Respaldo científico (especies) | — | — | — | Estudio UNAH/RCT 2014–2015: cianobacterias dominantes en Yojoa (*Microcystis aeruginosa*, *Chroococcus*, *Lyngbya*) con recomendación de plan de monitoreo de toxinas | 2014–2015 | Contexto | Confirma que en Yojoa existen las especies formadoras de floración nociva que el modelo busca anticipar | Rev. Ciencia y Tecnología (CAMJOL) | ALTA (científica) |

### 2.4 Embalse El Cajón — Honduras (exploratorio)

| # | Episodio (rango) | Pico | Chl-a pico (µg/L) | Nº det. | Evidencia encontrada | Fecha evidencia | Coincidencia | Resumen | Fuente | Confiabilidad |
|---|------------------|------|-------------------|---------|----------------------|-----------------|--------------|---------|--------|---------------|
| 1 | 2024-02-11 → 2024-03-04 | 2024-03-04 | 57.3 | 3 | — | | Sin evidencia | El Cajón **no tiene programa de monitoreo** público de calidad de agua/HAB | — | — |
| 2 | 2024-09-07 | 2024-09-07 | 85.4 | 1 | — | | Sin evidencia | Ídem | — | — |
| 3 | 2025-01-29 | 2025-01-29 | 26.1 | 1 | — | | Sin evidencia | Ídem | — | — |
| 4 | 2025-06-26 → 2026-04-27 | 2025-06-26 | 97.3 | 67 | — | | Sin evidencia | Ídem; episodio mayor sostenido sin verdad de campo disponible | — | — |

### 2.5 Golfo de Fonseca — Honduras

| # | Episodio (rango) | Pico | Chl-a pico (µg/L) | Nº det. | Evidencia encontrada | Fecha evidencia | Coincidencia | Resumen | Fuente | Confiabilidad |
|---|------------------|------|-------------------|---------|----------------------|-----------------|--------------|---------|--------|---------------|
| 1 | 2023-01-09 → 2023-04-17 | 2023-03-12 | 15.2 | 20 | LABTOX-UES (Universidad de El Salvador): muestreo del **14-mar-2023** descartó marea roja **tóxica**, pero halló **diatomeas abundantes** (*Actinocyclus* 26 320 cél/L; *Nitzschia* 12 500 cél/L) y trazas de *Gymnodinium catenatum*/*Alexandrium* | 2023-03-14/18 | **Parcial (clave)** | Coincidencia casi exacta con el pico del modelo (12-mar). Confirma **biomasa de fitoplancton elevada** justo cuando el modelo la detecta, y a la vez **descarta toxicidad** → valida el encuadre "biomasa ≠ floración nociva" | LABTOX-UES / El Universitario; DIGEPESCA (SICA, 18-mar-2023) | ALTA (universidad/gobierno) |
| 2 | 2023-05-20 → 2023-08-05 | 2023-06-26 | 31.3 | 10 | — | | Sin evidencia | — | — | — |
| 3 | 2023-09-01 → 2023-09-03 | 2023-09-03 | 6.6 | 3 | — | | Sin evidencia | — | — | — |
| 4 | 2023-10-02 → 2023-11-01 | 2023-10-29 | 20.7 | 5 | — | | Sin evidencia | — | — | — |
| 5 | 2023-12-18 | 2023-12-18 | 5.7 | 1 | — | | Sin evidencia | — | — | — |
| 6 | 2024-01-14 → 2024-04-15 | 2024-02-21 | 12.2 | 19 | — | | Sin evidencia | — | — | — |
| 7 | 2024-05-10 → 2024-07-11 | 2024-05-10 | 50.2 | 15 | — | | Sin evidencia | Pico alto (50 µg/L); sin monitoreo costero fechado disponible | — | — |
| 8 | 2024-09-07 → 2024-10-27 | 2024-10-21 | 14.6 | 10 | — | | Sin evidencia | — | — | — |
| 9 | 2024-11-18 → 2025-02-03 | 2024-11-19 | 58.0 | 14 | — | | Sin evidencia | Pico alto (58 µg/L) sin verdad de campo | — | — |
| 10 | 2025-03-02 → 2025-10-22 | 2025-05-08 | 28.1 | 51 | — | | Sin evidencia | — | — | — |
| 11 | 2026-01-05 → 2026-02-19 | 2026-01-23 | 19.6 | 14 | — | | Sin evidencia | — | — | — |
| 12 | 2026-06-04 → 2026-06-17 | 2026-06-08 | 6.8 | 3 | — | | Sin evidencia | — | — | — |

---

## 2bis. Síntesis de la validación documental (realizada el 2026-06-29)

Coincidencias **confirmadas con evidencia fechada** (las más defendibles para la tesis):

| Cuerpo | Episodio del modelo | Evidencia independiente | Δ días | Coincidencia |
|--------|---------------------|-------------------------|--------|--------------|
| **Lago de Yojoa** | E7 — 2024-10-31 (28.8 µg/L) | Mortandad masiva de peces por **falta de oxígeno + alta densidad de algas**, 3–5 nov-2024 (La Prensa, Tiempo, Proceso) | ~4 | **Directa** |
| **Golfo de Fonseca** | E1 — pico 2023-03-12 (15.2 µg/L) | LABTOX-UES, muestreo 14-mar-2023: **diatomeas abundantes** (biomasa alta), descarta marea roja tóxica | ~2 | **Parcial (clave)** |
| **Lago Okeechobee** | E3 — pico 2024-03-08 (101.7 µg/L) | FDEP: floración con **microcistina > umbral EPA**, 22–28 mar-2024 | ~14–20 | **Directa** |
| **Bahía de Tampa** | E9 — pico 2025-01-17 (45.0 µg/L) | **Marea roja de invierno** post-huracán Milton, oct-2024 → feb-2025 (FWC, WUSF) | dentro del rango | **Directa** |
| **Bahía de Tampa** | E1 — 2023-01 (25.2 µg/L) | Marea roja *K. brevis* en Pinellas, dic-2022–ene-2023 (FWC, prensa) | dentro del rango | **Directa** |
| **Lago de Yojoa** | E1 — 2023-02-14 (24.6 µg/L) | "Intervenido Lago de Yojoa…" por exceso de algas, 20-feb-2023 (La Tribuna) | ~6 | Parcial |
| **Lago Okeechobee** | E6 — pico 2025-08-13 | Floración anual de *Microcystis* verano 2025 (FAU/USF; Frontiers) | recurrencia | Parcial |

**Lecturas para la defensa:**

1. **El método de validación funciona donde hay monitoreo.** En los cuerpos con vigilancia operativa
   (Okeechobee, Tampa Bay) los episodios mayores del modelo coinciden de forma **directa** con
   floraciones tóxicas / mareas rojas documentadas por agencias (FDEP, FWC) y NASA/NOAA. Esto es la
   prueba de concepto: cuando existe verdad de campo, las detecciones del modelo aciertan.

2. **El hallazgo más valioso es Fonseca E1.** El modelo marcó biomasa elevada con pico el 12-mar-2023;
   un laboratorio universitario muestreó el 14-mar-2023 (±2 días) y encontró **fitoplancton abundante
   (diatomeas) pero NO marea roja tóxica**. Es la confirmación empírica e independiente de la tesis
   central de tu trabajo: **el modelo detecta biomasa, no toxicidad.** Conviene destacarlo.

3. **Yojoa: una coincidencia directa fuerte.** La mortandad de peces por hipoxia y "alta densidad de
   algas" de inicios de nov-2024 cae a ~4 días de la detección E7 del modelo (fin de oct-2024). Es la
   mejor evidencia de campo disponible para el cuerpo de interés principal.

4. **La ausencia de evidencia en Honduras NO es ausencia del fenómeno.** Para El Cajón (sin programa
   de monitoreo) y la mayoría de episodios de Yojoa/Fonseca no hay reportes fechados publicados: es un
   vacío de *verdad de terreno*, coherente con lo ya documentado en el proyecto (no existe in-situ
   público hondureño 2023–2026). Debe declararse como limitación de validación, no como fallo del modelo.

5. **Distinción biomasa/toxicidad al redactar.** Solo Okeechobee (microcistina) y Tampa (brevetoxina)
   tienen evidencia de toxicidad. En Yojoa/Fonseca lo defendible es "biomasa algal elevada coincidente
   con [evento documentado]", no "floración nociva confirmada".

### Fuentes consultadas (enlaces)

- NASA Earth Observatory — *Algae Bloom in Lake Okeechobee* (Landsat-9, 12-jun-2023): https://science.nasa.gov/earth/earth-observatory/algae-bloom-in-lake-okeechobee-151581/
- NOAA NCCOS — *Cyanobacteria Algal Bloom from Satellite in Lake Okeechobee, FL*: https://coastalscience.noaa.gov/science-areas/habs/hab-monitoring-system/cyanobacteria-algal-bloom-satellite-lake-okeechobee-fl/
- Florida DEP — *Weekly Updates and Subscription* (reportes de HAB): https://floridadep.gov/sec/sec/content/weekly-updates-and-subscription
- Center for Biological Diversity — petición EPA sobre algas tóxicas en Florida (29-may-2024, cita FDEP mar-2024): https://www.biologicaldiversity.org/campaigns/Floridas-toxic-algae/pdfs/05-29-2024-Final-EPA-Petition.pdf
- Frontiers in Water (2025) — *Diversity fluctuations during annual Microcystis blooms in Lake Okeechobee*: https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2025.1678547/full
- FAU Newsdesk / ScienceDaily (abr-2025) — *Toxic blooms in motion, Lake Okeechobee*: https://www.sciencedaily.com/releases/2025/04/250423112028.htm
- FWC — *Red Tide Current Status*: https://myfwc.com/research/redtide/statewide/
- WUSF (2-feb-2025) — *Red tide spreads from Tampa Bay to Key West*: https://www.wusf.org/environment/2025-02-02/red-tide-spreads-along-the-southwest-florida-coast-from-tampa-bay-to-key-west
- Tampa Bay Times (18-ene-2023) — red tide en Tampa Bay: https://www.tampabay.com/news/environment/2023/01/18/red-tide-florida-2023-tampa-bay-toxic-algae-bloom/
- La Prensa (HN) — *Falta de oxígeno ocasionó muerte de peces en canal del Lago de Yojoa* (nov-2024): https://www.laprensa.hn/honduras/falta-de-oxigeno-ocasiono-muerte-de-peces-en-canal-del-lago-de-IVLP909405
- Tiempo (HN) — *Investigan muerte masiva de peces cerca del Lago de Yojoa*: https://tiempo.hn/investigan-muerte-masiva-peces-cerca-lago-yojoa/
- La Tribuna (HN, 20-feb-2023) — *Intervenido Lago de Yojoa para evitar su muerte en diez años*: https://archivos.latribuna.hn/2023/02/20/intervenido-lago-de-yojoa-para-evitar-su-muerte-en-diez-anos/
- Contra Corriente (HN, 30-jun-2023) — *Los peces que destruyen el lago* (algas/eutrofización Yojoa): https://contracorriente.red/2023/06/30/los-peces-que-destruyen-el-lago/
- Infobae (31-may-2023) — Honduras suspende licencias de acuicultura en Yojoa: https://www.infobae.com/noticias/2023/05/31/honduras-lanzo-plan-de-proteccion-para-su-mayor-lago-de-agua-dulce/
- Revista Ciencia y Tecnología (CAMJOL) — *Cianobacterias del Lago de Yojoa 2014–2015*: https://www.camjol.info/index.php/RCT/article/view/5922
- LABTOX-UES / El Universitario — *LABTOX-UES descarta marea roja en Golfo de Fonseca* (mar-2023): https://eluniversitario.ues.edu.sv/labtox-ues-descarta-marea-roja-en-golfo-de-fonseca/
- DIGEPESCA / SICA — *DIGEPESCA descarta marea roja en Golfo de Fonseca*: https://www.sica.int/busqueda/Noticias.aspx?IDItem=111289&IDCat=2&IdEnt=47

> Nota: las búsquedas se hicieron en jun-2026 con herramienta de acceso web. Conviene reabrir los
> enlaces y guardar PDF/captura con fecha para el anexo de la tesis (las URL de prensa pueden cambiar).

---

## 2ter. Figuras: mapas del modelo en las fechas validadas

Generadas con `build_validacion_maps.py` (reusa `build_map_figure`, escena Sentinel-2 más cercana y
despejada a cada evento). Carpeta: `habs_forecast/entregables/validacion/`. Cada figura tiene 2 paneles
(imagen satelital real + mapa de biomasa prevista a +3 d) y pie con el evento documentado.

Todas las escenas se eligieron por **máxima cobertura de agua limpia dentro del episodio validado**
(puntuadas por agua coherente sin nubes), para que el panel 2 muestre el **gradiente espacial** de
clorofila (azul=bajo → rojo=alto), no parches por nubes. Las figuras se renderizan en modo
`gradient_focus` (`build_map_figure(..., gradient_focus=True)`): suaviza el campo (quita ruido
sal-y-pimienta) y aligera los contornos de umbral para que el gradiente sea legible — afecta solo la
**visualización**, no los stats ni el modelo (la app/CLI siguen en modo normal).

| Figura (archivo) | Cuerpo / evento | Escena | Chl-a media | Área floración | Lectura |
|------------------|-----------------|--------|-------------|----------------|---------|
| `mapa_val_okeechobee_e3_2024-03-15.png` | Okeechobee — microcistina>EPA (FDEP) | 2024-03-15 | 38.9 µg/L | **100%** | **Fuerte.** Gradiente claro azul→rojo; coincide con evento tóxico |
| `mapa_val_okeechobee_e6_2025-08-02.png` | Okeechobee — *Microcystis* verano 2025 | 2025-08-02 | 36.3 µg/L | 98% | **Fuerte.** Lago completo, verano 2025 |
| `mapa_val_yojoa_e4_2024-03-01.png` | Yojoa — **episodio MAYOR** época seca (pico ene-2024, 96 µg/L) | 2024-03-01 | 20.9 µg/L | 15% (100% elevada) | **La mejor de Yojoa.** Lago completo, gradiente nítido azul(centro)→rojo(orillas/sur) |
| `mapa_val_yojoa_e7_2024-11-06.png` | Yojoa — mortandad de peces nov-2024 | 2024-11-06 | 10.7 µg/L | 0% (57% elevada) | **Buena (evento).** Biomasa elevada; se ven estrías de algas (única escena del episodio, algo neblinosa) |
| `mapa_val_yojoa_e1_2023-03-02.png` | Yojoa — "Intervenido…" feb-2023 | 2023-03-02 | 29.6 µg/L | 100% | Floración generalizada |
| `mapa_val_fonseca_2023-03-02.png` | Fonseca — muestreo LABTOX (diatomeas) | 2023-03-02 | 5.1 µg/L | 18% | **Clave conceptual.** Gradiente nítido: mar abierto bajo (azul) → canales/manglares altos (rojo); coincide con diatomeas, sin marea roja → valida biomasa≠toxicidad |
| `mapa_val_cajon_e4_2026-01-25.png` | Cajón — episodio mayor (sin verdad de campo) | 2026-01-25 | 22.5 µg/L | 30% | Embalse completo y contiguo, **gradiente suave azul→rojo** (escena despejada de época seca) |
| `mapa_val_cajon_e1_2024-03-06.png` | Cajón — floración seca (pico 4-mar-2024, 57 µg/L) | 2024-03-06 | 35.3 µg/L | **100%** | **Enfocada al embalse.** Escena despejada; gradiente nítido azul→rojo en los brazos del río |
| `mapa_val_tampa_e1_2023-01-13.png` | Tampa Bay — marea roja ene-2023 | 2023-01-13 | 4.2 µg/L | 1% | ⚠️ **No usar como intensidad** (ver nota marina) |
| `mapa_val_tampa_e9_2025-01-12.png` | Tampa Bay — marea roja invierno 24-25 | 2025-01-12 | 3.7 µg/L | 4% | ⚠️ **No usar como intensidad** (ver nota marina) |

**Cómo usar estas figuras en la tesis:**
- **Figuras principales (agua dulce):** Okeechobee E3, Okeechobee E6, **Yojoa E4** (la mejor de Yojoa,
  lago completo con gradiente nítido) y Cajón E4. Muestran el mapa de biomasa con gradiente espacial
  coincidiendo con el periodo de floración validado. Son las más contundentes.
- **Figura de evento (Yojoa):** Yojoa E7 (2024-11-06) liga el mapa a la mortandad de peces; úsala junto
  a Yojoa E4 (E4 = mejor imagen del fenómeno; E7 = coincidencia con el evento documentado).
- **Figura clave conceptual:** Fonseca 2023-03-22 — ilustra biomasa elevada real (diatomeas) sin
  toxicidad; úsala para sostener el encuadre biomasa≠floración nociva.
- **Cajón:** ahora con escenas despejadas dentro del episodio (nov-2025 y feb-2024) el embalse se ve
  contiguo y con gradiente. Inclúyelas declarando que **no hay verdad de campo** (sin monitoreo público,
  razón de su estatus EXPLORATORIO): el modelo detecta floración pero no hay reporte independiente que
  contrastar.
- **Tampa Bay (marino):** ⚠️ el mapa por píxel da chl-a **baja ("NORMAL")** porque la marea roja
  (*Karenia brevis*, dinoflagelado) no se traduce en clorofila-a satelital alta como las cianobacterias,
  y porque en costa la alerta del modelo proviene de la **dinámica a nivel de cuerpo, no del mapa por
  píxel** (límite ya documentado en el proyecto). Para Tampa, la validación es la coincidencia de la
  ALERTA con el evento, **no** la figura. No presentar el mapa marino como prueba de intensidad.

> Regenerar: `python build_validacion_maps.py` (requiere los rasters en `imagenes/` y los modelos en
> `artifacts/models/`).

---

## 3. Prompt de investigación (listo para usar con Claude / asistente de búsqueda)

> Copia este bloque en una herramienta con acceso a búsqueda web. Las fechas ya están insertadas.
> Pídele que devuelva la tabla con las columnas de evidencia completadas.

```
Realiza una investigación exhaustiva para validar los resultados de mi modelo de detección de
biomasa algal y floraciones algales nocivas (FAN).

Objetivo: determinar si existen evidencias independientes (artículos científicos, informes técnicos,
noticias, comunicados gubernamentales, publicaciones de universidades, imágenes satelitales
analizadas o reportes de monitoreo) que coincidan con las fechas/episodios en que mi modelo detectó
biomasa elevada o floración.

Mi estudio cubre 5 cuerpos de agua (2023–2026). Para CADA cuerpo y CADA episodio listado abajo:
  1. Busca evidencia publicada dentro de un margen de ±15 días del rango o del pico del episodio.
  2. Indica si la coincidencia es Directa, Parcial, Sin evidencia o Contradice.
  3. Resume la evidencia. 4. Incluye la fuente original con enlace. 5. Indica el nivel de
     confiabilidad (artículo científico / institución gubernamental / universidad / organismo
     internacional / medio de comunicación). 6. Explica si la evidencia respalda la detección.

Busca en español e inglés. Considera: floraciones algales nocivas (HAB/FAN), incrementos de biomasa
algal, cianobacterias, cambios de color del agua, alertas de calidad del agua, mortandad de peces,
exceso de clorofila-a, eutrofización, y monitoreos satelitales (Sentinel-2/3, Landsat, Copernicus,
NASA, VIIRS/OLCI). Revisa repositorios académicos (Google Scholar, Scopus, Springer, ScienceDirect,
MDPI), organismos (NASA, Copernicus, ESA, NOAA, UNEP, FAO), instituciones hondureñas (MiAmbiente/
SERNA, UNAH, ICF) y de Florida (FWC, NOAA, USF, SFWMD) y medios de comunicación.

EPISODIOS A VALIDAR:

[Lago Okeechobee — Florida, EE.UU. — centroide 26.95N, 80.85O — agua dulce]
  E1: 2023-10-31 (pico 29.9 µg/L)
  E2: 2023-12-23 (pico 66.9 µg/L)
  E3: 2024-01-30 a 2024-10-05, pico 2024-03-08 (101.7 µg/L) — episodio mayor
  E4: 2024-12-14 (pico 40.4 µg/L)
  E5: 2025-02-05 a 2025-02-11, pico 2025-02-11 (36.5 µg/L)
  E6: 2025-05-01 a 2026-04-10, pico 2025-08-13 (101.7 µg/L) — episodio mayor

[Bahía de Tampa — Florida, EE.UU. — centroide 27.73N, 82.58O — marino/estuarino]
  E1: 2023-01-07 a 2023-01-26, pico 2023-01-23 (25.2 µg/L)
  E2: 2023-02-19 (9.1 µg/L)
  E3: 2023-03-29 a 2023-04-09, pico 2023-04-09 (15.7 µg/L)
  E4: 2023-07-24 (8.8 µg/L)
  E5: 2023-09-19 a 2023-10-12, pico 2023-10-12 (64.7 µg/L)
  E6: 2023-11-16 a 2024-01-29, pico 2023-11-16 (60.2 µg/L)
  E7: 2024-03-18 a 2024-03-28 (7.9 µg/L)
  E8: 2024-07-09 (13.6 µg/L)
  E9: 2024-08-06 a 2025-03-18, pico 2025-01-17 (45.0 µg/L) — episodio mayor
  E10: 2025-06-29 (6.8 µg/L)
  E11: 2025-08-10 a 2025-10-26, pico 2025-08-10 (31.3 µg/L)
  E12: 2025-12-06 a 2025-12-18 (9.5 µg/L)
  E13: 2026-01-18 a 2026-01-26, pico 2026-01-26 (24.3 µg/L)
  E14: 2026-02-22 (39.7 µg/L)

[Lago de Yojoa — Honduras — centroide 14.87N, 87.96O — agua dulce]
  E1: 2023-02-14 (24.6 µg/L)
  E2: 2023-09-26 (34.3 µg/L)
  E3: 2023-10-27 (54.8 µg/L)
  E4: 2023-12-25 a 2024-05-04, pico 2024-01-18 (95.9 µg/L) — episodio mayor (verano seco)
  E5: 2024-06-02 a 2024-06-04, pico 2024-06-02 (60.7 µg/L)
  E6: 2024-08-05 a 2024-10-07, pico 2024-08-25 (75.4 µg/L)
  E7: 2024-10-31 (28.8 µg/L)
  E8: 2024-12-28 (31.3 µg/L)
  E9: 2025-02-03 a 2025-03-06, pico 2025-03-01 (45.6 µg/L)
  E10: 2025-06-02 a 2025-06-07, pico 2025-06-07 (37.2 µg/L)
  E11: 2025-07-09 a 2025-07-25, pico 2025-07-09 (27.0 µg/L)
  E12: 2025-08-21 a 2025-09-06, pico 2025-08-21 (26.5 µg/L)
  E13: 2025-11-09 (66.5 µg/L)

[Embalse El Cajón — Honduras — centroide 14.83N, 87.69O — agua dulce]
  E1: 2024-02-11 a 2024-03-04, pico 2024-03-04 (57.3 µg/L)
  E2: 2024-09-07 (85.4 µg/L)
  E3: 2025-01-29 (26.1 µg/L)
  E4: 2025-06-26 a 2026-04-27, pico 2025-06-26 (97.3 µg/L) — episodio mayor

[Golfo de Fonseca — Honduras — centroide 13.18N, 87.60O — marino/estuarino]
  E1: 2023-01-09 a 2023-04-17, pico 2023-03-12 (15.2 µg/L)
  E2: 2023-05-20 a 2023-08-05, pico 2023-06-26 (31.3 µg/L)
  E3: 2023-09-01 a 2023-09-03 (6.6 µg/L)
  E4: 2023-10-02 a 2023-11-01, pico 2023-10-29 (20.7 µg/L)
  E5: 2023-12-18 (5.7 µg/L)
  E6: 2024-01-14 a 2024-04-15, pico 2024-02-21 (12.2 µg/L)
  E7: 2024-05-10 a 2024-07-11, pico 2024-05-10 (50.2 µg/L)
  E8: 2024-09-07 a 2024-10-27, pico 2024-10-21 (14.6 µg/L)
  E9: 2024-11-18 a 2025-02-03, pico 2024-11-19 (58.0 µg/L)
  E10: 2025-03-02 a 2025-10-22, pico 2025-05-08 (28.1 µg/L)
  E11: 2026-01-05 a 2026-02-19, pico 2026-01-23 (19.6 µg/L)
  E12: 2026-06-04 a 2026-06-17, pico 2026-06-08 (6.8 µg/L)

Términos de búsqueda sugeridos (combinar ES/EN):
  "Lago de Yojoa floración algal" / "Lake Yojoa cyanobacteria" / "Lake Yojoa chlorophyll" /
  "Lago de Yojoa calidad del agua" / "Yojoa eutrophication" / "cyanobacteria Honduras" /
  "El Cajón embalse algas" / "Golfo de Fonseca marea roja / floración" / "Gulf of Fonseca HAB" /
  "Lake Okeechobee algal bloom 2024" / "Tampa Bay red tide Karenia brevis" /
  "Sentinel Yojoa algae" / "Copernicus Yojoa" / "NASA chlorophyll Yojoa".

Presenta los resultados en una tabla:
| Cuerpo | Episodio | Pico | Evidencia encontrada | Fecha evidencia | Coincidencia | Resumen | Fuente | Confiabilidad |

IMPORTANTE: no asumas coincidencias sin evidencia documental. Cita solo fuentes verificables con
enlace. Si no hay evidencia para un episodio, indícalo explícitamente y sugiere la razón probable
(falta de monitoreo in-situ en Honduras, ausencia de publicaciones, cobertura mediática nula, etc.).
```

---

## 4. Notas para interpretar la validación

- **Esperable que Florida tenga mucha evidencia y Honduras poca.** Okeechobee y Tampa Bay cuentan con
  monitoreo operativo (FWC red tide, NOAA HAB, SFWMD) y prensa; las coincidencias deberían ser
  abundantes y servir como **prueba de concepto del método de validación**. En Honduras es probable
  encontrar **poca o ninguna evidencia documental** para fechas específicas — esto **no invalida** la
  detección: refleja la ausencia de programas de monitoreo (confirmada en `ESTADO_PROYECTO.md`: no
  existe in-situ público de Yojoa 2023–2026; El Cajón sin programa). Conviene declararlo como
  limitación de la *verdad de terreno*, no del modelo.
- **Estacionalidad como respaldo indirecto.** En Yojoa los episodios mayores caen en la **época seca
  (dic–abr)**, consistente con la limnología de lagos tropicales (mayor estabilidad, menor flushing →
  acumulación de biomasa). Si la literatura general de Yojoa describe ese patrón, cuenta como
  coincidencia *parcial* aunque no haya un reporte de fecha exacta.
- **Distinguir biomasa de toxicidad.** Al redactar, no escribir "el modelo predijo una floración
  nociva confirmada" salvo que la fuente documente toxinas/sanidad. Lo defendible es: "el modelo
  detectó biomasa elevada coincidente con [evento documentado]".
- **Detalle a nivel de fecha individual** (las 74/206/… fechas exactas por cuerpo) está en
  `artifacts/targets/combined_target.csv`; regenera el desglose con el umbral p85≤24 de `config.py`.

---

*Documento generado a partir de la serie target del modelo (`combined_target.csv`) y los umbrales de
`config.py`. No modifica el modelado ni los números de validación interna (`REPORTE_DEFENSA.md`).*
