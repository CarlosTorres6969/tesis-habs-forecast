# Manual de Usuario
## Sistema de Alerta Temprana de Biomasa Algal (HABs)
### Pronóstico de clorofila-a a 0–7 días

---

## 1. ¿Qué es este programa?

Es una herramienta de **alerta temprana** que **pronostica** la biomasa algal
—usando la **clorofila-a como proxy**— en cuerpos de agua, con **0 a 7 días de
anticipación**.

Trabaja a partir de **imágenes satelitales Sentinel-2** y modelos ya entrenados.
No reentrena nada: envuelve la lógica de pronóstico existente en una interfaz web
sencilla (Streamlit).

> ⚠️ **Muy importante:** es un **PRONÓSTICO a futuro**, no una detección sobre la
> imagen que usted sube. Estima cómo estará la biomasa dentro de 1, 3, 5 o 7 días.

### Lo que SÍ hace
- Estima la **clorofila-a media prevista** (µg/L) a +1, +3, +5 y +7 días.
- Genera un **mapa de biomasa algal prevista** por píxel.
- Da una **alerta por nivel** (Normal / Elevada / Floración).
- Calcula una **banda de incertidumbre** (P10–P90, ~80% de confianza).
- Estima la **probabilidad de una anomalía** (salto atípico para ese cuerpo).
- Explica **por qué** (variables más influyentes, SHAP).
- Permite **descargar** mapa, animación y resumen CSV.

### Lo que NO hace
- **No confirma toxicidad** ni identifica la especie (cianobacterias, marea roja).
  La clorofila-a mide el **pigmento**, no la toxina ni el organismo.
- **No acepta fotos comunes** (RGB de celular ni capturas de Google Maps).
- **No funciona fuera de los 5 cuerpos validados.**

---

## 2. Requisitos

- **Python** con las dependencias del proyecto instaladas.
- **Modelos de producción** presentes en `artifacts/models/`
  (si faltan, se generan con `python train_final.py`).
- Un navegador web (la app abre en el navegador automáticamente).

---

## 3. Cómo abrir el programa

Desde una terminal, ubicado en la carpeta `habs_forecast`:

```bash
streamlit run app.py
```

Se abrirá una página en el navegador con el título
**"Alerta temprana de biomasa algal (HABs)"**.

---

## 4. Cuerpos de agua disponibles (validados)

| Cuerpo             | Tipo                     | País           |
|--------------------|--------------------------|----------------|
| Lago Okeechobee    | Lago / agua dulce        | Estados Unidos |
| Bahía de Tampa     | Costa / marino-estuarino | Estados Unidos |
| Embalse El Cajón   | Lago / agua dulce        | Honduras       |
| Golfo de Fonseca   | Costa / marino-estuarino | Honduras       |
| Lago de Yojoa      | Lago / agua dulce        | Honduras       |

> 🔬 Algunos cuerpos pueden marcarse como **EXPLORATORIOS**: no tienen verdad de
> campo in-situ en la ventana 2023–2026 y tienen menos datos. Sus resultados son de
> **menor confianza** y la app lo advierte en pantalla.

---

## 5. Recorrido por la interfaz

### 5.1. Barra lateral (izquierda) — "Cómo leer esta herramienta"
Explica las reglas clave del sistema (es pronóstico, requiere Sentinel-2 de 5 bandas,
solo 5 cuerpos, clorofila = proxy). Léala una vez antes de usar la herramienta.

### 5.2. Selectores (parte superior)
1. **Cuerpo de agua** — elija uno de los 5 cuerpos validados.
2. **Horizonte de pronóstico** — +1, +3, +5 o +7 días (por defecto **+3**).
   - **+1 y +7 días** son horizontes *body-level*: el modelo predice el **nivel del
     cuerpo completo** y el mapa reparte ese nivel según el patrón espacial actual
     (es una estimación, no un pronóstico píxel a píxel).
   - **+3 y +5 días** usan señal espectral por píxel: el mapa muestra un **gradiente
     real** con detalle espacial.
3. **Tipo / País** — informativo (se completa solo según el cuerpo elegido).

### 5.3. Entrada de escena Sentinel-2
Dos modos:

**A) Usar escena de ejemplo** (recomendado para la demostración)
- La app **ordena las escenas por calidad** (agua más limpia primero).
- Con la casilla *"Usar automáticamente la mejor escena"* marcada, elige sola la
  mejor fecha (evita escenas nubladas donde el agua ni se ve).
- Si la desmarca, puede elegir la fecha manualmente en el desplegable (la mejor
  aparece con ⭐).
- Si la escena elegida tiene nubosidad/neblina, aparece una advertencia.

**B) Subir GeoTIFF**
- Debe ser un **raster georreferenciado Sentinel-2 de 5 bandas** en el orden
  **B2 (azul), B3 (verde), B4 (rojo), B5 (red-edge), B8 (NIR)**.
- Las bandas red-edge (B5) e infrarrojo (B8) son las que estiman la clorofila; por
  eso **una foto RGB común es rechazada**.
- El contexto no espectral (clorofila reciente, ERA5, in-situ) se toma de la última
  fecha disponible del cuerpo.

### 5.4. Botón "Analizar"
- (Opcional) Marque **"🎬 Mostrar animación tipo pronóstico del clima"** para generar
  un video corto que recorre +1 → +7 días. Debe marcarse **antes** de Analizar.
- Pulse **🔍 Analizar**. Aparece un indicador de progreso mientras procesa la escena y
  genera el pronóstico.

---

## 6. Cómo leer los resultados

Tras "Analizar" se muestra, de arriba hacia abajo:

### 6.1. Encabezado
Cuerpo, horizonte, fecha de la escena (**t0**) y **etiqueta de confianza**
(según frescura de datos, cobertura de píxeles de agua y estado).

### 6.2. Mapa (2 paneles)
- **Izquierda:** imagen satelital real de la escena.
- **Derecha:** **biomasa algal prevista** por píxel para el horizonte elegido.

### 6.3. Insignia de credibilidad
Compara el *skill* validado del modelo frente a la persistencia (línea base) en ese
horizonte.

### 6.4. Animación (si la activó)
Video en bucle que arranca en el estado observado de hoy y recorre la biomasa prevista
a 1, 3, 5 y 7 días, como un pronóstico del clima en la tele.

### 6.5. Nivel de biomasa algal (banner de color)
| Color | Nivel             | Significado                                      |
|-------|-------------------|--------------------------------------------------|
| 🟢    | **NORMAL**        | Clorofila prevista por debajo del umbral elevado |
| 🟡    | **BIOMASA ELEVADA** | Clorofila prevista por encima del umbral elevado |
| 🔴    | **FLORACIÓN**     | Clorofila prevista igual o mayor al umbral de floración |

Debajo se indica el **% de área** en floración y en biomasa elevada.

### 6.6. Dos señales que pueden diferir (¡no confundir!)
- **NIVEL (magnitud):** cuánta clorofila se prevé, comparada con umbrales biológicos
  (es el banner de color).
- **Probabilidad de anomalía (P85):** la probabilidad de un **salto atípico para ESE
  cuerpo** (clasificador calibrado).

Un embalse crónicamente alto puede dar **nivel alto** y **anomalía baja**: ese nivel
es *su* normal, no un evento inusual. Y al revés. La app muestra una nota aclaratoria
cuando ambas señales divergen.

### 6.7. Clorofila-a prevista (intensidad)
Valor medio previsto en µg/L con su **banda de incertidumbre P10–P90** (calibrada,
CQR ~80%).

### 6.8. Pestañas dinámicas
- **📈 Trayectoria 0–7 días:** clorofila prevista en cada horizonte con su banda; el
  punto grande es el horizonte seleccionado, el color indica el nivel.
- **🎯 Medidor de riesgo:** probabilidad calibrada de salto anómalo a +N días. La línea
  roja es el **umbral operativo real**: por encima, el sistema dispara alerta. El
  umbral es bajo a propósito (prioriza no perder eventos → alta sensibilidad).
- **🧠 ¿Por qué? (SHAP):** variables que más pesan en el pronóstico. En corto plazo
  domina la clorofila reciente; a mayor horizonte entran meteorología, nutrientes e
  índices espectrales.

---

## 7. Descargas

En la sección **💾 Descargas** hay tres botones:

| Botón                 | Archivo                              | Contenido                          |
|-----------------------|--------------------------------------|------------------------------------|
| 🖼️ **Mapa (PNG)**     | `mapa_<cuerpo>_h<N>_<fecha>.png`     | Los dos paneles del mapa           |
| 🎬 **Animación (GIF)** | `animacion_<cuerpo>_<fecha>.gif`     | Solo si activó la animación        |
| 📄 **Pronóstico (CSV)** | `pronostico_<cuerpo>_<fecha>.csv`    | Todos los horizontes (0–7 d)       |

El **CSV** incluye, por horizonte: clorofila prevista, P10, P90, probabilidad de
alerta y nivel; más una cabecera con metadatos y el recordatorio de que **no confirma
toxicidad**.

---

## 8. Interpretación responsable (léelo)

- La clorofila-a es un **proxy de biomasa**, **no** una medida de toxicidad ni de la
  especie. Una alerta indica **riesgo que amerita verificación de campo**.
- Confirmar cianobacterias, dinoflagelados o toxinas requiere **muestreo de campo**
  (microscopía / ensayos de toxinas).
- Fuera de los 5 cuerpos validados **no hay modelo ni calibración**: no use la
  herramienta ahí.
- Trate los resultados de cuerpos **exploratorios** con cautela.

> ⚠️ **Proxy de biomasa algal (clorofila-a).** NO confirma toxicidad ni floración
> nociva. Herramienta de **alerta temprana**; requiere **verificación de campo**.

---

## 9. Problemas frecuentes

| Situación | Causa probable | Qué hacer |
|-----------|----------------|-----------|
| "Faltan los modelos de producción" | No están en `artifacts/models/` | Ejecutar `python train_final.py` |
| "No hay modelo entrenado para … a +N días" | Falta ese modelo/horizonte | Entrenar o elegir otro horizonte |
| "El archivo NO tiene 5 bandas válidas" | Subió una foto RGB o un TIFF incompleto | Usar un GeoTIFF Sentinel-2 de 5 bandas (B2,B3,B4,B5,B8) |
| "Muy pocos píxeles de agua válidos" | Escena nublada o sin agua | Probar otra fecha o la mejor escena automática |
| Advertencia de nubosidad/neblina | La escena elegida tiene poco agua clara | Usar la mejor escena automática u otra fecha |
| "Explicabilidad SHAP no disponible" | No se generó SHAP | Ejecutar `python explain_model.py` |
| El botón "Analizar" está gris | No hay escena cargada | Seleccionar una escena de ejemplo o subir un GeoTIFF válido |

---

## 10. Ficha técnica (resumen)

- **Modelo de intensidad + intervalos:** XGBoost con CQR (regresión cuantílica
  conformalizada).
- **Alerta:** red neuronal (HABNet), por grupo ecológico y horizonte.
- **Validación:** pronóstico causal sin fuga de información (validación anidada).
- **Entrada:** escena Sentinel-2 de 5 bandas (B2, B3, B4, B5, B8) + contexto no
  espectral (clorofila reciente, ERA5, in-situ).
- **Salida:** clorofila-a prevista, banda P10–P90, nivel, probabilidad de anomalía,
  mapa por píxel y explicación SHAP.

---

*Documento generado como manual de usuario del sistema de alerta temprana de biomasa
algal (HABs). Herramienta de apoyo a la decisión; no sustituye la verificación de campo.*
