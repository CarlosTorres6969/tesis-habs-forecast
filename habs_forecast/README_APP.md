# App de demostración (Streamlit) — `app.py`

Interfaz que **envuelve** el sistema de pronóstico temprano de riesgo de biomasa algal (clorofila‑a
como proxy) a 0–7 días. **No** implementa modelado: reutiliza `make_maps.build_map_figure`,
`predict.forecast_body` y `guards.evaluate_guards`.

## Cómo correrla (LOCAL — recomendado para la defensa)

```bash
# desde la carpeta habs_forecast/
pip install -r ../requirements.txt        # incluye streamlit
streamlit run app.py
```

Se abre en el navegador (`http://localhost:8501`). Requiere los modelos de producción en
`artifacts/models/` (si faltan: `python train_final.py`) y las escenas Sentinel‑2 en `imagenes/`.

> Pensada para correr **en local** durante la defensa: los datos pesados (rasters, modelos) no están
> en el repo y la app los lee del disco.

## Flujo

1. Elegí un **cuerpo de agua** (solo los 5 validados: Okeechobee, Tampa Bay, Cajón, Golfo de Fonseca,
   Lago de Yojoa) — muestra su tipo (lago/costa) y país.
2. Elegí el **horizonte** (1, 3, 5 o 7 días).
3. Elegí la **escena Sentinel‑2**:
   - *Usar escena de ejemplo*: una de las `.tif` ya presentes del cuerpo (por fecha), o
   - *Subir GeoTIFF*: un raster Sentinel‑2 de **5 bandas** (B2, B3, B4, B5, B8).
4. **Analizar** → muestra: imagen satelital real, mapa de biomasa prevista (+contorno de riesgo),
   bandera de alerta calibrada, banda de incertidumbre P10–P90, y el disclaimer.

## Reglas de honestidad

- **No acepta fotos normales** (RGB de celular / Maps): necesita las bandas **red‑edge (B5)** y
  **NIR (B8)**, que estiman clorofila y una foto común no tiene.
- **Validada solo para los 5 cuerpos**; es **pronóstico a futuro**, no detección sobre la imagen.
- **Cajón** se marca **EXPLORATORIO** (menor confianza, sin verdad de campo 2023–2026).
- **Clorofila‑a = proxy de biomasa**, no de toxicidad: la alerta señala **riesgo** que requiere
  **verificación de campo**.
