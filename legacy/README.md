# legacy/ — Sistema ANTIGUO (con fuga de datos). NO USAR.

Estos notebooks y scripts corresponden al **primer sistema**, que tenía **fuga de datos** y por
eso reportaba métricas infladas (AUC≈1.0 falso):

- El objetivo (HAB) se derivaba de la clorofila estimada por las **mismas bandas Sentinel-2** que
  se usaban como predictores → **circularidad**.
- La validación usaba **shuffle** sobre píxeles autocorrelacionados → fuga temporal/espacial.
- El "lead-time" mantenía las bandas de t₀ y solo movía ERA5 → no demostraba anticipación real.

Era **detección/nowcasting con fuga**, no pronóstico.

## El sistema vigente está en `../habs_forecast/`
Pronóstico causal X(≤t₀) → clorofila-a(t₀+h), sin fuga, con validación anidada en test temporal
intacto, intervalos de incertidumbre y test de integridad (`check_integrity.py`). Ver
`habs_forecast/ESTADO_PROYECTO.md`.

Se conserva esta carpeta solo por **trazabilidad histórica**.
