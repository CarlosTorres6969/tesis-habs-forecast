# experiments/ — módulos experimentales (NO son parte del sistema que corre)

Estos módulos **no** los usa el pipeline de producción ni la validación de defensa.
Se conservan como **evidencia de rigor**: comparaciones de arquitectura y resultados
negativos documentados que justifican por qué el sistema final quedó como quedó.
Puedes borrarlos y el sistema (predict / run_forecast / evaluate_nested) sigue funcionando.

## Cómo ejecutarlos

Importan módulos del paquete principal (`config`, `train`, `evaluate_robust`), así que
se corren **desde la carpeta `habs_forecast/`** con el directorio padre en el path:

```bash
cd habs_forecast
PYTHONPATH=. python experiments/compare_models.py
```

## Qué hay aquí y su veredicto

- **compare_models.py** — compara arquitecturas (XGBoost vs HistGB vs Ridge vs MLP) con
  OOS + bootstrap en agua dulce. Veredicto: **XGBoost gana** (+0.168 [+0.135,+0.200]);
  MLP sobreajusta (−0.274). Justifica empíricamente usar árboles con N~2.600.
- **compare_lstm.py** — LSTM/GRU aparte (requieren secuencias regulares). Veredicto:
  LSTM +0.021 (no significativo) → no aporta sobre XGBoost.
- **tune_xgb.py** — barrido de hiperparámetros de XGBoost. Veredicto: mejora solo +0.008
  (modelo limitado por datos, no por tuning). Se aplicó `reg_lambda=3.0` y se cerró.
- **improve_coast.py** — pooling de grupo para la costa. Veredicto: arregló un +5d
  negativo pero el skill costero siguió no significativo → la costa estaba limitada por
  datos del predictor (luego resuelto densificando S2/OLCI, no aquí).
- **experiment_zonify_okeechobee.py** — hipótesis: dividir Okeechobee en 4 cuadrantes
  mejora el skill. Veredicto: **NO ayuda, empeora**; el cuello es la resolución del
  target (VIIRS 750 m), no la agregación espacial.
- **analyze_importance.py** — importancia de features (ganancia XGBoost). Superado por
  `explain_model.py` (SHAP) en el sistema principal; se conserva por referencia.
