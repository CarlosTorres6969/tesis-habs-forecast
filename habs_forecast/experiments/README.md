# experiments/ — módulos experimentales (NO son parte del sistema que corre)

Estos módulos **no** los usa el pipeline de producción ni la validación de defensa.
Se conservan como experimentos reproducibles. Todos los scripts usan cortes temporales
con embargo; los números obtenidos con versiones anteriores de partición por filas ya no
deben citarse como evidencia. Hay que ejecutar de nuevo cada experimento antes de reportarlo.
Puedes borrarlos y el sistema (predict / run_forecast / evaluate_nested) sigue funcionando.

## Cómo ejecutarlos

Importan módulos del paquete principal (`config`, `train`, `evaluate_robust`), así que
se corren **desde la carpeta `habs_forecast/`** con el directorio padre en el path:

```bash
cd habs_forecast
PYTHONPATH=. python experiments/compare_models.py
```

## Qué hay aquí

- **compare_models.py** — compara arquitecturas (XGBoost vs HistGB vs Ridge vs MLP)
  mediante evaluación temporal purgada.
- **compare_lstm.py** — compara LSTM/GRU, que requieren secuencias regulares, con una
  referencia XGBoost usando cortes temporales purgados.
- **tune_xgb.py** — barrido temporal purgado de hiperparámetros de XGBoost.
- **improve_coast.py** — prueba el pooling de grupo para la costa con evaluación temporal.
- **experiment_zonify_okeechobee.py** — prueba la hipótesis de dividir Okeechobee en
  cuatro cuadrantes mediante un holdout temporal común y purgado.
- **analyze_importance.py** — importancia de features (ganancia XGBoost). Superado por
  `explain_model.py` (SHAP) en el sistema principal; se conserva por referencia.
