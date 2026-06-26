"""
guards.py — Guardas de FRESCURA y COBERTURA para la capa operativa de alerta.

Determinan la CONFIANZA de un pronostico segun la calidad del DATO de entrada (no del modelo):
  - LOW_COVERAGE : la escena t0 tiene menos pixeles de agua que config.MIN_WATER_PIXELS
                   (la senal espectral no es fiable).
  - STALE        : la escena t0 es mas vieja que config.MAX_DATA_AGE_DAYS (dato desactualizado;
                   nubosidad prolongada -> sin imagen reciente).
  - EXPLORATORIO : cuerpo sin validacion suficiente (config.EXPLORATORY_BODIES, p.ej. Cajon).
  - OK           : ninguna condicion adversa.

La 'confianza' reportada es la PEOR condicion aplicable (orden config.CONFIDENCE_SEVERITY).
Las condiciones NO se silencian: se marcan para que el operador las vea.
Compartido por run_forecast.py (bucle operativo) y predict.py (predictor manual).
"""
from __future__ import annotations
import pandas as pd
import config as C


def data_age_days(t0, run_ts=None):
    """Antiguedad en dias de la escena t0 respecto al momento de ejecucion (run_ts).
    Por defecto usa el instante actual. Causal: la escena siempre es <= run_ts."""
    run_ts = pd.Timestamp.now() if run_ts is None else pd.Timestamp(run_ts)
    return int((run_ts.normalize() - pd.Timestamp(t0).normalize()).days)


def evaluate_guards(water_body, t0, n_water_px, run_ts=None):
    """Evalua las guardas para un pronostico y devuelve (confianza, flags, age_dias).
      - flags : lista de TODAS las condiciones adversas detectadas.
      - confianza : la PEOR de ellas (config.CONFIDENCE_SEVERITY); 'OK' si no hay ninguna.
    """
    age = data_age_days(t0, run_ts)
    flags = []
    if n_water_px is not None and n_water_px < C.MIN_WATER_PIXELS:
        flags.append("LOW_COVERAGE")
    if age > C.MAX_DATA_AGE_DAYS:
        flags.append("STALE")
    if water_body in C.EXPLORATORY_BODIES:
        flags.append("EXPLORATORIO")
    return worst_confidence(flags), flags, age


def worst_confidence(flags):
    """Devuelve la PEOR condicion presente segun el orden (peor->mejor) de config."""
    for level in C.CONFIDENCE_SEVERITY:
        if level == "OK":
            return "OK"
        if level in flags:
            return level
    return "OK"
