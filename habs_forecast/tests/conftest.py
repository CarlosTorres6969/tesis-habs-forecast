"""conftest.py — pone habs_forecast/ en sys.path para que los tests importen los modulos
(config, guards, run_forecast, verify_forecasts) sin depender del directorio de ejecucion."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
