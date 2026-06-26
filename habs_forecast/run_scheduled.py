"""
run_scheduled.py — Tarea PROGRAMADA: emite el pronostico operativo y verifica lo madurado.

Pensado para ejecutarse periodicamente (p.ej. diario via Task Scheduler de Windows):
  1) run_forecast.run()   -> emite pronostico de la ultima escena y lo apenda a forecast_log.csv
  2) verify_forecasts.main() -> evalua los pronosticos ya madurados (target real disponible)
Asi la bitacora MADURA sola con el tiempo y verify_forecasts muestra desempeno real SIN backfill.

Registra todo en artifacts/forecasts/scheduled.log (ademas de la consola). Robusto: si un paso
falla, lo loguea y continua. No entrena ni modifica modelos.

Uso manual:  python run_scheduled.py
Registrar tarea diaria (Windows, 06:00):  ver register_task.ps1 / README.
"""
from __future__ import annotations
import os, logging
import config as C

LOGFILE = os.path.join(C.DIR_FORECASTS, "scheduled.log")
os.makedirs(C.DIR_FORECASTS, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOGFILE, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("run_scheduled")


def main():
    log.info("=== Tarea programada: pronostico + verificacion ===")
    try:
        import run_forecast
        run_forecast.run()
        log.info("run_forecast OK")
    except Exception as e:
        log.exception("run_forecast fallo: %s", e)
    try:
        import verify_forecasts
        verify_forecasts.main()
        log.info("verify_forecasts OK")
    except Exception as e:
        log.exception("verify_forecasts fallo: %s", e)
    log.info("=== Fin tarea programada ===")


if __name__ == "__main__":
    main()
