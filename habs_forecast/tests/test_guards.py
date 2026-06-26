"""test_guards.py — guardas de frescura/cobertura/estado y regla de 'peor condicion'."""
import pandas as pd
import config as C
import guards

RUN = pd.Timestamp("2026-06-26")


def test_data_age_days():
    assert guards.data_age_days("2026-06-26", RUN) == 0
    assert guards.data_age_days("2026-06-12", RUN) == 14
    assert guards.data_age_days("2026-06-01", RUN) == 25


def test_ok_cuando_fresco_y_con_cobertura():
    conf, flags, age = guards.evaluate_guards("okeechobee", "2026-06-20",
                                              C.MIN_WATER_PIXELS + 100, RUN)
    assert conf == "OK" and flags == [] and age == 6


def test_stale_por_escena_vieja():
    conf, flags, _ = guards.evaluate_guards("okeechobee", "2026-05-01",
                                            C.MIN_WATER_PIXELS + 100, RUN)
    assert conf == "STALE" and "STALE" in flags


def test_low_coverage_por_pocos_pixeles():
    conf, flags, _ = guards.evaluate_guards("okeechobee", "2026-06-25",
                                            C.MIN_WATER_PIXELS - 1, RUN)
    assert conf == "LOW_COVERAGE" and "LOW_COVERAGE" in flags


def test_exploratorio_por_cuerpo():
    conf, flags, _ = guards.evaluate_guards("cajon", "2026-06-25",
                                            C.MIN_WATER_PIXELS + 100, RUN)
    assert conf == "EXPLORATORIO" and flags == ["EXPLORATORIO"]


def test_peor_condicion_manda():
    # escena vieja + Cajon + poca cobertura -> debe reportar la PEOR (LOW_COVERAGE)
    conf, flags, _ = guards.evaluate_guards("cajon", "2026-01-01",
                                            C.MIN_WATER_PIXELS - 5, RUN)
    assert set(flags) == {"LOW_COVERAGE", "STALE", "EXPLORATORIO"}
    assert conf == "LOW_COVERAGE"   # primero en CONFIDENCE_SEVERITY


def test_worst_confidence_vacio_es_ok():
    assert guards.worst_confidence([]) == "OK"
