"""test_verify_forecasts.py — el nucleo de verificacion cruza pronosticos con el target real
y calcula error, cobertura de banda y acierto de alerta sobre un caso SINTETICO pequeno."""
import pandas as pd
import verify_forecasts as VF

# Dos pronosticos madurables (h=1 tol [1,2]; h=3 tol [3,4]) y uno NO madurable (sin target).
LOG = pd.DataFrame([
    # h=1: real=12 cae en banda [5,15]; no evento (<25); no alerta -> hit (TN)
    {"run_ts": "r", "water_body": "lab", "group": "freshwater", "t0": "2025-01-01",
     "horizon": 1, "chl_pred": 10.0, "p10": 5.0, "p90": 15.0, "riesgo": False},
    # h=3: real=35 cae en banda [20,40]; evento (>=25); alerta -> hit (TP)
    {"run_ts": "r", "water_body": "lab", "group": "freshwater", "t0": "2025-01-01",
     "horizon": 3, "chl_pred": 30.0, "p10": 20.0, "p90": 40.0, "riesgo": True},
    # h=5: sin target en la ventana -> NO verificable (debe excluirse)
    {"run_ts": "r", "water_body": "lab", "group": "freshwater", "t0": "2025-06-01",
     "horizon": 5, "chl_pred": 99.0, "p10": 1.0, "p90": 2.0, "riesgo": True},
])
TARGET = pd.DataFrame([
    {"water_body": "lab", "fecha": "2025-01-02", "chl_ugl": 12.0},   # objetivo de h=1
    {"water_body": "lab", "fecha": "2025-01-04", "chl_ugl": 35.0},   # objetivo de h=3
])
THR = {"lab": 25.0}


def test_solo_madurados_se_verifican():
    detail, _ = VF.verify(LOG, TARGET, THR)
    assert len(detail) == 2                       # el h=5 sin target queda fuera
    assert set(detail["horizon"]) == {1, 3}


def test_error_y_banda():
    detail, _ = VF.verify(LOG, TARGET, THR)
    h1 = detail[detail.horizon == 1].iloc[0]
    assert h1["chl_real"] == 12.0
    assert h1["error"] == 10.0 - 12.0
    assert bool(h1["in_band"]) is True


def test_acierto_de_alerta():
    detail, _ = VF.verify(LOG, TARGET, THR)
    h1 = detail[detail.horizon == 1].iloc[0]      # no evento, no alerta -> acierto
    h3 = detail[detail.horizon == 3].iloc[0]      # evento, alerta -> acierto
    assert not bool(h1["event_real"]) and bool(h1["alert_hit"])
    assert bool(h3["event_real"]) and bool(h3["riesgo_pred"]) and bool(h3["alert_hit"])


def test_resumen_por_grupo_horizonte():
    _, summary = VF.verify(LOG, TARGET, THR)
    assert set(summary["horizon"]) == {1, 3}
    s1 = summary[summary.horizon == 1].iloc[0]
    assert s1["n"] == 1
    assert s1["MAE"] == 2.0
    assert s1["cobertura_banda"] == 1.0
    assert s1["hit_rate_alerta"] == 1.0


def test_fuera_de_banda_se_detecta():
    log = LOG.copy()
    log.loc[0, "p90"] = 11.0                       # ahora 12 > 11 -> fuera de banda
    detail, _ = VF.verify(log, TARGET, THR)
    h1 = detail[detail.horizon == 1].iloc[0]
    assert bool(h1["in_band"]) is False
