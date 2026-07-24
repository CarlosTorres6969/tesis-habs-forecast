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
LOG["evaluation_mode"] = "operational"
TARGET = pd.DataFrame([
    {"water_body": "lab", "fecha": "2025-01-02", "chl_ugl": 12.0},   # objetivo de h=1
    {"water_body": "lab", "fecha": "2025-01-04", "chl_ugl": 35.0},   # objetivo de h=3
])
THR = {"lab": 25.0}


def test_solo_madurados_se_verifican():
    detail, _ = VF.verify(LOG, TARGET, THR)
    assert len(detail) == 2                       # el h=5 sin target queda fuera
    assert set(detail["horizon"]) == {1, 3}


def test_retroactivo_y_legacy_no_cuentan_como_operacional():
    retrospective = LOG.assign(evaluation_mode="retrospective_in_sample")
    detail, summary = VF.verify(retrospective, TARGET, THR)
    assert detail.empty and summary.empty

    legacy = LOG.drop(columns=["evaluation_mode"])
    detail, summary = VF.verify(legacy, TARGET, THR)
    assert detail.empty and summary.empty


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


def test_metricas_de_alerta_pod_far():
    """POD/FAR/precision/F1 se desglosan bien. h=3: 1 evento real, 1 alerta correcta (TP)
    -> POD=1, FAR=0, F1=1. h=1: sin eventos ni alertas -> POD/FAR indefinidos (NaN)."""
    import pandas as pd
    _, summary = VF.verify(LOG, TARGET, THR)
    s3 = summary[summary.horizon == 3].iloc[0]
    assert s3["eventos_reales"] == 1 and s3["alertas_emitidas"] == 1
    assert s3["POD"] == 1.0 and s3["FAR"] == 0.0
    assert s3["precision"] == 1.0 and s3["F1"] == 1.0
    s1 = summary[summary.horizon == 1].iloc[0]      # ni evento ni alerta -> indefinidos
    assert s1["eventos_reales"] == 0 and s1["alertas_emitidas"] == 0
    assert pd.isna(s1["POD"]) and pd.isna(s1["FAR"])


def test_deduplica_reejecuciones():
    """Un mismo (cuerpo, horizonte, t0) repetido en dos run_ts NO debe contarse dos veces:
    se conserva la corrida mas reciente. Aqui la segunda corrida corrige la prediccion."""
    dup = pd.concat([
        LOG,
        LOG.assign(run_ts="a_vieja", chl_pred=999.0),    # corrida ANTERIOR (ordena antes que "r")
    ], ignore_index=True)
    detail, summary = VF.verify(dup, TARGET, THR)
    assert set(detail["horizon"]) == {1, 3}              # sigue habiendo 1 fila por horizonte
    s1 = summary[summary.horizon == 1].iloc[0]
    assert s1["n"] == 1                                  # no 2
    d1 = detail[detail.horizon == 1].iloc[0]
    assert d1["chl_pred"] == 10.0                        # se quedo con la corrida reciente ("r")


def test_far_con_falsa_alarma():
    """Una alerta emitida sobre un no-evento debe dar FAR=1 y POD=NaN (no hubo eventos).
    Se evalua el flag realmente emitido, no una politica reconstruida a posteriori."""
    import pandas as pd
    log = LOG.copy()
    log.loc[0, "riesgo"] = True                    # alerta guardada; real=12 < 25 -> falsa
    _, summary = VF.verify(log, TARGET, THR)
    s1 = summary[summary.horizon == 1].iloc[0]
    assert s1["alertas_emitidas"] == 1 and s1["eventos_reales"] == 0
    assert s1["FAR"] == 1.0 and s1["precision"] == 0.0
    assert pd.isna(s1["POD"])


def test_fuera_de_banda_se_detecta():
    log = LOG.copy()
    log.loc[0, "p90"] = 11.0                       # ahora 12 > 11 -> fuera de banda
    detail, _ = VF.verify(log, TARGET, THR)
    h1 = detail[detail.horizon == 1].iloc[0]
    assert bool(h1["in_band"]) is False
