"""test_run_forecast_schema.py — el esquema de salida de run_forecast.build_rows es estable
y correcto, alimentando un pronostico SINTETICO (sin tocar modelos ni disco)."""
import json
import pandas as pd
import run_forecast as RF

RUN_ISO = "2026-06-26 10:00:00"

# pronostico sintetico con la forma que devuelve predict.forecast_body
FC = {
    "water_body": "okeechobee", "group": "freshwater",
    "t0": pd.Timestamp("2026-06-20"), "chl0": 12.0, "thr_body": 42.8,
    "n_water_px": 5000, "alert_threshold": 0.07,
    "horizons": [
        {"horizon": 1, "chl_pred": 18.2, "p10": 3.5, "p90": 53.0, "prob_riesgo": 0.06, "riesgo": False},
        {"horizon": 7, "chl_pred": 8.4, "p10": None, "p90": None, "prob_riesgo": 0.51, "riesgo": True},
    ],
}
CARDS = {"freshwater_h1": {"commit_git": "abc1234", "fecha_entrenamiento": "2026-06-25",
                           "n_pares": 529, "skill_validado": [0.23, 0.14, 0.31]}}


def test_columnas_exactas_y_orden():
    rows = RF.build_rows(FC, RUN_ISO, CARDS, run_ts_for_age=pd.Timestamp("2026-06-26"))
    assert len(rows) == 2
    for r in rows:
        assert list(r.keys()) == RF.SCHEMA          # mismas columnas, mismo orden


def test_tipos_y_valores():
    rows = RF.build_rows(FC, RUN_ISO, CARDS, run_ts_for_age=pd.Timestamp("2026-06-26"))
    r0 = rows[0]
    assert isinstance(r0["riesgo"], bool) and r0["riesgo"] is False
    assert isinstance(r0["horizon"], int) and r0["horizon"] == 1
    assert r0["t0"] == "2026-06-20"
    assert r0["data_age_days"] == 6
    assert r0["confianza"] == "OK"
    # modelo_meta es JSON valido con el commit de la model card
    meta = json.loads(r0["modelo_meta"])
    assert meta["commit_git"] == "abc1234" and meta["n_pares"] == 529


def test_banda_nula_se_propaga_como_none():
    rows = RF.build_rows(FC, RUN_ISO, CARDS, run_ts_for_age=pd.Timestamp("2026-06-26"))
    assert rows[1]["p10"] is None and rows[1]["p90"] is None
    assert rows[1]["riesgo"] is True


def test_dataframe_respeta_esquema():
    rows = RF.build_rows(FC, RUN_ISO, CARDS, run_ts_for_age=pd.Timestamp("2026-06-26"))
    df = pd.DataFrame(rows, columns=RF.SCHEMA)
    assert list(df.columns) == RF.SCHEMA and len(df) == 2
