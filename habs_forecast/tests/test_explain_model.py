"""test_explain_model.py — la importancia SHAP es coherente con un modelo de juguete.

Contrato de shap_importance: sobre un XGBoost donde la salida depende SOLO de x0, esa feature
debe quedar primera (mayor mean_abs_shap) y las irrelevantes ultimas. Si falta shap o xgboost
(p.ej. CI minima) se OMITE, no falla.
"""
import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")
xgboost = pytest.importorskip("xgboost")

from explain_model import shap_importance


def _toy_model():
    """y = 3*x0 + ruido; x1,x2 irrelevantes. El modelo debe apoyarse en x0."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["x0", "x1", "x2"])
    y = 3.0 * X["x0"].values + 0.01 * rng.normal(size=300)
    reg = xgboost.XGBRegressor(n_estimators=60, max_depth=3, random_state=0).fit(X, y)
    return reg, X


def test_importancia_ordenada_y_bien_formada():
    reg, X = _toy_model()
    imp = shap_importance(reg, X)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert set(imp["feature"]) == {"x0", "x1", "x2"}
    assert (imp["mean_abs_shap"] >= 0).all()                 # magnitudes no negativas
    assert imp.iloc[0]["feature"] == "x0"                    # la feature que manda queda primera
    top = imp.set_index("feature")["mean_abs_shap"]
    assert top["x0"] > top["x1"] and top["x0"] > top["x2"]   # domina a las irrelevantes


def test_orden_descendente():
    reg, X = _toy_model()
    imp = shap_importance(reg, X)
    vals = imp["mean_abs_shap"].values
    assert (np.diff(vals) <= 1e-9).all()                     # ordenado desc
