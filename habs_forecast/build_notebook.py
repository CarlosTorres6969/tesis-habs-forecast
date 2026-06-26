"""
build_notebook.py — Genera el notebook reproducible Modelo_HABs_limpio.ipynb (anexo de tesis).

Walkthrough honesto del pipeline (reemplaza los .ipynb viejos con fuga): carga los pares causales,
corre el test de integridad, muestra la validacion anidada / intervalos / sensibilidad ERA5 desde
los reportes ya generados, y un ejemplo de prediccion desplegable. Llama a los modulos existentes
(no reescribe logica). Ejecutar con: python -m nbconvert --execute --to notebook ...
"""
from __future__ import annotations
import os
import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelo_HABs_limpio.ipynb")
nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Predicción temprana de riesgo de biomasa algal (HABs) a 0–7 días
### Notebook reproducible — sistema causal sin fuga

Este notebook reemplaza los notebooks antiguos (que tenían **fuga de datos**: AUC≈1.0 falso por
target derivado de las mismas bandas + validación con *shuffle*). Aquí el problema es **pronóstico
causal** X(≤t₀) → clorofila-a(t₀+h), con validación temporal honesta.

> El sistema es una **herramienta de alerta temprana** de condiciones de riesgo (biomasa /
> clorofila-a elevada), **no** un detector certero de toxicidad: la confirmación de nocividad
> requiere verificación de campo (cianobacterias, toxinas).

**Datos:** Sentinel-2 (predictor) + target satelital independiente — VIIRS (lagos) y **Sentinel-3
OLCI 300 m** (costa, hasta jun 2026) — + ERA5 + in-situ (fósforo, calidad de agua). Ventana 2023–2026.
**Reproducible:** los pares se guardan en orden canónico, así dos corridas dan resultados idénticos
(clave para una defensa honesta, a diferencia del sistema viejo con fuga).""")

md("## 0. Configuración e imports")
code("""import os, sys, json, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import config as C
from train import FEATURES, SPECTRAL, AUTOREG, ERA5, NUTRIENTS, WATERQUAL, PAIRS, _model
print("Horizontes:", [h for h in C.HORIZONS if h != 0], "| grupos:", C.GROUPS)""")

md("""## 1. Pares causales (predictor en t₀ → target en t₀+h)
Cada fila empareja el **estado espectral Sentinel-2 en t₀** (+ clorofila reciente, ERA5, in-situ)
con la **clorofila-a satelital en t₀+h** (VIIRS/OLCI, sensor independiente → rompe la circularidad).
El target en t₀+h **nunca** es predictor.""")
code("""df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])
print(f"Pares: {len(df)} | cuerpos: {sorted(df.water_body.unique())}")
display(df.groupby(["group","water_body"]).size().rename("pares").to_frame())
# causalidad: para h>0 el target es estrictamente futuro
viol = df[(df.horizon>0) & (df.fecha_target<=df.fecha_t0)]
print(f"Pares con fuga temporal (target<=t0, h>0): {len(viol)}  -> 0 = sin fuga")""")

md("## 2. Test de integridad (sin fuga / causal / consistente)\nSe ejecuta el script `check_integrity.py` que afirma **14 condiciones** de honestidad (11 de causalidad/sin-fuga + 3 de la capa operativa de alerta).")
code("""r = subprocess.run([sys.executable, "check_integrity.py"], capture_output=True, text=True)
print(r.stdout[-1200:])""")

md("""## 3. Familias de features (todas causales, ≤ t₀)
- **AUTOREG**: trayectoria reciente de clorofila (backbone).
- **SPECTRAL**: bandas/índices Sentinel-2 (NDCI, CI_red, FAI, turbidez).
- **ERA5**: meteorología (temp, radiación, precipitación, viento).
- **INSITU**: fósforo + calidad de agua (contexto de medio/largo plazo).""")
code("""for name, fam in [("AUTOREG",AUTOREG),("SPECTRAL",SPECTRAL),("ERA5",ERA5),
                  ("INSITU",NUTRIENTS+WATERQUAL)]:
    print(f"{name:9s}: {fam}")""")

md("""## 4. Validación anidada — TEST FINAL INTACTO
Se reserva el último ~25% del tiempo como test nunca tocado; la selección de features se hace
solo en DEV. Es el **número defendible**. `*` = IC95% bootstrap excluye 0 (skill significativo).""")
code("""rep = json.load(open(os.path.join(C.DIR_REPORTS,"nested_metrics.json")))
fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)
for ax,(grp,nm) in zip(axes, [("freshwater","Lagos"),("marine","Costa")]):
    hs=[1,3,5,7]; med=[rep[grp][str(h)]["skill_nested"][0] for h in hs]
    lo=[rep[grp][str(h)]["skill_nested"][1] for h in hs]; hi=[rep[grp][str(h)]["skill_nested"][2] for h in hs]
    err=[[m-l for m,l in zip(med,lo)],[h2-m for m,h2 in zip(med,hi)]]
    ax.bar([f"+{h}d" for h in hs], med, color="#2c7fb8", alpha=.8)
    ax.errorbar([f"+{h}d" for h in hs], med, yerr=err, fmt="none", ecolor="k", capsize=4)
    ax.axhline(0, color="grey", lw=.8); ax.set_title(f"{nm} — skill vs persistencia (test intacto)")
    ax.set_ylabel("skill score")
plt.tight_layout(); plt.show()
pd.DataFrame({grp:{f"+{h}d": f"{rep[grp][str(h)]['skill_nested'][0]:+.2f}" for h in [1,3,5,7]}
             for grp in ["freshwater","marine"]})""")

md("""## 5. Intervalos de incertidumbre (regresión cuantil conformalizada, CQR)
Cada pronóstico de intensidad lleva una banda **P10–P90**. Validada en el test intacto:
cobertura empírica cercana a **0.80** = intervalos calibrados (no sobreconfiados).""")
code("""iv = json.load(open(os.path.join(C.DIR_REPORTS,"interval_metrics.json")))
fig, ax = plt.subplots(figsize=(8,4))
for grp,nm,c in [("freshwater","Lagos","#2c7fb8"),("marine","Costa","#d95f0e")]:
    hs=[1,3,5,7]; cov=[iv[grp][str(h)]["cobertura_cqr"][0] for h in hs]
    ax.plot([f"+{h}d" for h in hs], cov, "o-", color=c, label=nm)
ax.axhline(0.80, ls="--", color="grey", label="nominal 0.80")
ax.set_ylim(0.5,1.0); ax.set_ylabel("cobertura empírica P10–P90"); ax.legend()
ax.set_title("Cobertura de los intervalos (CQR) en el test intacto"); plt.tight_layout(); plt.show()""")

md("## 6. Sensibilidad ERA5 (reanálisis vs pronóstico)\nAblación (aporte real de ERA5) y estrés de ruido. Aporte pequeño + curva plana ⇒ se puede operar con ERA5 de pronóstico sin perder skill.")
code("""er = json.load(open(os.path.join(C.DIR_REPORTS,"era5_sensitivity.json")))
rows=[]
for grp in ["freshwater","marine"]:
    for h in [1,3,5,7]:
        nd=er[grp][str(h)]
        rows.append({"grupo":grp,"h":f"+{h}d","skill_con_ERA5":round(nd["skill_con_era5"][0],3),
                     "aporte_ERA5":round(nd["aporte_era5"],3),
                     "skill_ruido_100%":round(nd["ruido_curva"]["1.0"][0],3)})
pd.DataFrame(rows)""")

md("""## 7. Predicción desplegable (ejemplo)
`predict.py` produce, para un cuerpo y fecha, la clorofila-a esperada por horizonte + **banda de
incertidumbre** + probabilidad de **riesgo** calibrada (sí/no).""")
code("""r = subprocess.run([sys.executable, "predict.py", "okeechobee"], capture_output=True, text=True)
print(r.stdout)""")

md("""## 8. Demostración en vivo: el modelo supera a la persistencia
Entrenamiento rápido (un cuerpo, walk-forward temporal) para ilustrar que el pronóstico tiene
skill real frente al baseline de persistencia (proyectar el último valor conocido).""")
code("""from sklearn.metrics import mean_squared_error
feats = [f for f in FEATURES if f in df.columns]
d = df[(df.water_body=="okeechobee") & (df.horizon==5)].sort_values("fecha_t0")
cut = d.fecha_t0.quantile(0.75)
tr, te = d[d.fecha_t0<=cut], d[d.fecha_t0>cut]
m = _model().fit(tr[feats], tr.log_chl_target)
rmse_model = np.sqrt(mean_squared_error(te.log_chl_target, m.predict(te[feats])))
rmse_persist = np.sqrt(mean_squared_error(te.log_chl_target, te.log_chl_t0))
print(f"Okeechobee +5d | RMSE(log) modelo={rmse_model:.3f}  persistencia={rmse_persist:.3f}")
print(f"Skill = {1 - rmse_model/rmse_persist:+.2f}  (>0 => mejor que persistencia)")""")

md("""## 9. Capa operativa de alerta
`forecast_body` (reusada por `predict.py` y `run_forecast.py`) emite el pronóstico estructurado por
horizonte. `guards.py` añade una etiqueta de **confianza** según la frescura/cobertura de la escena
(STALE / LOW_COVERAGE / EXPLORATORIO). `run_forecast.py` registra cada corrida en una bitácora y
`verify_forecasts.py` la verifica cuando el target real madura (MAE, cobertura de banda, hit-rate).""")
code("""from predict import forecast_body
import guards
fc = forecast_body("tampa_bay")                      # último t0 disponible (solo datos <= t0)
conf, flags, age = guards.evaluate_guards("tampa_bay", fc["t0"], fc["n_water_px"])
print(f"tampa_bay | t0={fc['t0'].date()} | edad={age}d | confianza={conf} {flags}")
display(pd.DataFrame(fc["horizons"])[["horizon","chl_pred","p10","p90","prob_riesgo","riesgo"]])""")

md("""## 10. Reproducibilidad
Los pares se guardan en **orden canónico** (`water_body, horizon, fecha_t0, fecha_target`). Esto evita
que el muestreo de XGBoost (`subsample`/`colsample` con semilla fija, que selecciona filas por
posición) cambie entre corridas. Resultado: la validación anidada es **idéntica corrida a corrida**.""")
code("""dfc = pd.read_csv(PAIRS)
canon = dfc.sort_values(["water_body","horizon","fecha_t0","fecha_target"]).reset_index(drop=True)
print("Pares en orden canónico (pipeline reproducible):", dfc.equals(canon))""")

md("""## Reproducir todo
```bash
pip install -r ../requirements.txt
python run_pipeline.py        # build_scene_state -> ... -> build_final_report
python check_integrity.py     # 14/14 OK
python run_forecast.py        # pronóstico operativo + bitácora
python verify_forecasts.py    # verificación cuando el target madura
```
Los datos pesados (rasters Sentinel-2, ERA5) se descargan con `fetch_*` / `ingest_*`
(ver `run_pipeline.py`). **Conclusión:** pronóstico causal 0–7 d **reproducible**, con skill
significativo en lagos (1/5/7 d) y **costa mejorada con OLCI fresco** (1/3/5 d), intervalos
calibrados y una capa operativa de alerta — herramienta de **riesgo**, no de confirmación de toxicidad.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook -> {OUT} ({len(C)} celdas)")
