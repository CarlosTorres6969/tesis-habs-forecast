"""
config.py — Configuracion central del sistema de prediccion temprana de HABs (0-7 dias).

Decisiones de diseno (cerradas en Fase 1, ver README.md):
  - Problema: PRONOSTICO causal a 0-7 d (no nowcast). Features con marca temporal <= t;
    objetivo = estado del bloom en t+h. Nunca la imagen de t+h como feature.
  - Estado del bloom = agregado por ESCENA (no por muestra in-situ), para densificar la serie.
  - Etiqueta de validacion final: clorofila in-situ donde exista; estado satelital para entrenar.
  - Dos grupos ECOLOGICOS (no opticos): dulce (cianoHAB) y marino/estuarino (dinoflagelados).
  - Salida hibrida: regresion log-chl + probabilidad de exceedancia (alerta) + clase ordinal.
  - MODELOS SEPARADOS por horizonte (h=0,1,3,5,7), decision del usuario 2026-06-24:
    con N pequeno son mas faciles de validar y permiten arrancar con los horizontes que
    tengan datos. (El flag MULTI_HORIZON_SINGLE_MODEL permite volver al modelo unico.)
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
BASE = os.environ.get("HABS_BASE", r"C:\Users\JC\Desktop\Tesis")
DIR_IMAGENES   = os.path.join(BASE, "imagenes")
DIR_DATASETS   = os.path.join(BASE, "datasets")
DIR_ERA5_NC    = os.path.join(BASE, "era5_temp_nc")
DIR_MAPAS      = os.path.join(BASE, "mapas_finales")

# Salidas del nuevo pipeline (aisladas, no tocan el material existente)
DIR_OUT        = os.path.join(BASE, "habs_forecast", "artifacts")
DIR_STATE      = os.path.join(DIR_OUT, "state_series")   # serie de estado por escena
DIR_PAIRS      = os.path.join(DIR_OUT, "pairs")          # pares causales multi-horizonte
DIR_MODELS     = os.path.join(DIR_OUT, "models")
DIR_REPORTS    = os.path.join(DIR_OUT, "reports")
for _d in (DIR_OUT, DIR_STATE, DIR_PAIRS, DIR_MODELS, DIR_REPORTS):
    os.makedirs(_d, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Cuerpos de agua y agrupacion ecologica
#   Justificacion: agua dulce -> cianobacterias, limitacion por P, residencia larga.
#                  marino/estuarino -> dinoflagelados, control por salinidad/mareas.
#   La separacion NO es optica (ambos son aguas Caso 2) sino biogeoquimica.
# --------------------------------------------------------------------------------------
# Nombre de carpeta en imagenes/  ->  (clave canonica, grupo ecologico)
REGIONS = {
    "Okeechobee":     {"key": "okeechobee", "group": "freshwater", "country": "USA"},
    "TampaBay":       {"key": "tampa_bay",  "group": "marine",     "country": "USA"},
    "Cajon":          {"key": "cajon",      "group": "freshwater", "country": "HND"},
    "Golfo_Fonseca":  {"key": "fonseca",    "group": "marine",     "country": "HND"},
    "Lago de Yojoa":  {"key": "yojoa",      "group": "freshwater", "country": "HND"},
}
GROUPS = ("freshwater", "marine")
FRESHWATER = [r["key"] for r in REGIONS.values() if r["group"] == "freshwater"]
MARINE     = [r["key"] for r in REGIONS.values() if r["group"] == "marine"]

# --------------------------------------------------------------------------------------
# Horizontes de prediccion (dias). Modelo unico multi-horizonte.
#   Bineado por gap real entre escenas: una escena t se empareja con escenas futuras
#   cuyo gap caiga en la tolerancia del horizonte.
# --------------------------------------------------------------------------------------
HORIZONS = [0, 1, 3, 5, 7]
# Tolerancia +/- dias para asignar un par (t -> t+gap) a un horizonte nominal.
HORIZON_TOLERANCE = {0: (0, 0), 1: (1, 2), 3: (3, 4), 5: (5, 6), 7: (7, 8)}
# Estrategia de modelado: separados por horizonte (decision usuario). True => modelo unico.
MULTI_HORIZON_SINGLE_MODEL = False

# --------------------------------------------------------------------------------------
# Bandas y construccion del estado del bloom (Modulo A -> agregado por escena)
# --------------------------------------------------------------------------------------
S2_BANDS = ["B2", "B3", "B4", "B5", "B8"]          # azul, verde, rojo, red-edge, NIR
BAND_SCALE_THRESHOLD = 1.5                          # si max>1.5 -> reflectancia*10000, dividir
SCENE_SUBSAMPLE = 4                                 # submuestreo espacial al agregar escena

# Mascara de agua / control de calidad (consistente con el pipeline previo)
NDWI_MIN = -0.5          # excluir vegetacion/tierra
NDVI_MAX = 0.4           # excluir vegetacion terrestre dominante
MIN_WATER_PIXELS = 50    # escenas con menos pixeles de agua validos se descartan

# --------------------------------------------------------------------------------------
# Indices espectrales: SOLO los justificados para HAB (ver Fase 1).
#   Conservar: NDCI (principal), FAI (nata flotante), CI red-edge, turbidez (control).
#   Descartar como predictor: NDVI (solo filtro), NDWI/MNDWI (solo mascara), SWIR (no in-water).
# --------------------------------------------------------------------------------------
def spectral_indices(B2, B3, B4, B5, B8, eps=1e-10):
    """Indices justificados para deteccion/prediccion de HAB. Entrada: reflectancias 0-1."""
    return {
        "NDCI":     (B5 - B4) / (B5 + B4 + eps),                       # clorofila red-edge (principal)
        "CI_red":   (B5 / (B4 + eps)) - 1.0,                           # chlorophyll index red-edge
        "FAI":      B8 - (B4 + (B5 - B4) * (833 - 665) / (705 - 665)), # algas flotantes / nata
        "turbidity": B4 / (B3 + eps),                                  # CONTROL de sedimento (confunde NDCI)
        # auxiliares solo para enmascarado / QA (no entran como predictor del bloom):
        "NDVI":     (B8 - B4) / (B8 + B4 + eps),
        "NDWI":     (B3 - B8) / (B3 + B8 + eps),
    }

# Indices que SI son features predictivas del estado del bloom
STATE_SPECTRAL_FEATURES = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
# Indices reservados solo a control de calidad / mascara (NO predictores)
QA_ONLY_INDICES = ["NDVI", "NDWI"]

# --------------------------------------------------------------------------------------
# Variables ERA5 (drivers dinamicos del pronostico).
#   Usar componentes u/v del viento (no direccion angular). Presion: marginal (opcional).
#   Drivers actuan por ACUMULACION/retardo -> se construyen lags/rolling causales aguas abajo.
# --------------------------------------------------------------------------------------
ERA5_VARS = [
    "temp_air_2m",       # crecimiento algal, estratificacion (preferible LSWT si disponible)
    "solar_radiation",   # fotosintesis / estratificacion
    "precipitation",     # pulsos de nutrientes (efecto retardado) vs flushing
    "wind_speed_10m",    # mezcla / acumulacion de nata (no lineal)
    "wind_u_10m",        # adveccion (componente, no angulo)
    "wind_v_10m",
    "surface_pressure",  # marginal; regimen sinoptico
]
# Valores promedio de respaldo SOLO para imputacion de emergencia (no para "predecir").
ERA5_FALLBACK_MEAN = {
    "temp_air_2m": 295.0, "solar_radiation": 18000000.0, "precipitation": 0.003,
    "wind_speed_10m": 3.5, "wind_u_10m": -0.5, "wind_v_10m": 1.2, "surface_pressure": 1013.25,
}

# Ventanas causales (solo pasado) para drivers ERA5 acumulados/medios
ERA5_ROLLING_WINDOWS = [3, 7, 14, 30]   # dias

# --------------------------------------------------------------------------------------
# Variables de calidad del agua (CONTEXTO de baja frecuencia, no driver diario).
#   Para 0-7 d los nutrientes fijan la susceptibilidad, no la dinamica.
#   N:P como rasgo estructural; Secchi/TSS como confundidores opticos.
# --------------------------------------------------------------------------------------
WATERQUALITY_CONTEXT = ["TP", "TN", "NP_ratio", "secchi"]   # se unen como contexto estatico/estacional

# --------------------------------------------------------------------------------------
# Objetivo (salida hibrida)
# --------------------------------------------------------------------------------------
CHL_COL = "clorofila_ugl"
# Umbrales de intensidad anclados a guias (cianobacterias / estado trofico). Ajustables.
#   moderado ~ eutrofico; severo ~ alerta sanitaria recreativa.
THRESHOLDS = {"moderate": 10.0, "severe": 24.0}   # ug/L de chl-a
LOG_CHL_EPS = 1e-3                                  # para log1p estable

# Umbral de alerta RELATIVO por ecosistema: una "floracion" se define como exceder el
# percentil de la climatologia local de cada cuerpo (anomalia), no un corte absoluto.
#   Justificacion: un bloom en agua costera oligotrofica != bloom en lago eutrofico.
#   Hace evaluables (Recall/PR-AUC) tanto lagos como costa. Es definicion de etiqueta,
#   no entra al modelo. Se calcula sobre toda la serie target del cuerpo (climatologia).
USE_RELATIVE_THRESHOLD = True
RELATIVE_PERCENTILE = 85          # top 15% del cuerpo = evento de alerta

def chl_to_class(chl):
    """0 = sin floracion, 1 = moderada, 2 = severa."""
    if chl is None:
        return None
    if chl >= THRESHOLDS["severe"]:
        return 2
    if chl >= THRESHOLDS["moderate"]:
        return 1
    return 0


def alert_threshold_ugl(thr_relative):
    """Umbral OPERATIVO de floracion (ug/L) = el relativo del cuerpo (p85) ACOTADO por el
    nivel biologico absoluto 'severe'. Asi un cuerpo hipereutrofico (Cajon p85=64, Yojoa,
    Okeechobee) no exige niveles absurdos: una floracion real (>=24 ug/L) SIEMPRE dispara,
    mientras que en agua oligotrofica se mantiene la sensibilidad relativa (p85 < 24)."""
    try:
        return float(min(float(thr_relative), THRESHOLDS["severe"]))
    except (TypeError, ValueError):
        return float(THRESHOLDS["severe"])


def biomass_level(chl, thr_floracion):
    """Nivel de biomasa en 3 grados, consistente entre mapa, app y validacion:
    'floracion' (>= umbral del cuerpo, max 24) · 'elevada' (>= 10) · 'normal' (< 10)."""
    if chl is None:
        return None
    if chl >= thr_floracion:
        return "floracion"
    if chl >= THRESHOLDS["moderate"]:
        return "elevada"
    return "normal"


# etiquetas de presentacion (mapa / app)
LEVEL_ES = {"floracion": "FLORACION", "elevada": "BIOMASA ELEVADA", "normal": "NORMAL"}

# --------------------------------------------------------------------------------------
# Validacion
# --------------------------------------------------------------------------------------
VALIDATION = {
    "scheme": "walk_forward",          # titular: ventana expansiva (realismo operativo)
    "blocked_test_year": 2026,         # split temporal por bloques (test = ano mas reciente)
    "purge_days": 8,                   # embargo >= horizonte maximo para evitar fuga
    "lowbo_within_group": True,        # Leave-One-Water-Body-Out solo dentro del grupo ecologico
}
# Baselines obligatorios que el modelo debe superar
BASELINES = ["persistence", "climatology"]

# --------------------------------------------------------------------------------------
# Capa OPERATIVA (alerta en produccion): guardas de frescura/cobertura y confianza.
#   No afectan el modelado ni los numeros de validacion: gobiernan COMO se reporta cada
#   pronostico operativo (run_forecast.py / predict.py via guards.py).
# --------------------------------------------------------------------------------------
DIR_FORECASTS = os.path.join(DIR_OUT, "forecasts")   # snapshots + bitacora de pronosticos
os.makedirs(DIR_FORECASTS, exist_ok=True)

MAX_DATA_AGE_DAYS = 14            # escena t0 mas vieja que esto -> confianza STALE (dato viejo)
EXPLORATORY_BODIES = ["cajon"]   # cuerpos en estado exploratorio (sin validacion suficiente)
# Severidad de las guardas, de PEOR a mejor: 'confianza' toma la PEOR condicion aplicable.
#   LOW_COVERAGE (la escena casi no tiene agua valida; no fiable) es lo mas grave; luego STALE
#   (dato desactualizado); luego EXPLORATORIO (cuerpo sin verdad de campo); OK = sin reparos.
CONFIDENCE_SEVERITY = ["LOW_COVERAGE", "STALE", "EXPLORATORIO", "OK"]
