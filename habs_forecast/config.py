"""
config.py — Configuración central del sistema de predicción temprana de HABs (0-7 días).

Decisiones de diseño (cerradas en Fase 1, ver README.md):
  - Problema: PRONÓSTICO causal a 0-7 d (no nowcast). Features con marca temporal <= t;
    objetivo = estado del bloom en t+h. Nunca la imagen de t+h como feature.
  - Estado del bloom = agregado por ESCENA (no por muestra in-situ), para densificar la serie.
  - Etiqueta de validación final: clorofila in-situ donde exista; estado satelital para entrenar.
  - Dos grupos ECOLÓGICOS (no ópticos): dulce (cianoHAB) y marino/estuarino (dinoflagelados).
  - Salida híbrida: regresión log-chl + probabilidad de exceedancia (alerta) + clase ordinal.
  - MODELOS SEPARADOS por horizonte (h=0,1,3,5,7), decisión del usuario 2026-06-24:
    con N pequeño son más fáciles de validar y permiten arrancar con los horizontes que
    tengan datos. (El flag MULTI_HORIZON_SINGLE_MODEL permite volver al modelo único.)
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

# Calibracion externa del target: se congela antes del periodo evaluado.
TARGET_CALIBRATION_END = "2023-12-31"

# Frescura maxima compartida por entrenamiento e inferencia.
MAX_TARGET_AGE_DAYS = 14
MAX_ERA5_AGE_DAYS = 10
MAX_NUTRIENT_AGE_DAYS = 45
MAX_WATERQUALITY_AGE_DAYS = 14

# --------------------------------------------------------------------------------------
# Cuerpos de agua y agrupación ecológica
#   Justificación: agua dulce -> cianobacterias, limitación por P, residencia larga.
#                  marino/estuarino -> dinoflagelados, control por salinidad/mareas.
#   La separación NO es óptica (ambos son aguas Caso 2) sino biogeoquímica.
# --------------------------------------------------------------------------------------
# Nombre de carpeta en imágenes/  ->  (clave canónica, grupo ecológico)
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
# Horizontes de predicción (días). Modelo único multi-horizonte.
#   Bineado por gap real entre escenas: una escena t se empareja con escenas futuras
#   cuyo gap caiga en la tolerancia del horizonte.
# --------------------------------------------------------------------------------------
HORIZONS = [0, 1, 3, 5, 7]
# Tolerancia +/- días para asignar un par (t -> t+gap) a un horizonte nominal.
HORIZON_TOLERANCE = {0: (0, 0), 1: (1, 2), 3: (3, 4), 5: (5, 6), 7: (7, 8)}
# Estrategia de modelado: separados por horizonte (decisión usuario). True => modelo único.
MULTI_HORIZON_SINGLE_MODEL = False

# --------------------------------------------------------------------------------------
# Bandas y construcción del estado del bloom (Módulo A -> agregado por escena)
# --------------------------------------------------------------------------------------
S2_BANDS = ["B2", "B3", "B4", "B5", "B8"]          # azul, verde, rojo, red-edge, NIR
BAND_SCALE_THRESHOLD = 1.5                          # si max>1.5 -> reflectancia*10000, dividir
SCENE_SUBSAMPLE = 4                                 # submuestreo espacial al agregar escena

# Máscara de agua / control de calidad (consistente con el pipeline previo)
NDWI_MIN = -0.5          # excluir vegetación/tierra
NDVI_MAX = 0.4           # excluir vegetación terrestre dominante
MIN_WATER_PIXELS = 50    # escenas con menos píxeles de agua válidos se descartan

# --- Rechazo de nube / bruma (SCL no atrapa nubes finas ni bordes; sin esto la
#     mediana de escena se contamina y el modelo lee la nube como floración) ---
# Agua limpia tiene azul B2 ~0.04 (<0.10); una nube es brillante en TODO el visible
# (min alto) y PLANA/blanca (B2~B3, ratio ~0.99). La nata algal es VERDE (B2 << B3,
# ratio <0.7): falla el test de nube y se conserva. Verificado en scene_state.csv:
# las 99 escenas con B2>0.15 tienen B2/B3=0.99 (100% nube), no floraciones.
MIN_VIS_CLOUD = 0.10     # si min(B2,B3,B4) supera esto -> píxel brillante (nube/bruma)
CLOUD_WHITE_RATIO = 0.80 # y además B2 >= ratio*B3 (espectro plano/blanco) -> nube


def water_mask(B2, B3, B4, B5, B8, eps=1e-10):
    """Máscara booleana de agua válida con rechazo de nube/bruma (fuente única de
    verdad para build_scene_state y make_maps). Entrada: reflectancias 0-1 por píxel."""
    import numpy as _np
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    valid = (B2 > 0) & (B3 > 0) & (B4 > 0) & (B5 > 0) & (B8 > 0)
    vis_min = _np.minimum(_np.minimum(B2, B3), B4)
    cloud = (vis_min > MIN_VIS_CLOUD) & (B2 >= CLOUD_WHITE_RATIO * B3)
    return valid & (ndwi > NDWI_MIN) & (ndvi < NDVI_MAX) & ~cloud

# --------------------------------------------------------------------------------------
# Índices espectrales: SOLO los justificados para HAB (ver Fase 1).
#   Conservar: NDCI (principal), FAI (nata flotante), CI red-edge, turbidez (control).
#   Descartar como predictor: NDVI (solo filtro), NDWI/MNDWI (solo máscara), SWIR (no in-water).
# --------------------------------------------------------------------------------------
def spectral_indices(B2, B3, B4, B5, B8, eps=1e-10):
    """Índices justificados para detección/predicción de HAB. Entrada: reflectancias 0-1."""
    return {
        "NDCI":     (B5 - B4) / (B5 + B4 + eps),                       # clorofila red-edge (principal)
        "CI_red":   (B5 / (B4 + eps)) - 1.0,                           # chlorophyll index red-edge
        "FAI":      B8 - (B4 + (B5 - B4) * (833 - 665) / (705 - 665)), # algas flotantes / nata
        "turbidity": B4 / (B3 + eps),                                  # CONTROL de sedimento (confunde NDCI)
        # auxiliares solo para enmascarado / QA (no entran como predictor del bloom):
        "NDVI":     (B8 - B4) / (B8 + B4 + eps),
        "NDWI":     (B3 - B8) / (B3 + B8 + eps),
    }

# Índices que SÍ son features predictivas del estado del bloom
STATE_SPECTRAL_FEATURES = ["B2", "B3", "B4", "B5", "B8", "NDCI", "CI_red", "FAI", "turbidity"]
# Índices reservados solo a control de calidad / máscara (NO predictores)
QA_ONLY_INDICES = ["NDVI", "NDWI"]

# --------------------------------------------------------------------------------------
# Variables ERA5 (drivers dinámicos del pronóstico).
#   Usar componentes u/v del viento (no dirección angular). Presión: marginal (opcional).
#   Drivers actúan por ACUMULACIÓN/retardo -> se construyen lags/rolling causales aguas abajo.
# --------------------------------------------------------------------------------------
ERA5_VARS = [
    "temp_air_2m",       # crecimiento algal, estratificación (preferible LSWT si disponible)
    "solar_radiation",   # fotosíntesis / estratificación
    "precipitation",     # pulsos de nutrientes (efecto retardado) vs flushing
    "wind_speed_10m",    # mezcla / acumulación de nata (no lineal)
    "wind_u_10m",        # advección (componente, no ángulo)
    "wind_v_10m",
    "surface_pressure",  # marginal; régimen sinóptico
]
# Valores promedio de respaldo SOLO para imputación de emergencia (no para "predecir").
ERA5_FALLBACK_MEAN = {
    "temp_air_2m": 295.0, "solar_radiation": 18000000.0, "precipitation": 0.003,
    "wind_speed_10m": 3.5, "wind_u_10m": -0.5, "wind_v_10m": 1.2, "surface_pressure": 1013.25,
}

# Ventanas causales (solo pasado) para drivers ERA5 acumulados/medios
ERA5_ROLLING_WINDOWS = [3, 7, 14, 30]   # dias

# --------------------------------------------------------------------------------------
# Variables de calidad del agua (CONTEXTO de baja frecuencia, no driver diario).
#   Para 0-7 d los nutrientes fijan la susceptibilidad, no la dinámica.
#   N:P como rasgo estructural; Secchi/TSS como confundidores ópticos.
# --------------------------------------------------------------------------------------
WATERQUALITY_CONTEXT = ["TP", "TN", "NP_ratio", "secchi"]   # se unen como contexto estatico/estacional

# --------------------------------------------------------------------------------------
# Objetivo (salida hibrida)
# --------------------------------------------------------------------------------------
CHL_COL = "clorofila_ugl"
# Umbrales de intensidad anclados a guías (cianobacterias / estado trófico). Ajustables.
#   moderado ~ eutrófico; severo ~ alerta sanitaria recreativa.
THRESHOLDS = {"moderate": 10.0, "severe": 24.0}   # ug/L de chl-a
LOG_CHL_EPS = 1e-3                                  # para log1p estable

# Umbral de alerta RELATIVO por ecosistema: una "floracion" se define como exceder el
# percentil de la climatología local de cada cuerpo (anomalía), no un corte absoluto.
#   Justificación: un bloom en agua costera oligotrófica != bloom en lago eutrófico.
#   Hace evaluables (Recall/PR-AUC) tanto lagos como costa. Es definición de etiqueta,
#   no entra al modelo. Se calcula sobre toda la serie target del cuerpo (climatología).
USE_RELATIVE_THRESHOLD = True
RELATIVE_PERCENTILE = 85          # top 15% del cuerpo = evento de alerta

def chl_to_class(chl):
    """0 = sin floración, 1 = moderada, 2 = severa."""
    if chl is None:
        return None
    if chl >= THRESHOLDS["severe"]:
        return 2
    if chl >= THRESHOLDS["moderate"]:
        return 1
    return 0


def alert_threshold_ugl(thr_relative):
    """Umbral OPERATIVO de floración (ug/L) = el relativo del cuerpo (p85) ACOTADO por el
    nivel biológico absoluto 'severe'. Así un cuerpo hipereutrófico (Cajon p85=64, Yojoa,
    Okeechobee) no exige niveles absurdos: una floración real (>=24 ug/L) SIEMPRE dispara,
    mientras que en agua oligotrófica se mantiene la sensibilidad relativa (p85 < 24)."""
    try:
        return float(min(float(thr_relative), THRESHOLDS["severe"]))
    except (TypeError, ValueError):
        return float(THRESHOLDS["severe"])


def elevated_threshold_ugl(thr_floracion):
    """Umbral de 'biomasa elevada' (banda de aviso por DEBAJO de la floración). Se define
    relativo al umbral de floración del cuerpo (60%), sin pasar del nivel eutrófico de 10 ug/L.
    Garantiza el orden correcto SIEMPRE: elevada < floracion, tambien en cuerpos marinos/
    oligotroficos donde el umbral de floracion (p85) ya es bajo (p.ej. 5.7 -> elevada 3.4)."""
    return float(min(THRESHOLDS["moderate"], 0.6 * float(thr_floracion)))


def biomass_level(chl, thr_floracion, thr_elevada=None):
    """Nivel de biomasa en 3 grados, consistente entre mapa, app y validación:
    'floracion' (>= umbral del cuerpo) · 'elevada' (>= umbral de aviso) · 'normal'."""
    if chl is None:
        return None
    if thr_elevada is None:
        thr_elevada = elevated_threshold_ugl(thr_floracion)
    if chl >= thr_floracion:
        return "floracion"
    if chl >= thr_elevada:
        return "elevada"
    return "normal"


# etiquetas de presentación (mapa / app)
LEVEL_ES = {"floracion": "FLORACIÓN", "elevada": "BIOMASA ELEVADA", "normal": "NORMAL"}

# --------------------------------------------------------------------------------------
# Validacion
# --------------------------------------------------------------------------------------
VALIDATION = {
    "scheme": "walk_forward",          # titular: ventana expansiva (realismo operativo)
    "blocked_test_year": 2026,         # split temporal por bloques (test = año más reciente)
    "purge_days": 8,                   # embargo >= horizonte máximo para evitar fuga
    "lowbo_within_group": True,        # Leave-One-Water-Body-Out solo dentro del grupo ecológico
}
# Baselines obligatorios que el modelo debe superar
BASELINES = ["persistence", "climatology"]

# --------------------------------------------------------------------------------------
# Capa OPERATIVA (alerta en producción): guardas de frescura/cobertura y confianza.
#   No afectan el modelado ni los números de validación: gobiernan CÓMO se reporta cada
#   pronóstico operativo (run_forecast.py / predict.py via guards.py).
# --------------------------------------------------------------------------------------
DIR_FORECASTS = os.path.join(DIR_OUT, "forecasts")   # snapshots + bitácora de pronósticos
os.makedirs(DIR_FORECASTS, exist_ok=True)

MAX_DATA_AGE_DAYS = 14            # escena t0 más vieja que esto -> confianza STALE (dato viejo)
EXPLORATORY_BODIES = ["cajon"]   # cuerpos en estado exploratorio (sin validación suficiente)
# Severidad de las guardas, de PEOR a mejor: 'confianza' toma la PEOR condición aplicable.
#   LOW_COVERAGE (la escena casi no tiene agua válida; no fiable) es lo más grave; luego STALE
#   (dato desactualizado); luego EXPLORATORIO (cuerpo sin verdad de campo); OK = sin reparos.
CONFIDENCE_SEVERITY = [
    "STALE_TARGET", "LOW_COVERAGE", "STALE", "MISSING_CONTEXT",
    "EXPLORATORIO", "OK",
]
