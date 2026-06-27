"""
app.py — Interfaz Streamlit (demostracion desplegable y HONESTA) del sistema de pronostico
temprano de riesgo de biomasa algal (clorofila-a como proxy) a 0-7 dias.

NO implementa modelado: ENVUELVE la logica que ya existe.
  - mapas (satelital + biomasa prevista por pixel):  make_maps.build_map_figure
  - intensidad + banda P10-P90 + alerta calibrada:    predict.forecast_body
  - etiqueta de confianza (frescura/cobertura/estado): guards.evaluate_guards
Solo funciona para los 5 cuerpos validados (config.REGIONS) y con escenas Sentinel-2 de 5 bandas
(B2,B3,B4,B5,B8). Es PRONOSTICO a futuro, no deteccion sobre la misma imagen.

Correr (local, para la defensa):  streamlit run app.py
"""
from __future__ import annotations
import os, glob, re, tempfile, logging
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import joblib
import torch
import streamlit as st
import config as C
from predict import forecast_body, GROUP, SPEC, MODELS
from make_maps import build_map_figure, _scene_pixels, KEY2FOLDER
from train_nn import HABNet
import guards

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

# Metadatos por cuerpo (nombre legible, grupo, pais) desde config.REGIONS — solo los 5 validados
KEY2META = {meta["key"]: {"folder": folder, "group": meta["group"], "country": meta["country"]}
            for folder, meta in C.REGIONS.items()}
NICE = {"okeechobee": "Lago Okeechobee", "tampa_bay": "Bahia de Tampa",
        "cajon": "Embalse El Cajon", "fonseca": "Golfo de Fonseca", "yojoa": "Lago de Yojoa"}
GRP_ES = {"freshwater": "lago / agua dulce", "marine": "costa / marino-estuarino"}
PAIS_ES = {"USA": "Estados Unidos", "HND": "Honduras"}
DISCLAIMER = ("⚠️ **Proxy de biomasa algal (clorofila-a).** NO confirma toxicidad ni floracion "
              "nociva. Herramienta de **alerta temprana**; requiere **verificacion de campo** "
              "(identificacion de cianobacterias, toxinas).")

# --------------------------------------------------------------------------------------
# Tema visual (acuatico) — CSS + encabezado "hero". Solo presentacion, no toca la logica.
# --------------------------------------------------------------------------------------
THEME_CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px; }

/* Encabezado hero con degradado oceanico animado */
.hab-hero {
  position: relative; border-radius: 18px; padding: 2.0rem 2.2rem 2.7rem; margin-bottom: 1.0rem;
  overflow: hidden; color: #f7ffff;
  background: linear-gradient(120deg,#062b3f 0%,#0a6b6b 42%,#1aa39a 72%,#46c39b 100%);
  background-size: 220% 220%; animation: habflow 16s ease infinite;
  box-shadow: 0 10px 30px rgba(6,43,63,.28);
}
@keyframes habflow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hab-hero h1 { margin:0; font-size:2.0rem; font-weight:800; text-shadow:0 2px 10px rgba(0,0,0,.25); }
.hab-hero p { margin:.55rem 0 0; font-size:1.0rem; opacity:.96; max-width:64ch; }
.hab-tags { margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.hab-tag { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.30);
  padding:.28rem .7rem; border-radius:999px; font-size:.8rem; }
.hab-wave { position:absolute; left:0; right:0; bottom:-1px; line-height:0; }
.hab-wave svg { width:100%; height:44px; display:block; }

/* Botones con acento agua */
.stButton > button { border-radius:10px; font-weight:700; border:0; color:#04302f;
  background:linear-gradient(120deg,#0fa3a3,#46c39b); transition:transform .08s ease, box-shadow .15s ease; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(15,163,163,.35); color:#04302f; }

/* Metricas como tarjeta */
[data-testid="stMetric"] { background:#ffffff; border:1px solid #cfeae7; border-left:5px solid #0fa3a3;
  border-radius:12px; padding:.8rem 1rem; box-shadow:0 2px 8px rgba(6,43,63,.06); }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }

/* Encabezados y sidebar */
h2, h3 { color:#0a6b6b; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#e1f3f1 0%,#eef9f8 100%);
  border-right:1px solid #cfeae7; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#0a6b6b; }
</style>
"""

HERO_HTML = """
<div class="hab-hero">
  <h1>🌊 Alerta temprana de biomasa algal (HABs)</h1>
  <p>Pronostico de riesgo de floraciones algales a <b>0–7 dias</b>. La clorofila-a se usa como
     <b>proxy de biomasa</b>: la herramienta senala <b>riesgo</b>, no confirma toxicidad.</p>
  <div class="hab-tags">
    <span class="hab-tag">🛰️ Sentinel-2 (5 bandas)</span>
    <span class="hab-tag">💧 5 cuerpos validados</span>
    <span class="hab-tag">📈 Horizontes 0–7 dias</span>
    <span class="hab-tag">🧪 XGBoost + Red neuronal</span>
  </div>
  <div class="hab-wave">
    <svg viewBox="0 0 1440 60" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <path fill="#f2fbfa" d="M0,32 C240,64 480,4 720,24 C960,46 1200,64 1440,26 L1440,60 L0,60 Z"></path>
    </svg>
  </div>
</div>
"""


@st.cache_resource(show_spinner=False)
def load_resources():
    """Carga UNA sola vez (cacheada entre interacciones) los modelos de produccion:
    umbrales por cuerpo, calibradores de alerta, bundles XGBoost (+cuantiles CQR) y redes NN.
    Devuelve None si falta lo esencial (se avisa en la UI, no se truena)."""
    thr_path = os.path.join(MODELS, "thr_body.pkl")
    if not os.path.exists(thr_path):
        return None
    res = {"thr_body": joblib.load(thr_path), "calib": {}, "bundles": {}, "nn": {}}
    for group in ("freshwater", "marine"):
        cf = os.path.join(MODELS, f"alert_calib_{group}.pkl")
        res["calib"][group] = joblib.load(cf) if os.path.exists(cf) else None
        for h in (1, 3, 5, 7):
            f = os.path.join(MODELS, f"{group}_h{h}.pkl")
            nnf = os.path.join(MODELS, f"{group}_h{h}_nn.pt")
            if os.path.exists(f) and os.path.exists(nnf):
                b = joblib.load(f)
                net = HABNet(b["n_in"]); net.load_state_dict(torch.load(nnf)); net.eval()
                res["bundles"][(group, h)] = b
                res["nn"][(group, h)] = net
    return res


def list_example_scenes(wb):
    """Lista (fecha, ruta) de las escenas Sentinel-2 disponibles del cuerpo (mas recientes primero)."""
    folder = KEY2META[wb]["folder"]
    tifs = glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True)
    items = []
    for p in tifs:
        if os.path.basename(p).startswith("LS_"):           # Landsat: 4 bandas (sin B5/B8 red-edge)
            continue
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(p))
        items.append((m.group(1) if m else os.path.basename(p), p))
    return sorted(items, reverse=True)


def body_median_spectral(path):
    """Mediana espectral del agua de una escena externa (GeoTIFF subido) -> para forecast_body.
    Robusto: si el archivo no es un raster valido o no tiene 5 bandas, devuelve None (no trona)."""
    try:
        sp = _scene_pixels(path)                 # lanza si el archivo no es un raster valido
    except Exception as e:
        log.warning("GeoTIFF invalido: %s", e)
        return None
    if sp is None:                               # no tiene 5 bandas
        return None
    feats2d, water = sp
    if int(water.sum()) < 50:
        return "low_water"
    return {f: float(np.median(feats2d[f][water])) for f in SPEC}


# ----------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------
st.set_page_config(page_title="Alerta temprana de biomasa algal (HABs)", page_icon="🌊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(THEME_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("Como leer esta herramienta")
    st.markdown(
        "- **Es un PRONOSTICO a 0-7 dias**, no una deteccion sobre la imagen: estima la biomasa "
        "algal (clorofila-a) **a futuro** a partir del estado en t0.\n"
        "- **NO acepta fotos normales** (RGB de celular / capturas de Maps). Requiere una escena "
        "**Sentinel-2 de 5 bandas** (B2 azul, B3 verde, B4 rojo, **B5 red-edge**, **B8 NIR**): las "
        "bandas red-edge e infrarrojo son las que estiman clorofila; una foto comun no las tiene.\n"
        "- **Validado solo para 5 cuerpos** (abajo). Fuera de ellos no hay modelo ni calibracion.\n"
        "- **Clorofila-a = proxy de biomasa**, no de toxicidad. La alerta marca **riesgo** que "
        "amerita verificacion de campo.")
    st.divider()
    st.caption("Modelo: XGBoost (intensidad + intervalos CQR) + Red neuronal (alerta), por "
               "grupo ecologico y horizonte. Pronostico causal sin fuga (validacion anidada).")

st.markdown(HERO_HTML, unsafe_allow_html=True)
st.caption(DISCLAIMER)

# --- Selectores ---
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    wb = st.selectbox("Cuerpo de agua", list(KEY2META.keys()),
                      format_func=lambda k: NICE.get(k, k))
with c2:
    # +3 por defecto: h3/h5 usan senal espectral por pixel -> mapa con gradiente.
    # h1 y h7 son body-level -> el mapa de biomasa sale uniforme (sin detalle espacial).
    h = st.selectbox("Horizonte de pronostico", [1, 3, 5, 7], index=1, format_func=lambda x: f"+{x} dias")
    if h in (1, 7):
        st.caption("ℹ️ +1 y +7 dias son horizontes *body-level*: el modelo predice el NIVEL "
                   "del cuerpo, no por pixel. El mapa reparte ese nivel segun el patron "
                   "espacial ACTUAL (estimacion), no es un pronostico pixel-a-pixel.")
with c3:
    meta = KEY2META[wb]
    st.metric("Tipo", GRP_ES[meta["group"]].split(" / ")[0].capitalize())
    st.caption(f"Pais: {PAIS_ES.get(meta['country'], meta['country'])}")

# etiqueta de cuerpo exploratorio (no se oculta)
if wb in C.EXPLORATORY_BODIES:
    st.warning(f"🔬 **{NICE.get(wb, wb)} esta en estado EXPLORATORIO**: sin verdad de campo in-situ "
               "en la ventana 2023-2026 y con menos datos. Sus resultados son de **menor confianza**.")

# --- Entrada de escena ---
modo = st.radio("Escena Sentinel-2", ["Usar escena de ejemplo", "Subir GeoTIFF"], horizontal=True)
path, t0, spec_override, scene_err = None, None, None, None

if modo == "Usar escena de ejemplo":
    escenas = list_example_scenes(wb)
    if not escenas:
        scene_err = f"No hay escenas Sentinel-2 de ejemplo para {NICE.get(wb, wb)}."
    else:
        fechas = [e[0] for e in escenas]
        sel = st.selectbox(f"Escena disponible ({len(escenas)} fechas)", fechas)
        path = dict(escenas)[sel]
        t0 = pd.Timestamp(sel) if re.match(r"\d{4}-\d{2}-\d{2}", sel) else None
else:
    up = st.file_uploader("Sube un GeoTIFF Sentinel-2 de 5 bandas (orden B2,B3,B4,B5,B8)",
                          type=["tif", "tiff"])
    st.caption("Debe ser un raster georreferenciado de 5 bandas. Una foto RGB comun sera rechazada.")
    if up is not None:
        tmp = os.path.join(tempfile.gettempdir(), f"app_upload_{up.name}")
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        sm = body_median_spectral(tmp)
        if sm is None:
            scene_err = ("El archivo NO tiene 5 bandas validas (B2,B3,B4,B5,B8). "
                         "No es una escena Sentinel-2 valida — no se puede pronosticar.")
        elif sm == "low_water":
            scene_err = "La escena tiene muy pocos pixeles de agua validos para analizar."
        else:
            path, spec_override = tmp, sm
            # contexto NO espectral = ultima fecha disponible del cuerpo
            try:
                from predict import _load, SCENE
                sc = _load(SCENE, wb)
                t0 = sc["fecha"].max() if len(sc) else None
            except Exception:
                t0 = None
            st.info("Escena externa valida. El contexto no-espectral (clorofila reciente, ERA5, "
                    "in-situ) se toma de la ultima fecha disponible del cuerpo.")

if scene_err:
    st.error(scene_err)

# --- Analizar ---
disabled = path is None
if st.button("🔍 Analizar", type="primary", disabled=disabled):
    res = load_resources()
    if res is None:
        st.error("Faltan los modelos de produccion (artifacts/models/). Corre `python train_final.py`.")
        st.stop()
    if (GROUP[wb], h) not in res["bundles"]:
        st.error(f"No hay modelo entrenado para {NICE.get(wb, wb)} a +{h} dias.")
        st.stop()
    try:
        with st.spinner("Procesando escena y generando pronostico..."):
            fig, stats = build_map_figure(wb, h, path, t0, res=res)
            fc = forecast_body(wb, t0, spec_override=spec_override, res=res)
    except ValueError as e:
        st.error(f"No se pudo analizar la escena: {e}"); st.stop()
    except Exception as e:
        log.exception("fallo en analisis")
        st.error(f"Error inesperado: {type(e).__name__}: {e}"); st.stop()

    if fc is None:
        st.error("No hay datos suficientes del cuerpo para construir el pronostico."); st.stop()
    hh = next((x for x in fc["horizons"] if x["horizon"] == h), None)

    # confianza (frescura / cobertura / estado)
    conf, flags, age = guards.evaluate_guards(wb, fc["t0"], stats["n_water_px"])

    st.divider()
    st.subheader(f"Resultado — {NICE.get(wb, wb)} · pronostico a +{h} dias")
    cap = f"Escena t0 = {fc['t0'].date() if fc['t0'] is not None else '?'}"
    cap += f"  ·  confianza: **{conf}**" + (f" ({', '.join(flags)})" if flags else "")
    st.caption(cap)

    # ELEMENTOS 1 y 2: imagen satelital real + mapa de biomasa prevista (2 paneles, estilo make_maps)
    st.pyplot(fig, use_container_width=True)

    # ELEMENTOS 3 y 4: alerta + banda de incertidumbre
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Nivel de biomasa algal**")
        nivel = hh["nivel"] if hh is not None else None
        if nivel == "floracion":
            st.error(f"🔴 FLORACION — chl-a prevista {hh['chl_pred']:.1f} µg/L (≥ {fc['thr_floracion']:.0f})")
        elif nivel == "elevada":
            st.warning(f"🟡 BIOMASA ELEVADA — chl-a prevista {hh['chl_pred']:.1f} µg/L "
                       f"(≥ {C.THRESHOLDS['moderate']:.0f})")
        elif nivel is not None:
            st.success(f"🟢 NORMAL — chl-a prevista {hh['chl_pred']:.1f} µg/L "
                       f"(< {C.THRESHOLDS['moderate']:.0f})")
        st.caption(f"Area en floracion (≥ {stats['thr']:.0f} µg/L): **{stats['pct_alert']:.0f}%**  ·  "
                   f"biomasa elevada (≥ {stats['thr_elev']:.0f}): **{stats['pct_elev']:.0f}%**  ·  "
                   f"prob. anomalia (P85): {hh['prob_riesgo']*100:.0f}%")
    with cB:
        st.markdown("**Clorofila-a prevista (intensidad)**")
        if hh is not None:
            banda = (f"P10–P90: {hh['p10']:.1f} – {hh['p90']:.1f} µg/L"
                     if hh["p10"] is not None else "banda no disponible")
            st.metric(f"clorofila-a media prevista (+{h} d)", f"{hh['chl_pred']:.1f} µg/L")
            st.caption(f"Banda de incertidumbre calibrada (CQR ~80%) · {banda}")

    # ELEMENTO 5: disclaimer fijo
    st.divider()
    st.info(DISCLAIMER)
    if wb in C.EXPLORATORY_BODIES:
        st.caption("🔬 Cuerpo EXPLORATORIO: interpretar con cautela (sin validacion de campo 2023-2026).")
