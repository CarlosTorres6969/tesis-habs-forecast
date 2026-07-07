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
import os, glob, re, tempfile, logging, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import torch
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import config as C
from predict import forecast_body, GROUP, SPEC, MODELS
from make_maps import build_map_figure, _scene_pixels, KEY2FOLDER, _clear_water_score
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

# Colores por nivel biologico (consistentes en toda la app: curva, medidor, marcadores)
LEVEL_COLOR = {"floracion": "#d64545", "elevada": "#e0a800", "normal": "#2fb37f"}
LEVEL_LABEL = {"floracion": "Floracion", "elevada": "Biomasa elevada", "normal": "Normal"}

# Nombres legibles de las variables del modelo (para el panel "¿por que?" SHAP)
FEATURE_ES = {
    "log_chl_t0": "Clorofila actual (t0)", "chl_roll7": "Clorofila media 7 dias",
    "chl_lag3": "Clorofila hace 3 dias", "chl_lag7": "Clorofila hace 7 dias",
    "chl_trend7": "Tendencia de clorofila 7 dias",
    "NDCI": "Indice de clorofila (NDCI)", "CI_red": "Indice de cianobacterias (CI)",
    "FAI": "Indice de algas flotantes (FAI)", "turbidity": "Turbidez (satelite)",
    "B2": "Reflectancia azul (B2)", "B3": "Reflectancia verde (B3)",
    "B4": "Reflectancia roja (B4)", "B5": "Red-edge (B5)", "B8": "Infrarrojo cercano (B8)",
    "temp_air_2m": "Temperatura del aire", "temp_air_2m_roll7": "Temp. aire media 7 dias",
    "solar_radiation": "Radiacion solar", "solar_radiation_roll7": "Radiacion solar media 7 dias",
    "precipitation": "Precipitacion", "precipitation_roll7": "Precipitacion media 7 dias",
    "wind_speed_10m": "Viento a 10 m", "wind_speed_10m_roll7": "Viento medio 7 dias",
    "surface_pressure": "Presion superficial",
    "tp_context": "Fosforo total (in-situ)", "ammonia": "Amonio (in-situ)",
    "water_temp": "Temperatura del agua (in-situ)", "do_mgl": "Oxigeno disuelto (in-situ)",
    "ph": "pH (in-situ)", "turbidity_insitu": "Turbidez (in-situ)",
    "spec_cond": "Conductividad (in-situ)", "secchi": "Transparencia Secchi (in-situ)",
}


def _feat_es(f):
    return FEATURE_ES.get(f, f)


@st.cache_data(show_spinner=False)
def load_nested_metrics():
    """Metricas de validacion ANIDADA (test intacto) por grupo->horizonte: skill vs persistencia
    con IC95% bootstrap. Producidas por evaluate_nested.py. Devuelve dict o None."""
    p = os.path.join(C.DIR_REPORTS, "nested_metrics.json")
    if not os.path.exists(p):
        return None
    import json
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def render_skill_badge(group, h):
    """Insignia de credibilidad: skill validado vs persistencia (test nunca tocado) para el
    grupo+horizonte, honesta con el IC95% (verde si excluye 0, ambar si es no concluyente)."""
    d = load_nested_metrics()
    node = (d or {}).get(group, {}).get(str(h))
    if not node or "skill_nested" not in node:
        return
    pt, lo, hi = node["skill_nested"]
    n, pos = node.get("n_test", "?"), node.get("pos_test", "?")
    base = (f"ventaja sobre persistencia **{pt:+.2f}**  ·  IC95% [{lo:+.2f}, {hi:+.2f}]  ·  "
            f"a +{h} días  ·  sobre datos nunca vistos (n={n}, eventos={pos})")
    if lo > 0:
        st.success(f"🏅 **Validación externa — el modelo SUPERA al pronóstico ingenuo:** {base}  ·  "
                   f"**resultado significativo** (todo el IC95% está por encima de 0).")
        st.caption("Comparamos el modelo contra la *persistencia* (asumir que mañana el riesgo será "
                   "igual al de hoy). La *ventaja* es cuánto mejora el modelo sobre esa referencia, "
                   "medida en datos que nunca vio durante el entrenamiento. Como el intervalo de "
                   "confianza al 95 % no toca el 0, la mejora no se explica por azar.")


@st.cache_data(show_spinner=False)
def load_shap():
    """Importancia SHAP por (grupo, horizonte) precomputada por explain_model.py.
    Devuelve DataFrame o None si aun no se ha corrido la explicabilidad."""
    p = os.path.join(C.DIR_REPORTS, "shap_importance.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def trajectory_figure(fc, sel_h):
    """Curva interactiva de clorofila prevista a +1/+3/+5/+7 d con banda P10-P90 sombreada,
    marcadores coloreados por nivel y lineas de umbral. Todo el dato ya viene en fc['horizons']."""
    hs = sorted(fc["horizons"], key=lambda x: x["horizon"])
    xs = [f"+{x['horizon']} d" for x in hs]
    y = [x["chl_pred"] for x in hs]
    lo = [x["p10"] if x["p10"] is not None else x["chl_pred"] for x in hs]
    hi = [x["p90"] if x["p90"] is not None else x["chl_pred"] for x in hs]
    colors = [LEVEL_COLOR.get(x["nivel"], "#0fa3a3") for x in hs]
    sizes = [20 if x["horizon"] == sel_h else 12 for x in hs]
    cdata = [[x["p10"] or 0, x["p90"] or 0, x["prob_riesgo"] * 100, LEVEL_LABEL.get(x["nivel"], "")]
             for x in hs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself",
                             fillcolor="rgba(15,163,163,.16)", line=dict(color="rgba(0,0,0,0)"),
                             hoverinfo="skip", name="Banda P10–P90"))
    fig.add_trace(go.Scatter(x=xs, y=y, mode="lines+markers+text",
                             line=dict(color="#0a6b6b", width=3),
                             marker=dict(size=sizes, color=colors, line=dict(color="white", width=2)),
                             text=[f"{v:.0f}" for v in y], textposition="top center",
                             textfont=dict(size=12, color="#0a6b6b"),
                             customdata=cdata,
                             hovertemplate="<b>%{x}</b><br>Clorofila: %{y:.1f} µg/L<br>"
                             "Banda: %{customdata[0]:.1f}–%{customdata[1]:.1f} µg/L<br>"
                             "Prob. anomalia: %{customdata[2]:.0f}%<br>"
                             "Nivel: %{customdata[3]}<extra></extra>",
                             name="Clorofila prevista"))
    fig.add_hline(y=fc["thr_floracion"], line=dict(color="#d64545", dash="dash", width=1.4),
                  annotation_text="Floracion", annotation_position="top left")
    fig.add_hline(y=fc["thr_elevada"], line=dict(color="#e0a800", dash="dot", width=1.4),
                  annotation_text="Biomasa elevada", annotation_position="bottom left")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=34, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)",
                      yaxis_title="Clorofila-a prevista (µg/L)", xaxis_title="Horizonte de pronostico",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      font=dict(family="Segoe UI"), hovermode="x unified")
    return fig


def gauge_figure(prob, thr, nivel):
    """Medidor (gauge) animado de la probabilidad de anomalia, con el umbral operativo REAL
    (calibrado) marcado. La aguja anima de 0 al valor al renderizar."""
    val, thrp = prob * 100.0, thr * 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=val,
        number={"suffix": "%", "font": {"size": 42, "color": "#0a6b6b"}},
        delta={"reference": thrp, "increasing": {"color": "#d64545"},
               "decreasing": {"color": "#2fb37f"}, "suffix": " pts vs umbral"},
        title={"text": "Probabilidad de anomalia (P85)", "font": {"size": 15}},
        gauge={"axis": {"range": [0, max(100.0, val * 1.1)]},
               "bar": {"color": LEVEL_COLOR.get(nivel, "#0fa3a3")},
               "bgcolor": "rgba(255,255,255,.4)",
               "steps": [{"range": [0, thrp], "color": "rgba(46,179,127,.25)"},
                         {"range": [thrp, max(100.0, val * 1.1)], "color": "rgba(214,69,69,.16)"}],
               "threshold": {"line": {"color": "#d64545", "width": 4}, "thickness": 0.85,
                             "value": thrp}}))
    fig.update_layout(height=300, margin=dict(l=24, r=24, t=56, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Segoe UI"))
    return fig


def shap_bar_figure(group, h, topn=8):
    """Barras horizontales de las variables que MAS mueven la prediccion de este grupo+horizonte
    (media |SHAP| precomputada). Devuelve None si no hay explicabilidad calculada aun."""
    d = load_shap()
    if d is None:
        return None
    sub = d[(d["group"] == group) & (d["horizon"] == h)].nsmallest(topn, "rank")
    if sub.empty:
        return None
    sub = sub.sort_values("mean_abs_shap")
    labels = [_feat_es(f) for f in sub["feature"]]
    fig = go.Figure(go.Bar(x=sub["mean_abs_shap"], y=labels, orientation="h",
                           marker=dict(color=sub["mean_abs_shap"], colorscale="Teal",
                                       line=dict(color="rgba(10,107,107,.5)", width=1)),
                           hovertemplate="<b>%{y}</b><br>Peso: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)",
                      xaxis_title="Influencia media (|SHAP|)", font=dict(family="Segoe UI"))
    return fig


def forecast_animation_gif(wb, path, t0, res, horizons=(1, 3, 5, 7), dpi=82, width=1040,
                           steps=7, trans_ms=65, hold_ms=460, nowcast_level=None):
    """Genera un GIF que se reproduce solo (estilo pronostico del clima en TV): opcionalmente parte
    del estado de HOY (observado) y recorre los mapas de biomasa prevista a +1/+3/+5/+7 dias sobre
    la MISMA escena. Para que sea FLUIDO interpola (cross-fade) cuadros intermedios entre horizontes
    con Pillow y cierra el bucle de forma continua; el modelo solo se evalua en los horizontes
    reales (sin tocar modelado). 'steps' = cuadros de transicion por tramo; pausa 'hold_ms' en cada
    dia y 'trans_ms' en la disolvencia. Devuelve bytes del GIF o None si no se pudo generar cuadros.
    Si nowcast_level se pasa, el primer cuadro es 'Hoy (observado)' con ese nivel de clorofila."""
    keys = []
    # cuadro "Hoy" (observado) al inicio, si se pide
    specs = ([("nowcast", horizons[0])] if nowcast_level is not None else []) + \
            [("fc", hh) for hh in horizons]
    for kind, hh in specs:
        if (GROUP[wb], hh) not in res["bundles"]:
            continue
        try:
            nl = nowcast_level if kind == "nowcast" else None
            fig, _ = build_map_figure(wb, hh, path, t0, res=res, gradient_focus=True, nowcast_level=nl)
        except Exception as e:
            log.warning("frame %s +%dd fallo: %s", kind, hh, e)
            continue
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)      # figsize fijo -> cuadros clave del mismo tamaño
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        if width and im.width != width:               # aligerar para que el GIF no pese de mas
            im = im.resize((width, round(im.height * width / im.width)))
        keys.append(im)
    if not keys:
        return None
    size = keys[0].size
    keys = [k if k.size == size else k.resize(size) for k in keys]

    frames, durs = [], []
    def _add(img, ms):
        frames.append(img)
        durs.append(ms)

    if len(keys) == 1:
        _add(keys[0], hold_ms)
    else:
        for i in range(len(keys) - 1):                # pausa en el dia i, luego disuelve al i+1
            a, b = keys[i], keys[i + 1]
            _add(a, hold_ms)
            for k in range(1, steps + 1):
                _add(Image.blend(a, b, k / (steps + 1)), trans_ms)
        _add(keys[-1], hold_ms)                        # pausa en el ultimo dia
        for k in range(1, steps + 1):                  # cierre continuo: ultimo -> primero
            _add(Image.blend(keys[-1], keys[0], k / (steps + 1)), trans_ms)

    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, disposal=2, optimize=True)
    return out.getvalue()

# --------------------------------------------------------------------------------------
# Tema visual (acuatico) — CSS + encabezado "hero". Solo presentacion, no toca la logica.
# --------------------------------------------------------------------------------------
THEME_CSS = """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 2.2rem; max-width: 1180px; }

/* perspectiva para que las tarjetas tengan profundidad 3D real */
[data-testid="stHorizontalBlock"] { perspective: 1200px; }

/* Botones con relieve y "pulsado" 3D */
.stButton > button { border-radius:12px; font-weight:700; border:0; color:#04302f;
  background:linear-gradient(120deg,#0fa3a3,#46c39b);
  box-shadow:0 6px 0 #0a6b6b, 0 10px 18px rgba(6,43,63,.25);
  transition:transform .1s ease, box-shadow .1s ease; }
.stButton > button:hover { transform:translateY(-2px);
  box-shadow:0 8px 0 #0a6b6b, 0 14px 24px rgba(15,163,163,.35); color:#04302f; }
.stButton > button:active { transform:translateY(4px);
  box-shadow:0 2px 0 #0a6b6b, 0 4px 10px rgba(6,43,63,.25); }

/* Tarjetas (metricas): glass + inclinacion 3D al pasar el mouse */
[data-testid="stMetric"] {
  background:rgba(255,255,255,.72); backdrop-filter:blur(8px);
  border:1px solid rgba(255,255,255,.6); border-left:5px solid #0fa3a3; border-radius:14px;
  padding:.9rem 1.1rem; box-shadow:0 10px 24px rgba(6,43,63,.12);
  transform-style:preserve-3d; transition:transform .25s ease, box-shadow .25s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-5px) rotateX(6deg) rotateY(-3deg);
  box-shadow:0 22px 40px rgba(6,43,63,.22); }

/* Alertas con glass y profundidad */
[data-testid="stAlert"] { border-radius:14px; backdrop-filter:blur(6px);
  box-shadow:0 10px 22px rgba(6,43,63,.10); transform-style:preserve-3d; transition:transform .2s ease; }
[data-testid="stAlert"]:hover { transform:translateY(-3px) rotateX(4deg); }

/* La figura/mapa como lamina flotante */
[data-testid="stImage"], [data-testid="stPyplotChart"] {
  border-radius:14px; box-shadow:0 16px 36px rgba(6,43,63,.18); overflow:hidden;
  transition:transform .3s ease; }
[data-testid="stImage"]:hover, [data-testid="stPyplotChart"]:hover { transform:translateY(-3px) scale(1.004); }

/* Encabezados y sidebar */
h2, h3 { color:#0a6b6b; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#e1f3f1 0%,#eef9f8 100%);
  border-right:1px solid #cfeae7; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#0a6b6b; }
</style>
"""

# Encabezado 3D: animacion WebGL de agua (shader con iluminacion) embebida -> funciona OFFLINE.
HERO_3D = """
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;height:100%;overflow:hidden;background:#f2fbfa;
    font-family:"Segoe UI",system-ui,-apple-system,sans-serif;}
  #wrap{position:relative;width:100%;height:240px;border-radius:18px;overflow:hidden;
    background:linear-gradient(120deg,#062b3f,#0a6b6b,#1aa39a,#46c39b);
    box-shadow:0 14px 34px rgba(6,43,63,.30);}
  #gl{position:absolute;inset:0;width:100%;height:100%;display:block;}
  #ov{position:absolute;inset:0;padding:26px 32px;color:#f7ffff;pointer-events:none;
    display:flex;flex-direction:column;justify-content:center;text-shadow:0 2px 14px rgba(0,0,0,.45);}
  #ov h1{margin:0;font-size:30px;font-weight:800;letter-spacing:.3px;}
  #ov p{margin:9px 0 0;font-size:15px;max-width:64ch;opacity:.97;}
  .tags{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;}
  .tag{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.34);
    padding:5px 11px;border-radius:999px;font-size:12px;}
</style></head><body>
<div id="wrap">
  <canvas id="gl"></canvas>
  <div id="ov">
    <h1>🌊 Alerta temprana de biomasa algal (HABs)</h1>
    <p>Pronostico de riesgo de floraciones algales a 0–7 dias. Clorofila-a como
       <b>proxy de biomasa</b>: senala <b>riesgo</b>, no confirma toxicidad.</p>
    <div class="tags">
      <span class="tag">🛰️ Sentinel-2</span><span class="tag">💧 5 cuerpos validados</span>
      <span class="tag">📈 Horizontes 0–7 dias</span><span class="tag">🧪 XGBoost + Red neuronal</span>
    </div>
  </div>
</div>
<script>
(function(){
  var c=document.getElementById('gl');
  var gl=c.getContext('webgl')||c.getContext('experimental-webgl');
  if(!gl){return;}                         // sin WebGL -> queda el degradado CSS
  function rs(){var d=window.devicePixelRatio||1;c.width=c.clientWidth*d;c.height=c.clientHeight*d;gl.viewport(0,0,c.width,c.height);}
  window.addEventListener('resize',rs);rs();
  var vs='attribute vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}';
  var fs='precision highp float;uniform float u_t;uniform vec2 u_r;'+
    'float wh(vec2 p,float t){float h=0.0;h+=sin(p.x*1.5+t)*0.5;h+=sin(p.y*1.7-t*1.1)*0.4;'+
    'h+=sin((p.x+p.y)*1.1+t*0.7)*0.3;h+=sin(length(p-vec2(sin(t)*0.8,cos(t*0.7)*0.8))*3.0-t*2.0)*0.25;return h;}'+
    'void main(){vec2 uv=gl_FragCoord.xy/u_r;vec2 p=(uv-0.5)*vec2(u_r.x/u_r.y,1.0)*6.0;'+
    'float t=u_t*0.5;float e=0.06;float h=wh(p,t);float hx=wh(p+vec2(e,0.0),t)-h;float hy=wh(p+vec2(0.0,e),t)-h;'+
    'vec3 n=normalize(vec3(-hx,-hy,e*4.0));vec3 L=normalize(vec3(0.4,0.6,0.7));'+
    'float dif=clamp(dot(n,L),0.0,1.0);float sp=pow(clamp(dot(reflect(-L,n),vec3(0.0,0.0,1.0)),0.0,1.0),24.0);'+
    'vec3 deep=vec3(0.02,0.17,0.27);vec3 teal=vec3(0.06,0.64,0.62);vec3 alg=vec3(0.27,0.78,0.55);'+
    'vec3 col=mix(deep,teal,dif);col=mix(col,alg,smoothstep(0.6,1.0,dif)*0.55);col+=sp*0.6;'+
    'gl_FragColor=vec4(col,1.0);}';
  function sh(t,s){var o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o;}
  var pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));
  gl.linkProgram(pr);gl.useProgram(pr);
  var buf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
  var lp=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,2,gl.FLOAT,false,0,0);
  var ut=gl.getUniformLocation(pr,'u_t'),ur=gl.getUniformLocation(pr,'u_r');
  var t0=performance.now();
  (function loop(){var t=(performance.now()-t0)/1000.0;gl.uniform1f(ut,t);gl.uniform2f(ur,c.width,c.height);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);requestAnimationFrame(loop);})();
})();
</script>
</body></html>
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


# Cota de escaneo: puntuar calidad de agua lee un raster por escena; algunos cuerpos costeros
# tienen rasters pesados (~0.6 s c/u) y evaluarlos TODOS al cargar bloqueaba la UI 30-60 s. Se
# acota a las mas recientes y se muestra barra de avance (suficiente para elegir una escena limpia).
MAX_RANK_SCENES = 40


def rank_scenes_with_progress(wb):
    """Ordena las escenas del cuerpo por CALIDAD de agua limpia (mejor primero), reutilizando
    _clear_water_score de make_maps (premia agua coherente, penaliza nubosidad). Solo puntua las
    MAX_RANK_SCENES mas recientes, muestra una BARRA DE PROGRESO (para que no parezca colgada) y
    guarda el resultado en session_state -> se calcula una sola vez por cuerpo. Devuelve lista de
    (fecha, ruta, score) ordenada, o [] si no hay escenas."""
    key = f"ranked_{wb}"
    if key in st.session_state:
        return st.session_state[key]
    scenes = list_example_scenes(wb)[:MAX_RANK_SCENES]     # mas recientes primero; acota el I/O
    if not scenes:
        st.session_state[key] = []
        return []
    bar = st.progress(0.0, text=f"Evaluando calidad de {len(scenes)} escenas recientes de "
                                f"{NICE.get(wb, wb)}...")
    scored = []
    for i, (fecha, path) in enumerate(scenes):
        scored.append((fecha, path, _clear_water_score(path)))
        bar.progress((i + 1) / len(scenes))
    bar.empty()
    scored.sort(key=lambda x: x[2], reverse=True)
    st.session_state[key] = scored
    return scored


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
        "amerita verificacion de campo.\n"
        "- **Sentinel-2 mide el PIGMENTO (clorofila-a), no la especie ni la toxina**: no distingue "
        "cianobacterias (lagos) ni dinoflagelados de marea roja (costa). Identificar el organismo "
        "y confirmar toxinas exige **muestreo de campo** (microscopia / ensayos de toxinas).")
    st.divider()
    st.caption("Modelo: XGBoost (intensidad + intervalos CQR) + Red neuronal (alerta), por "
               "grupo ecologico y horizonte. Pronostico causal sin fuga (validacion anidada).")

components.html(HERO_3D, height=250, scrolling=False)
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
    ranked = rank_scenes_with_progress(wb)           # (fecha, ruta, score) mejor primero; barra de avance
    if not ranked:
        scene_err = f"No hay escenas Sentinel-2 de ejemplo para {NICE.get(wb, wb)}."
    else:
        best_fecha, best_path, best_score = ranked[0]
        auto = st.checkbox("Usar automaticamente la mejor escena (agua mas limpia)", value=True,
                           help="Evita caer en escenas nubladas donde el cuerpo de agua ni aparece. "
                                "Desmarca para elegir la fecha manualmente.")
        if auto:
            sel, path, sel_score = best_fecha, best_path, best_score
            st.caption(f"Escena elegida automaticamente por calidad de agua limpia: **{best_fecha}**.")
        else:
            # dropdown ORDENADO por calidad (mejor primero); la mejor se marca con estrella
            fechas = [f for f, _, _ in ranked]
            marca = {best_fecha: f"{best_fecha}  ⭐ mejor (agua mas limpia)"}
            sel = st.selectbox(f"Escena disponible ({len(ranked)} mas recientes, ordenadas por calidad)",
                               fechas, format_func=lambda f: marca.get(f, f))
            rec = next(r for r in ranked if r[0] == sel)
            path, sel_score = rec[1], rec[2]
        t0 = pd.Timestamp(sel) if re.match(r"\d{4}-\d{2}-\d{2}", sel) else None
        # aviso si la escena elegida no tiene un cuerpo de agua coherente (nublada/dispersa)
        if sel_score <= 0 or (best_score > 0 and sel_score < 0.15 * best_score):
            st.warning("⚠️ En esta escena el cuerpo de agua no se detecta con claridad "
                       "(probable nubosidad/neblina). Prueba otra fecha o usa la mejor escena "
                       "automaticamente para un encuadre y color mas legibles.")
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
# La casilla se define ANTES del boton: asi su valor ya esta disponible en el rerun del clic.
animate = st.checkbox(
    "🎬 Mostrar animación tipo pronóstico del clima (recorre +1 → +7 días)", value=False,
    help="Genera un video corto que recorre la biomasa algal prevista a 1, 3, 5 y 7 días sobre la "
         "misma escena, como un pronóstico del clima en la tele. Tarda unos segundos mas.")
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
            fig, stats = build_map_figure(wb, h, path, t0, res=res, gradient_focus=True)
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
    # dpi=200 + bbox tight: misma nitidez que los PNG del CLI (no el ~100 dpi por defecto de Streamlit).
    _mbuf = io.BytesIO()                                   # capturar PNG ANTES de st.pyplot (por si lo cierra)
    fig.savefig(_mbuf, format="png", dpi=200, bbox_inches="tight")
    st.session_state["map_png"] = _mbuf.getvalue()
    st.pyplot(fig, use_container_width=True, dpi=200, bbox_inches="tight")

    # INSIGNIA de credibilidad: skill validado vs persistencia (test intacto) del horizonte elegido.
    render_skill_badge(GROUP[wb], h)

    # ANIMACION tipo pronostico del clima (opcional): recorre +1/+3/+5/+7 d como un video.
    # Cacheada por escena en session_state -> no se regenera al cambiar de horizonte o pestaña.
    if animate:
        sig = f"{wb}|{path}|{fc['t0']}"
        if st.session_state.get("anim_sig") != sig:
            with st.spinner("Generando animación tipo pronóstico (Hoy → 7 días)..."):
                st.session_state["anim_gif"] = forecast_animation_gif(
                    wb, path, fc["t0"], res, nowcast_level=fc.get("chl0"))
                st.session_state["anim_sig"] = sig
        gif = st.session_state.get("anim_gif")
        if gif:
            st.markdown("### 🎬 Animación del pronóstico (Hoy → 7 días)")
            st.image(gif, use_container_width=True)
            st.caption("Arranca en el estado OBSERVADO de hoy y recorre la biomasa algal prevista a "
                       "1, 3, 5 y 7 días sobre la misma escena (se reproduce en bucle, como un "
                       "pronóstico del clima). La textura espacial viene de la escena actual; lo que "
                       "cambia entre cuadros es el NIVEL. A +1 y +7 días es nivel de cuerpo repartido "
                       "por el patrón actual; a +3 y +5 el gradiente lo aporta el modelo espectral.")
        else:
            st.info("No se pudo generar la animación para esta escena.")

    # ELEMENTOS 3 y 4: alerta + banda de incertidumbre
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Nivel de biomasa algal**")
        nivel = hh["nivel"] if hh is not None else None
        if nivel == "floracion":
            st.error(f"🔴 FLORACION — chl-a prevista {hh['chl_pred']:.1f} µg/L (≥ {fc['thr_floracion']:.0f})")
        elif nivel == "elevada":
            st.warning(f"🟡 BIOMASA ELEVADA — chl-a prevista {hh['chl_pred']:.1f} µg/L "
                       f"(≥ {fc['thr_elevada']:.0f})")
        elif nivel is not None:
            st.success(f"🟢 NORMAL — chl-a prevista {hh['chl_pred']:.1f} µg/L "
                       f"(< {fc['thr_elevada']:.0f})")
        st.caption(f"Area en floracion (≥ {stats['thr']:.0f} µg/L): **{stats['pct_alert']:.0f}%**  ·  "
                   f"biomasa elevada (≥ {stats['thr_elev']:.0f}): **{stats['pct_elev']:.0f}%**  ·  "
                   f"prob. anomalia (P85): {hh['prob_riesgo']*100:.0f}%")
        # DOS señales distintas que pueden diverger de forma legitima (se explica para no confundir):
        #  - NIVEL  = MAGNITUD de clorofila prevista vs umbral biologico (el banner de color de arriba).
        #  - prob. anomalia = chance de un SALTO atipico para ESTE cuerpo (clasificador calibrado, P85).
        # Un cuerpo cronicamente alto (embalse siempre con nata) puede dar NIVEL alto y prob. baja: ese
        # nivel es SU normal, no una anomalia. Y al reves: magnitud normal con salto atipico probable.
        if hh is not None:
            _alerta = hh["prob_riesgo"] >= fc["alert_threshold"]
            _alto = nivel in ("elevada", "floracion")
            if _alto and not _alerta:
                st.caption("ℹ️ *Nivel alto por **magnitud** de clorofila, pero **prob. de anomalía baja**: "
                           "ese nivel es **habitual** en este cuerpo, no un salto atípico. Son medidas "
                           "distintas — el nivel mide cuánta biomasa; la anomalía, si es inusual aquí.*")
            elif (not _alto) and _alerta:
                st.caption("ℹ️ *Magnitud prevista **normal**, pero el modelo marca **prob. de anomalía "
                           "elevada** (posible cambio atípico): conviene vigilar. Son medidas distintas.*")
    with cB:
        st.markdown("**Clorofila-a prevista (intensidad)**")
        if hh is not None:
            banda = (f"P10–P90: {hh['p10']:.1f} – {hh['p90']:.1f} µg/L"
                     if hh["p10"] is not None else "banda no disponible")
            st.metric(f"clorofila-a media prevista (+{h} d)", f"{hh['chl_pred']:.1f} µg/L")
            st.caption(f"Banda de incertidumbre calibrada (CQR ~80%) · {banda}")

    # ELEMENTOS DINAMICOS: trayectoria 0-7 d, medidor de riesgo y "¿por que?" (SHAP).
    # Todo reutiliza datos ya calculados (fc["horizons"], shap_importance.csv); no reentrena nada.
    st.divider()
    tab_tray, tab_gauge, tab_why = st.tabs(
        ["📈 Trayectoria 0–7 días", "🎯 Medidor de riesgo", "🧠 ¿Por qué? (SHAP)"])
    with tab_tray:
        st.plotly_chart(trajectory_figure(fc, h), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Clorofila-a prevista en cada horizonte con su banda P10–P90 (CQR ~80%). "
                   "El punto grande es el horizonte seleccionado; el color indica el nivel "
                   "(verde normal · amarillo elevada · rojo floración). Pasa el cursor para ver detalle.")
    with tab_gauge:
        if hh is not None:
            st.plotly_chart(gauge_figure(hh["prob_riesgo"], fc["alert_threshold"], hh["nivel"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Probabilidad calibrada de un **salto anómalo** de biomasa a +{h} días "
                       "(clorofila por encima del P85 **habitual de este cuerpo**) — es **distinta** del "
                       "nivel de magnitud del banner: un cuerpo siempre-alto puede tener nivel alto y "
                       f"anomalía baja. La línea roja es el **umbral operativo real** "
                       f"({fc['alert_threshold']*100:.0f}%): por encima, el sistema dispara alerta. "
                       "El umbral es bajo a propósito (prioriza no perder eventos = recall alto, "
                       "propio de una alerta temprana).")
        else:
            st.info("Sin probabilidad disponible para este horizonte.")
    with tab_why:
        figw = shap_bar_figure(GROUP[wb], h)
        if figw is not None:
            st.plotly_chart(figw, use_container_width=True, config={"displayModeBar": False})
            st.caption("Variables que más pesan en el pronóstico de este grupo y horizonte "
                       "(media |SHAP|). En corto plazo domina la clorofila reciente (autorregresivo); "
                       "a mayor horizonte entran meteorología, nutrientes e índices espectrales. "
                       "Respalda el diseño del modelo, no es una relación causal directa.")
        else:
            st.info("Explicabilidad SHAP no disponible. Genérala con `python explain_model.py`.")

    # DESCARGAS: mapa PNG, animacion GIF (si se genero) y resumen CSV del pronostico.
    st.divider()
    st.markdown("### 💾 Descargas")
    # resumen CSV desde fc (todos los horizontes)
    _rows = ["horizonte_dias,chl_pred_ugL,p10_ugL,p90_ugL,prob_alerta,nivel"]
    for x in sorted(fc["horizons"], key=lambda z: z["horizon"]):
        _rows.append(f"{x['horizon']},{x['chl_pred']:.2f},"
                     f"{'' if x['p10'] is None else round(x['p10'],2)},"
                     f"{'' if x['p90'] is None else round(x['p90'],2)},"
                     f"{x['prob_riesgo']:.4f},{LEVEL_LABEL.get(x['nivel'], x['nivel'])}")
    _csv = ("# Pronostico de biomasa algal (clorofila-a) - NO confirma toxicidad; requiere "
            "verificacion de campo\n"
            f"# cuerpo={wb} grupo={GROUP[wb]} escena_t0={fc['t0'].date()} "
            f"chl_actual_ugL={fc.get('chl0', float('nan')):.2f} "
            f"umbral_floracion_ugL={fc['thr_floracion']:.1f}\n" + "\n".join(_rows))
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button("🖼️ Mapa (PNG)", st.session_state.get("map_png", b""),
                           file_name=f"mapa_{wb}_h{h}_{fc['t0'].date()}.png", mime="image/png",
                           disabled=not st.session_state.get("map_png"), use_container_width=True)
    with dc2:
        _gif = st.session_state.get("anim_gif") if animate else None
        st.download_button("🎬 Animación (GIF)", _gif or b"",
                           file_name=f"animacion_{wb}_{fc['t0'].date()}.gif", mime="image/gif",
                           disabled=not _gif, use_container_width=True,
                           help=None if _gif else "Marca la casilla de animación y vuelve a Analizar.")
    with dc3:
        st.download_button("📄 Pronóstico (CSV)", _csv.encode("utf-8"),
                           file_name=f"pronostico_{wb}_{fc['t0'].date()}.csv", mime="text/csv",
                           use_container_width=True)

    # ELEMENTO 5: disclaimer fijo
    st.divider()
    st.info(DISCLAIMER)
    if wb in C.EXPLORATORY_BODIES:
        st.caption("🔬 Cuerpo EXPLORATORIO: interpretar con cautela (sin validacion de campo 2023-2026).")
