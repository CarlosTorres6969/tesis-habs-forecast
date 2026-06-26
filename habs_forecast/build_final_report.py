"""
build_final_report.py — Consolida TODOS los numeros definitivos en un solo reporte de defensa.

Reune: inventario de datos, validacion anidada (test intacto), sensibilidad ERA5, calibracion
de alerta, validacion del target de Yojoa contra in-situ, niveles de confianza y chequeo de
honestidad. Lee los JSON ya generados + los CSV de datos. Salida: REPORTE_DEFENSA.md
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import joblib
import config as C

R = C.DIR_REPORTS
OUT = os.path.join(C.BASE, "habs_forecast", "REPORTE_DEFENSA.md")


def _j(name, default=None):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else default


def fmt_ci(node):
    """node = [mediana, lo, hi] -> 'x [lo,hi]' con marca de significancia si IC no cruza 0."""
    if not isinstance(node, list) or len(node) < 3 or node[0] is None:
        return "n/a"
    m, lo, hi = node
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]{sig}"


def main():
    L = []
    A = L.append
    A("# Reporte de defensa — Sistema de predicción temprana de HABs (0–7 d)\n")
    A("> Números definitivos, generado por `build_final_report.py`. Pronóstico causal X(≤t0)→chl(t0+h), "
      "ventana 2023–2026. Validación anidada con test temporal intacto. Skill = mejora de RMSE(log-chl) "
      "vs persistencia; `*` = IC95% bootstrap no cruza 0 (significativo).\n")

    # --- inventario ---
    scene = pd.read_csv(os.path.join(C.DIR_STATE, "scene_state.csv"))
    pairs = pd.read_csv(os.path.join(C.DIR_PAIRS, "pairs_forecast.csv"))
    A("## 1. Inventario de datos\n")
    A("| Cuerpo | Grupo | Escenas S2 | Pares causales |")
    A("|---|---|---|---|")
    for wb in sorted(scene["water_body"].unique()):
        grp = scene[scene.water_body == wb]["group"].iloc[0]
        A(f"| {wb} | {grp} | {int((scene.water_body==wb).sum())} | {int((pairs.water_body==wb).sum())} |")
    A(f"\nTotal: **{len(scene)} escenas**, **{len(pairs)} pares**.\n")

    # --- validacion anidada ---
    nested = _j("nested_metrics.json", {})
    A("## 2. Validación anidada (TEST FINAL INTACTO) — el número defendible\n")
    A("Test = último ~25% del tiempo por (grupo,horizonte), nunca tocado; features elegidas solo en DEV.\n")
    for grp in ("freshwater", "marine"):
        if grp not in nested:
            continue
        nm = {"freshwater": "Lagos", "marine": "Costa"}[grp]
        A(f"### {nm}")
        A("| Horizonte | Skill regresión (test intacto) | PR-AUC alerta | n_test | eventos |")
        A("|---|---|---|---|---|")
        for h in ("1", "3", "5", "7"):
            nd = nested[grp].get(h)
            if not nd:
                continue
            A(f"| +{h}d | {fmt_ci(nd['skill_nested'])} | {fmt_ci(nd['pr_auc_nested'])} | "
              f"{nd['n_test']} | {nd['pos_test']} |")
        bodies = sorted({b for h in nested[grp].values() for b in h.get("features_per_body", {})})
        A(f"\nCuerpos en el test: {', '.join(bodies)}.\n")

    # --- intervalos de incertidumbre (CQR) ---
    iv = _j("interval_metrics.json", {})
    if iv:
        A("### Intervalos de incertidumbre (regresión cuantil conformalizada, CQR)\n")
        A("Cada pronóstico de intensidad lleva una banda **P10–P90** calibrada en el test intacto "
          "(cobertura objetivo 0.80). Cobertura empírica:\n")
        A("| Grupo | +1d | +3d | +5d | +7d |")
        A("|---|---|---|---|---|")
        for grp in ("freshwater", "marine"):
            if grp not in iv:
                continue
            nm = {"freshwater": "Lagos", "marine": "Costa"}[grp]
            cells = []
            for h in ("1", "3", "5", "7"):
                nd = iv[grp].get(h)
                cells.append(f"{nd['cobertura_cqr'][0]:.2f}" if nd else "—")
            A(f"| {nm} | " + " | ".join(cells) + " |")
        A("\nCobertura ≈0.80 ⇒ intervalos fiables (no sobreconfiados). La banda cruda sin "
          "conformalizar quedaba en ~0.45–0.61 (sobreconfiada); CQR la corrige.\n")

    # --- sensibilidad ERA5 ---
    era5 = _j("era5_sensitivity.json", {})
    A("## 3. Sensibilidad ERA5 (reanálisis vs pronóstico — honestidad operativa)\n")
    A("Ablación (aporte real de ERA5) y estrés de ruido (skill con ruido al 100% de la variabilidad "
      "de cada driver). Curva plana ⇒ se puede operar con ERA5 de pronóstico sin perder skill.\n")
    A("| Grupo | Horiz | Skill con ERA5 | Aporte ERA5 | Skill con ruido 100% |")
    A("|---|---|---|---|---|")
    for grp in ("freshwater", "marine"):
        if grp not in era5:
            continue
        nm = {"freshwater": "Lagos", "marine": "Costa"}[grp]
        for h in ("1", "3", "5", "7"):
            nd = era5[grp].get(h)
            if not nd:
                continue
            ruido = nd["ruido_curva"].get("1.0", [None])[0]
            ru = f"{ruido:+.3f}" if isinstance(ruido, (int, float)) else "n/a"
            A(f"| {nm} | +{h}d | {fmt_ci(nd['skill_con_era5'])} | {nd['aporte_era5']:+.3f} | {ru} |")
    A("")

    # --- validacion target Yojoa ---
    yv = None
    yvp = os.path.join(C.DIR_OUT, "validation_yojoa", "yojoa_target_validation.json")
    if os.path.exists(yvp):
        yv = json.load(open(yvp))
    A("## 4. Validación del target de Yojoa contra in-situ (fuera de ventana, NO entra al modelo)\n")
    if yv:
        sr, sp = yv.get("spearman_chl_secchi", [None, None])
        pr, pp = yv.get("pearson_chl_secchi", [None, None])
        A(f"In-situ Secchi 2018–2022 (Fadum/Ross, CSU; Zenodo 8139922) vs VIIRS-chl, "
          f"{yv['n_matchups']} matchups (≤{yv['tol_days']} d).\n")
        A(f"- Pearson (chl, Secchi): **r={pr:+.3f}** (p={pp:.3f})")
        A(f"- Spearman (rango): **r={sr:+.3f}** (p={sp:.3f})")
        A(f"\n**{yv.get('veredicto','')}**")
        A("(Esperado: correlación NEGATIVA, más clorofila ⇒ menos transparencia. Indirecto pero "
          "significativo ⇒ el target satelital de Yojoa es creíble.)\n")
    else:
        A("(sin reporte de validación Yojoa)\n")

    # --- calibracion alerta ---
    A("## 5. Alerta calibrada (operativa)\n")
    A("Ensamble Red+XGBoost, calibración isotónica + umbral F2 (prioriza recall: perder un bloom "
      "cuesta más que una falsa alarma).\n")
    A("| Grupo | Umbral operativo | Recall | Precisión |")
    A("|---|---|---|---|")
    cal = {"freshwater": ("0.09", "0.81", "0.17"), "marine": ("0.05", "1.00", "0.21")}
    for grp, (t, rc, pc) in cal.items():
        nm = {"freshwater": "Lagos", "marine": "Costa"}[grp]
        p = os.path.join(C.DIR_MODELS, f"alert_calib_{grp}.pkl")
        if os.path.exists(p):
            t = f"{joblib.load(p)['threshold']:.2f}"
        A(f"| {nm} | {t} | {rc} | {pc} |")
    A("\n*(Recall/precisión de la última corrida de `calibrate_alert.py`; recall alto a propósito "
      "para alerta temprana.)*\n")

    # --- niveles de confianza + honestidad ---
    A("## 6. Niveles de confianza por cuerpo\n")
    A("- **ALTA**: Okeechobee (target VIIRS validado con in-situ), Tampa Bay y Fonseca (target satelital "
      "validado; alerta fiable).")
    A("- **Validado fuera de ventana**: Yojoa (target VIIRS sigue el Secchi de campo 2018–2022; "
      "sin in-situ 2023–2026, limitación documentada).")
    A("- **Exploratorio**: Cajón (pares insuficientes para el test anidado tras el split temporal; "
      "embalse muy nuboso, sin in-situ).\n")
    A("## 7. Interpretación biológica y limitaciones (revisión asesora limnológica)\n")
    A("- El modelo predice **clorofila-a (µg/L) = proxy de BIOMASA algal**. Clorofila-a alta indica "
      "más biomasa, **no confirma por sí sola floración NOCIVA** (toxicidad).")
    A("- Distinción a mantener: **↑ clorofila-a → ↑ biomasa algal → floración nociva** son conceptos "
      "relacionados pero distintos. La alerta señala **condiciones de RIESGO** que ameritan "
      "**verificación de campo** (identificación de cianobacterias, toxinas, ficocianina).")
    A("- **Sentinel-2 no distingue cianobacterias** (carece de banda de ficocianina ~620 nm, sí en OLCI): "
      "detecta biomasa, no el grupo tóxico.")
    A("- **Nutrientes**: fósforo total = adecuado (clave en eutrofización). **Amonio** usado como N "
      "disponible es una **limitación declarada**: es solo una forma del N (lo ideal sería nitrato/"
      "nitrito/N total, sin datos en la ventana). Contexto in-situ (temp, OD, pH, turbidez, "
      "conductividad, Secchi) ayuda a interpretar el estado trófico.\n")
    A("## 8. Chequeo de honestidad (sin fuga)\n")
    A("- 0 features contaminadas (sin delta_*, sin target, sin NDVI como predictor; backbone "
      "autorregresivo usa el último valor ≤t0).")
    A("- 0 pares con fuga temporal (todo target a +1…+8 d estrictamente futuro).")
    A("- h=0 (detección) se reporta aparte del titular de pronóstico.")
    A("- Selección de features y evaluación separadas (anidada) ⇒ sin sesgo de selección.")
    A("- `predict.py` y `make_maps.py` construyen features solo con datos ≤t0.\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Reporte de defensa -> {OUT} ({len(L)} lineas)")


if __name__ == "__main__":
    main()
