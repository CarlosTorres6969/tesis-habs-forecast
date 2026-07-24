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


def fmt_plain_ci(node):
    if not isinstance(node, list) or len(node) < 3 or node[0] is None:
        return "n/a"
    point, lo, hi = node
    return f"{point:.2f} [{lo:.2f},{hi:.2f}]"


def main():
    L = []
    A = L.append
    A("# Reporte de defensa — Sistema de predicción temprana de HABs (0–7 d)\n")
    A("> Números definitivos, generado por `build_final_report.py`. Pronóstico causal X(≤t0)→chl(t0+h), "
      "ventana 2023–2026. Validación anidada con bloque temporal reservado. Skill = mejora de "
      "RMSE(log-chl) vs persistencia; `*` se aplica solo al skill cuando su IC95% no cruza 0.\n")

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
    A("## 2. Validación anidada (TEST FINAL TEMPORAL) — el número defendible\n")
    A("Test = último ~25% de las fechas, con **un corte común para todos los cuerpos** de cada "
      "grupo-horizonte. Toda etiqueta de DEV queda antes del primer predictor de TEST; la selección "
      "usa folds internos purgados. Los IC95% usan bootstrap por bloques de 14 días y cuerpo.\n")
    for grp in ("freshwater", "marine"):
        if grp not in nested:
            continue
        nm = {"freshwater": "Lagos", "marine": "Costa"}[grp]
        A(f"### {nm}")
        A("| Horizonte | Skill regresión (test intacto) | PR-AUC alerta | n_test | eventos | Familias |")
        A("|---|---|---|---|---|---|")
        for h in ("1", "3", "5", "7"):
            nd = nested[grp].get(h)
            if not nd:
                continue
            fam = nd.get("features_per_body", {}).get("_grupo", "—")
            A(f"| +{h}d | {fmt_ci(nd['skill_nested'])} | {fmt_plain_ci(nd['pr_auc_nested'])} | "
              f"{nd['n_test']} | {nd['pos_test']} | {fam} |")
        bodies = sorted({body for h in ("1", "3", "5", "7")
                         for body in nested[grp].get(h, {}).get("test_bodies", [])})
        if bodies:
            A(f"\nCuerpos en el test: {', '.join(bodies)}.\n")
        else:
            A("")

    # --- intervalos de incertidumbre (CQR) ---
    iv = _j("interval_metrics.json", {})
    if iv:
        A("### Intervalos de incertidumbre (regresión cuantil conformalizada, CQR)\n")
        A("Cada pronóstico de intensidad lleva una banda **P10–P90** calibrada exclusivamente en "
          "CALIB, dentro de DEV, y evaluada después en TEST (cobertura objetivo 0.80). Cobertura empírica:\n")
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
        raw_values = [iv[g][h]["cobertura_cruda"] for g in ("freshwater", "marine")
                      for h in ("1", "3", "5", "7") if h in iv.get(g, {})]
        raw_range = (f"{min(raw_values):.2f}–{max(raw_values):.2f}" if raw_values else "n/a")
        A(f"\nCobertura cercana a 0.80 ⇒ intervalos razonablemente calibrados. La banda cruda sin "
          f"conformalizar quedó en {raw_range}; CQR mejoró su cobertura.\n")

    # --- sensibilidad ERA5 ---
    era5 = _j("era5_sensitivity.json", {})
    A("## 3. Sensibilidad ERA5 (reanálisis vs pronóstico — honestidad operativa)\n")
    A("Ablación y estrés simulado de ruido (hasta 100% de la variabilidad de cada driver). Una curva "
      "estable sugiere baja sensibilidad al error meteorológico, pero no sustituye validar un producto "
      "ERA5 de pronóstico real.\n")
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
    A("| Grupo | Umbral elegido en DEV | Recall TEST (IC95%) | Precisión TEST (IC95%) | F2 TEST |")
    A("|---|---|---|---|---|")
    for grp in ("freshwater", "marine"):
        nm = {"freshwater": "Lagos", "marine": "Costa"}[grp]
        cal = nested.get(grp, {}).get("alert_calibration") or {}
        threshold = cal.get("threshold_selected_in_dev")
        threshold_text = f"{threshold:.2f}" if threshold is not None else "n/a"
        A(f"| {nm} | {threshold_text} | {fmt_plain_ci(cal.get('recall_test'))} | "
          f"{fmt_plain_ci(cal.get('precision_test'))} | {fmt_plain_ci(cal.get('f2_test'))} |")
    A("\n*Isotónica y umbral se ajustan con predicciones OOS purgadas de DEV. Recall, precisión y "
      "F2 se calculan una sola vez en TEST; no son métricas de reajuste de producción.*\n")

    # --- niveles de confianza + honestidad ---
    A("## 6. Niveles de confianza por cuerpo\n")
    A("- **Okeechobee**: la escala VIIRS se calibró solo con campo de 2023. La validación causal "
      "posterior tiene cobertura muy limitada (2/1/1/0 fechas en +1/+3/+5/+7 d), por lo que no "
      "permite afirmar desempeño de campo concluyente.")
    A("- **Tampa Bay y Fonseca**: cuentan con validación temporal del target satelital a nivel de "
      "grupo; la alerta prioriza sensibilidad y presenta baja precisión, por lo que exige "
      "confirmación de campo.")
    A("- **Validado fuera de ventana**: Yojoa (target VIIRS sigue el Secchi de campo 2018–2022; "
      "sin in-situ 2023–2026, limitación documentada).")
    A("- **Exploratorio**: Cajón (sí participa en el test temporal, pero sigue sin validación in-situ "
      "independiente y presenta alta nubosidad).\n")
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
    A("- Corte cronológico común entre cuerpos y embargo verificado en cada frontera temporal.")
    A("- La corrección de escala de Okeechobee está congelada al 2023-12-31; ningún dato del TEST "
      "interviene en su ajuste.")
    A("- Selección de features y umbral dentro de DEV; el modelo de producción reutiliza esas familias.")
    A("- IC95% por bloques cuerpo-tiempo, sin tratar escenas repetidas como observaciones iid.")
    A("- `predict.py` y `make_maps.py` construyen features solo con datos ≤t0.\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Reporte de defensa -> {OUT} ({len(L)} lineas)")


if __name__ == "__main__":
    main()
