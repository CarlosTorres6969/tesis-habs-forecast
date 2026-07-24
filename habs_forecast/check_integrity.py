"""
check_integrity.py — TEST DE INTEGRIDAD del pipeline (sin fuga, causal, consistente).

Convierte en aserciones reproducibles las verificaciones de honestidad del sistema. Corre sobre
los pares y los modelos de produccion. Exit 0 = todo OK; exit 1 = alguna falla (apto para CI).
Respalda en la defensa que el pronostico es causal y libre de fuga (a diferencia del sistema
viejo con AUC=1.0 por circularidad/shuffle).

Uso:  python check_integrity.py
"""
from __future__ import annotations
import os, sys, glob, json, joblib
import numpy as np
import pandas as pd
import config as C
from train import FEATURES, AUTOREG, PAIRS

CHECKS = []          # (descripcion, ok:bool, detalle:str)


def chk(desc, ok, detalle=""):
    CHECKS.append((desc, bool(ok), detalle))


def main():
    target_cols = {"log_chl_target", "chl_target", "hab_target", "fecha_target", "gap_real", "thr_body"}

    # --- Checks ESTATICOS (no requieren datos; corren tambien en CI) ---
    # 1) Ninguna feature es el target ni un patron prohibido (fuga)
    forbidden = ["delta", "target", "future", "lag14", "lag30", "t-14", "t-30"]
    bad = [f for f in FEATURES if f in target_cols or any(s in f.lower() for s in forbidden)]
    chk("Ninguna feature contaminada/prohibida en FEATURES", not bad, f"sospechosas={bad}")

    # 2) NDVI no es predictor (solo mascara/QA)
    chk("NDVI NO es predictor (solo QA)", "NDVI" not in FEATURES)

    # 3) Backbone autorregresivo presente y causal por nombre
    chk("Backbone autorregresivo (log_chl_t0) presente", "log_chl_t0" in FEATURES)

    # 7) target (log_chl_target) NO esta dentro de FEATURES
    chk("El target no aparece como feature", "log_chl_target" not in FEATURES)

    # --- Checks ESTATICOS de la CAPA OPERATIVA (guards; sin datos ni torch -> corren en CI) ---
    import guards
    # 12) severidad de confianza bien formada: ordenada peor->mejor, termina en OK e incluye guardas
    sev = C.CONFIDENCE_SEVERITY
    ok_sev = sev[-1] == "OK" and {"LOW_COVERAGE", "STALE", "EXPLORATORIO"}.issubset(set(sev))
    chk("Severidad de confianza bien formada (termina en OK)", ok_sev, f"sev={sev}")
    # 13) la guarda reporta la PEOR condicion (vacio->OK; LOW_COVERAGE manda sobre STALE)
    ok_worst = (guards.worst_confidence([]) == "OK" and
                guards.worst_confidence(["STALE", "LOW_COVERAGE"]) == "LOW_COVERAGE")
    chk("Guarda de confianza: vacio=OK y respeta la peor condicion", ok_worst)
    # 14) los cuerpos exploratorios son cuerpos validos definidos en config
    bodies = {m["key"] for m in C.REGIONS.values()}
    chk("EXPLORATORY_BODIES son cuerpos validos", set(C.EXPLORATORY_BODIES).issubset(bodies),
        f"desconocidos={set(C.EXPLORATORY_BODIES) - bodies}")

    # Entrenamiento e inferencia deben leer exactamente la misma serie de escenas.
    # predict importa torch, que en CI (dependencias minimas) no esta instalado. Se importa
    # de forma tolerante: si falta torch se omiten este check y el contrato operacional de mas
    # abajo (solo corren en local, con artifacts/ + torch); los estaticos ya se ejecutaron.
    import match_pairs
    try:
        import predict
    except ModuleNotFoundError as e:
        predict = None
        print(f"(sin modulo '{e.name}' -> se omiten los checks que dependen de predict; "
              "en CI es esperado)")
    if predict is not None:
        chk("Entrenamiento e inferencia usan la misma serie de escenas",
            os.path.abspath(match_pairs.SCENE_FILE) == os.path.abspath(predict.SCENE),
            f"train={match_pairs.SCENE_FILE}; inferencia={predict.SCENE}")

    # La correccion de escala debe estar congelada antes de toda evaluacion final.
    correction_meta = os.path.join(C.DIR_OUT, "targets", "satellite_chl_correction_meta.json")
    if os.path.exists(correction_meta):
        meta = json.load(open(correction_meta, encoding="utf-8"))
        chk("Correccion del target usa un periodo de calibracion congelado",
            meta.get("calibration_end") == C.TARGET_CALIBRATION_END,
            f"meta={meta.get('calibration_end')}; config={C.TARGET_CALIBRATION_END}")

    # --- Checks que requieren los PARES (se omiten en CI si no hay datos) ---
    if not os.path.exists(PAIRS):
        print("(sin pares en disco -> solo checks estaticos; en CI esto es esperado)")
        _report(); return
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])

    # 4) Causalidad: para h>0 el target es ESTRICTAMENTE futuro
    viol = df[(df["horizon"] > 0) & (df["fecha_target"] <= df["fecha_t0"])]
    chk("Sin fuga temporal: target h>0 estrictamente futuro", len(viol) == 0,
        f"pares con target<=t0: {len(viol)}")

    # 5) gap_real dentro de la tolerancia declarada por horizonte
    okgap = True; det = []
    for h in [x for x in C.HORIZONS if x != 0]:
        lo, hi = C.HORIZON_TOLERANCE[h]
        g = df[df["horizon"] == h]["gap_real"]
        if len(g) and (g.min() < lo or g.max() > hi):
            okgap = False; det.append(f"h{h}:[{g.min()},{g.max()}]!~[{lo},{hi}]")
    chk("gap_real dentro de HORIZON_TOLERANCE", okgap, " ".join(det))

    # 6) Todas las FEATURES presentes en los pares
    missing = [f for f in FEATURES if f not in df.columns]
    chk("Todas las FEATURES presentes en los pares", not missing, f"faltan={missing}")

    if "target_age_t0_days" in df:
        max_target_age = float(df["target_age_t0_days"].max())
        chk("Target autorregresivo fresco en todos los pares",
            max_target_age <= C.MAX_TARGET_AGE_DAYS,
            f"max={max_target_age}; permitido={C.MAX_TARGET_AGE_DAYS}")
    else:
        chk("Target autorregresivo fresco en todos los pares", False,
            "falta target_age_t0_days; regenere match_pairs.py")

    # 8) Sin pares duplicados EXACTOS (fila identica). NB: misma fecha con varias escenas S2
    #    (tiles/pasadas distintas, espectro diferente) es legitimo y NO se cuenta como duplicado.
    dup_exact = df.duplicated().sum()
    multiscene = int(df.duplicated(["water_body", "horizon", "fecha_t0", "fecha_target"]).sum())
    chk("Sin pares duplicados exactos", dup_exact == 0,
        f"exactos={dup_exact} (multi-escena mismo dia, legitimo={multiscene})")

    # 9) Umbral de alerta por cuerpo presente y positivo
    chk("thr_body presente y > 0", "thr_body" in df.columns and (df["thr_body"] > 0).all())

    # 10) Modelos de produccion: features subset de columnas + cuantiles de incertidumbre presentes
    nested_path = os.path.join(C.DIR_REPORTS, "nested_metrics.json")
    nested = json.load(open(nested_path, encoding="utf-8")) if os.path.exists(nested_path) else {}
    okmodels, detm = True, []
    for pkl in glob.glob(os.path.join(C.DIR_MODELS, "*_h*.pkl")):
        if pkl.endswith("_nn.pt"):
            continue
        b = joblib.load(pkl)
        tag = os.path.basename(pkl)
        if not set(b.get("feats", [])).issubset(df.columns):
            okmodels = False; detm.append(f"{tag}: feats no en pares")
        if b.get("qlo") is None or b.get("qhi") is None or "q_conformal" not in b:
            okmodels = False; detm.append(f"{tag}: sin cuantiles de incertidumbre")
        group, horizon = b.get("group"), str(b.get("horizon"))
        expected_thresholds = nested.get(group, {}).get(horizon, {}).get("event_thresholds_from_dev")
        if (b.get("alert_label_source") != "nested_development_only" or
                b.get("event_thresholds") != expected_thresholds):
            okmodels = False; detm.append(f"{tag}: etiquetas no alineadas con DEV")
    chk("Modelos: features validas + intervalos (CQR) guardados", okmodels, " ".join(detm))

    # 11) Las features de los modelos no incluyen el target (doble chequeo en bundles)
    okf = True
    for pkl in glob.glob(os.path.join(C.DIR_MODELS, "*_h*.pkl")):
        if pkl.endswith("_nn.pt"):
            continue
        feats = joblib.load(pkl).get("feats", [])
        if any(f in target_cols for f in feats):
            okf = False
    chk("Features de los modelos sin columnas de target", okf)

    ok_calib, det_calib = True, []
    for group in ("freshwater", "marine"):
        path = os.path.join(C.DIR_MODELS, f"alert_calib_{group}.pkl")
        if not os.path.exists(path):
            ok_calib = False; det_calib.append(f"falta {os.path.basename(path)}")
            continue
        artifact = joblib.load(path)
        if artifact.get("label_source") != "nested_development_only_per_horizon":
            ok_calib = False; det_calib.append(f"{group}: etiquetas no alineadas")
    chk("Calibradores usan las mismas etiquetas validadas por horizonte", ok_calib,
        " ".join(det_calib))

    # Contrato de salida operacional: la banda contiene el punto y la bandera respeta
    # exactamente el umbral probabilistico guardado. Los cuerpos sin input fresco se omiten.
    # (requiere predict -> torch; si no esta disponible se omite este contrato).
    if predict is not None:
        coherent, checked, det_forecast = True, 0, []
        for water_body in predict.GROUP:
            forecast = predict.forecast_body(water_body)
            if forecast is None:
                continue
            for horizon in forecast["horizons"]:
                checked += 1
                band_ok = (horizon["p10"] is None or
                           horizon["p10"] <= horizon["chl_pred"] <= horizon["p90"])
                alert_ok = (horizon["alerta_anomalia"] ==
                            (horizon["prob_riesgo"] >= forecast["alert_threshold"]))
                if not band_ok or not alert_ok:
                    coherent = False
                    det_forecast.append(f"{water_body}+{horizon['horizon']}d")
        chk("Salida operacional: intervalo coherente y alerta reproducible", coherent,
            f"revisados={checked}; fallas={det_forecast}")
    _report()


def _report():
    print("=" * 68)
    print("CHECK DE INTEGRIDAD DEL PIPELINE (sin fuga / causal / consistente)")
    print("=" * 68)
    nfail = 0
    for desc, ok, det in CHECKS:
        mark = "[OK]  " if ok else "[FALLA]"
        line = f"{mark} {desc}"
        if not ok and det:
            line += f"  -> {det}"
        print(line)
        nfail += (not ok)
    print("=" * 68)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} OK" + ("" if nfail == 0 else f"  | {nfail} FALLA(S)"))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
