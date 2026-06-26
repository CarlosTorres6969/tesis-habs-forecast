"""
check_integrity.py — TEST DE INTEGRIDAD del pipeline (sin fuga, causal, consistente).

Convierte en aserciones reproducibles las verificaciones de honestidad del sistema. Corre sobre
los pares y los modelos de produccion. Exit 0 = todo OK; exit 1 = alguna falla (apto para CI).
Respalda en la defensa que el pronostico es causal y libre de fuga (a diferencia del sistema
viejo con AUC=1.0 por circularidad/shuffle).

Uso:  python check_integrity.py
"""
from __future__ import annotations
import os, sys, glob, joblib
import numpy as np
import pandas as pd
import config as C
from train import FEATURES, AUTOREG, PAIRS

CHECKS = []          # (descripcion, ok:bool, detalle:str)


def chk(desc, ok, detalle=""):
    CHECKS.append((desc, bool(ok), detalle))


def main():
    df = pd.read_csv(PAIRS, parse_dates=["fecha_t0", "fecha_target"])

    # 1) Ninguna feature es el target ni un patron prohibido (fuga)
    target_cols = {"log_chl_target", "chl_target", "hab_target", "fecha_target", "gap_real", "thr_body"}
    forbidden = ["delta", "target", "future", "lag14", "lag30", "t-14", "t-30"]
    bad = [f for f in FEATURES if f in target_cols or any(s in f.lower() for s in forbidden)]
    chk("Ninguna feature contaminada/prohibida en FEATURES", not bad, f"sospechosas={bad}")

    # 2) NDVI no es predictor (solo mascara/QA)
    chk("NDVI NO es predictor (solo QA)", "NDVI" not in FEATURES)

    # 3) Backbone autorregresivo presente y causal por nombre
    chk("Backbone autorregresivo (log_chl_t0) presente", "log_chl_t0" in FEATURES)

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

    # 7) target (log_chl_target) NO esta dentro de FEATURES
    chk("El target no aparece como feature", "log_chl_target" not in FEATURES)

    # 8) Sin pares duplicados EXACTOS (fila identica). NB: misma fecha con varias escenas S2
    #    (tiles/pasadas distintas, espectro diferente) es legitimo y NO se cuenta como duplicado.
    dup_exact = df.duplicated().sum()
    multiscene = int(df.duplicated(["water_body", "horizon", "fecha_t0", "fecha_target"]).sum())
    chk("Sin pares duplicados exactos", dup_exact == 0,
        f"exactos={dup_exact} (multi-escena mismo dia, legitimo={multiscene})")

    # 9) Umbral de alerta por cuerpo presente y positivo
    chk("thr_body presente y > 0", "thr_body" in df.columns and (df["thr_body"] > 0).all())

    # 10) Modelos de produccion: features subset de columnas + cuantiles de incertidumbre presentes
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

    # --- reporte ---
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
