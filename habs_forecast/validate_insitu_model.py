"""
validate_insitu_model.py — VALIDACION DEL MODELO CONTRA VERDAD DE CAMPO (in-situ).

Confronta las PREDICCIONES del modelo de produccion con mediciones de clorofila-a IN-SITU
reales (gold standard), no contra otro satelite. Es la validacion externa mas fuerte.

Diseno (causal, sin fuga):
  Para cada medida in-situ (fecha D, chl_real) y cada horizonte h en {1,3,5,7}:
    - se busca una escena Sentinel-2 en t0 tal que el gap (D - t0) caiga en la tolerancia de h,
    - se construyen las features en t0 (reusando predict.build_features; solo datos <= t0),
    - se predice chl(t0+h) con el modelo de produccion y se compara con el in-situ en D.
  El in-situ NO se usa para entrenar -> prueba externa. Baseline: persistencia (chl en t0).

Metricas por horizonte: n, Pearson/Spearman (seguimiento temporal), MAE del modelo y de la
persistencia, y SKILL de campo (1 - MAE_modelo/MAE_persistencia).

NOTA honesta: (1) solo Okeechobee tiene chl-a in-situ en 2023-2026 (Honduras no tiene;
Yojoa solo Secchi <=2022). (2) El target satelital de Okeechobee fue escalado a in-situ
(bias_correct_target), asi que la ESCALA absoluta no es 100% independiente; lo que SI es
independiente es el SEGUIMIENTO TEMPORAL (correlacion) y el batir a la persistencia contra
campo (ambos usan la misma escala -> comparacion justa). (3) in-situ = punto/estacion vs
prediccion = agregado del cuerpo: hay ruido espacial inevitable (se declara).

Salida: artifacts/reports/insitu_model_validation.csv (+ resumen) y figuras
  fig_insitu_dispersion.png, fig_insitu_serie.png

Uso:  python validate_insitu_model.py
"""
from __future__ import annotations
import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import config as C
from predict import build_features, GROUP, SCENE, _load

INSITU = os.path.join(C.DIR_OUT, "targets", "insitu_chl.csv")
MODELS = C.DIR_MODELS
HORIZONS = [1, 3, 5, 7]
OUT_CSV = os.path.join(C.DIR_REPORTS, "insitu_model_validation.csv")
OUT_SUM = os.path.join(C.DIR_REPORTS, "insitu_model_validation_summary.csv")


def _match_scene(scene_dates, D, h):
    """Escena t0 cuyo gap (D - t0) cae en la tolerancia de h; la mas cercana al nominal h."""
    lo, hi = C.HORIZON_TOLERANCE[h]
    cand = [t0 for t0 in scene_dates if lo <= (D - t0).days <= hi]
    if not cand:
        return None
    return min(cand, key=lambda t0: abs((D - t0).days - h))


def validate_body(wb):
    group = GROUP[wb]
    ins = pd.read_csv(INSITU, parse_dates=["fecha"])
    ins = ins[ins["water_body"] == wb]
    if ins.empty:
        return pd.DataFrame()
    # AGREGADO POR FECHA: el modelo predice un valor del CUERPO; el in-situ tiene varias estaciones
    # por dia con gran dispersion espacial (lago grande/heterogeneo). Comparar agregado vs agregado
    # (mediana de estaciones del dia) es lo justo; comparar contra un punto suelto mezcla la
    # variabilidad ESPACIAL (intra-lago) con el error del pronostico.
    ins = (ins.groupby(ins["fecha"].dt.normalize())
              .agg(chl_ugl=("chl_ugl", "median"), n_est=("chl_ugl", "size"))
              .reset_index().sort_values("fecha"))
    scene_dates = sorted(_load(SCENE, wb)["fecha"].unique())
    scene_dates = [pd.Timestamp(d) for d in scene_dates]

    rows = []
    for h in HORIZONS:
        bundle = joblib.load(os.path.join(MODELS, f"{group}_h{h}.pkl"))
        feats = bundle["feats"]
        for _, r in ins.iterrows():
            D, y_real = r["fecha"], r["chl_ugl"]
            t0 = _match_scene(scene_dates, D, h)
            if t0 is None:
                continue
            built = build_features(wb, t0)
            if built is None:
                continue
            X, chl0, _ = built
            y_pred = float(np.expm1(bundle["reg"].predict(X.reindex(columns=feats))[0]))
            rows.append({"water_body": wb, "horizon": h, "fecha_insitu": D.date().isoformat(),
                         "t0": t0.date().isoformat(), "gap": (D - t0).days,
                         "chl_insitu": float(y_real), "chl_pred": max(y_pred, 0.0),
                         "chl_persist": float(chl0)})
    return pd.DataFrame(rows)


def anchor_target_vs_insitu(wb, tol_days=2):
    """ANCLA de campo: ¿el target satelital que el modelo pronostica sigue la realidad in-situ?
    Correlaciona el target combinado (lo que el modelo aprende a predecir) con el in-situ agregado
    por fecha, emparejando por fecha cercana (<= tol_days). Es el FUNDAMENTO de toda la cadena:
    si el target sigue al campo, un buen pronostico del target es un buen pronostico del campo."""
    ins = pd.read_csv(INSITU, parse_dates=["fecha"])
    ins = ins[ins["water_body"] == wb]
    if ins.empty:
        return None
    daily = (ins.groupby(ins["fecha"].dt.normalize())["chl_ugl"].median()
             .reset_index().rename(columns={"chl_ugl": "chl_insitu"}).sort_values("fecha"))
    tg = pd.read_csv(os.path.join(C.DIR_OUT, "targets", "combined_target.csv"), parse_dates=["fecha"])
    tg = tg[tg["water_body"] == wb][["fecha", "chl_ugl"]].copy()
    tg["fecha"] = pd.to_datetime(tg["fecha"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    m = pd.merge_asof(daily.sort_values("fecha"), tg.sort_values("fecha"), on="fecha",
                      direction="nearest", tolerance=pd.Timedelta(days=tol_days)).dropna()
    if len(m) < 4:
        return {"n": len(m)}
    return {"n": int(len(m)), "pearson": round(float(pearsonr(m["chl_insitu"], m["chl_ugl"])[0]), 3),
            "spearman": round(float(spearmanr(m["chl_insitu"], m["chl_ugl"])[0]), 3)}


def summarize(detail):
    out = []
    for h in HORIZONS:
        d = detail[detail["horizon"] == h]
        if len(d) < 5:
            out.append({"horizon": h, "n": len(d)}); continue
        yr, yp, yq = d["chl_insitu"].values, d["chl_pred"].values, d["chl_persist"].values
        mae_m = float(np.mean(np.abs(yp - yr)))
        mae_p = float(np.mean(np.abs(yq - yr)))
        pr = pearsonr(yr, yp)[0]; sp = spearmanr(yr, yp)[0]
        out.append({"horizon": h, "n": len(d),
                    "pearson": round(pr, 3), "spearman": round(sp, 3),
                    "MAE_modelo": round(mae_m, 2), "MAE_persistencia": round(mae_p, 2),
                    "skill_campo": round(1 - mae_m / mae_p, 3) if mae_p > 0 else np.nan})
    return pd.DataFrame(out)


def fig_dispersion(detail, wb):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, h in zip(axes, HORIZONS):
        d = detail[detail["horizon"] == h]
        if len(d) < 3:
            ax.set_title(f"+{h}d (n={len(d)})"); continue
        ax.scatter(d["chl_insitu"], d["chl_pred"], s=26, alpha=0.7, color="#1f77b4", edgecolor="none")
        hi = float(np.nanpercentile(np.r_[d["chl_insitu"], d["chl_pred"]], 98)) or 1.0
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        r = pearsonr(d["chl_insitu"], d["chl_pred"])[0]
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f"+{h}d  (r={r:.2f}, n={len(d)})", fontsize=10)
        ax.set_xlabel("Clorofila-a IN-SITU (ug/L)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Clorofila-a PREDICHA (ug/L)")
    fig.suptitle(f"{wb.upper()}: prediccion del modelo vs verdad de campo (in-situ) — "
                 f"validacion externa por horizonte", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(C.DIR_REPORTS, "fig_insitu_dispersion.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def fig_serie(detail, wb, h=1):
    d = detail[detail["horizon"] == h].copy()
    if d.empty:
        return
    d["fecha"] = pd.to_datetime(d["fecha_insitu"])
    d = d.sort_values("fecha")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(d["fecha"], d["chl_insitu"], "-o", color="black", ms=4, label="IN-SITU (campo, real)")
    ax.plot(d["fecha"], d["chl_pred"], "-s", color="#1f77b4", ms=4, label=f"Predicho por el modelo (+{h}d)")
    ax.plot(d["fecha"], d["chl_persist"], "--", color="#999", lw=1.2, label="Persistencia")
    ax.set_ylabel("Clorofila-a (ug/L)"); ax.set_xlabel("Fecha"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{wb.upper()}: el modelo sigue la clorofila in-situ de campo (+{h}d)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(C.DIR_REPORTS, "fig_insitu_serie.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def main():
    if not os.path.exists(INSITU):
        print(f"Sin in-situ ({INSITU})."); return
    ins = pd.read_csv(INSITU)
    bodies = [wb for wb in ins["water_body"].unique() if wb in GROUP]
    print(f"Cuerpos con in-situ chl Y modelo: {bodies}")
    if not bodies:
        print("Ningun cuerpo in-situ coincide con los modelados (Honduras sin in-situ)."); return

    all_detail = []
    for wb in bodies:
        d = validate_body(wb)
        if d.empty:
            print(f"  {wb}: sin matchups escena<->in-situ en tolerancia."); continue
        all_detail.append(d)
        print(f"\n=== {wb.upper()} — validacion contra campo (in-situ) ===")
        anc = anchor_target_vs_insitu(wb)
        if anc and "pearson" in anc:
            print(f"ANCLA (target satelital vs in-situ, n={anc['n']}): "
                  f"Pearson={anc['pearson']:+.2f}  Spearman={anc['spearman']:+.2f}  "
                  f"-> lo que el modelo pronostica SI sigue al campo")
        nfechas = d["fecha_insitu"].nunique()
        print(f"(in-situ utilizable: {nfechas} fechas con escena alineada; cobertura LIMITADA)")
        s = summarize(d)
        print(s.to_string(index=False))
        fig_dispersion(d, wb)
        fig_serie(d, wb, h=1)
    if not all_detail:
        print("Sin matchups."); return
    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(OUT_CSV, index=False)
    summ = pd.concat([summarize(detail[detail.water_body == wb]).assign(water_body=wb)
                      for wb in detail.water_body.unique()], ignore_index=True)
    summ.to_csv(OUT_SUM, index=False)
    print(f"\nDetalle -> {OUT_CSV}\nResumen -> {OUT_SUM}")
    print("\nLectura: Pearson/Spearman = seguimiento temporal (independiente del bias). "
          "skill_campo>0 = el modelo le gana a la persistencia CONTRA verdad de campo.")


if __name__ == "__main__":
    main()
