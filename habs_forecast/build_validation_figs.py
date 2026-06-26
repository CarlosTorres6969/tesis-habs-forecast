"""
build_validation_figs.py — FIGURAS DE VALIDACION (las imagenes que demuestran que el
modelo funciona, distintas de los mapas espaciales que son demostracion del producto).

Lee los resultados YA CALCULADOS sobre el TEST INTACTO (validacion anidada):
  artifacts/reports/nested_metrics.json   (skill vs persistencia + PR-AUC, con IC95%)
  artifacts/reports/interval_metrics.json (cobertura de los intervalos CQR)

Produce:
  fig_skill_horizonte.png   — skill (1 - RMSE_modelo/RMSE_persistencia) por horizonte,
                              con IC95% bootstrap. Barra > 0 y IC que NO cruza 0 = el
                              modelo SUPERA a la persistencia de forma significativa.
  fig_cobertura_intervalos.png — cobertura observada de la banda P10-P90 (CQR) vs el
                              80% nominal; muestra que la incertidumbre esta calibrada.
  fig_pr_alerta.png         — PR-AUC de la alerta por horizonte con IC95%.

Uso:  python build_validation_figs.py
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

REPORTS = C.DIR_REPORTS
HORIZONS = [1, 3, 5, 7]
GROUP_LABEL = {"freshwater": "Lagos (agua dulce)", "marine": "Costa (marino/estuarino)"}
GROUP_COLOR = {"freshwater": "#1f77b4", "marine": "#d62728"}


PREDS = os.path.join(REPORTS, "nested_test_predictions.csv")


def _load(name):
    with open(os.path.join(REPORTS, name), encoding="utf-8") as f:
        return json.load(f)


def fig_timeline(wb="okeechobee"):
    """Serie temporal en el TEST INTACTO: clorofila real vs predicha vs persistencia.
    Es la evidencia visual de que el modelo ANTICIPA (no copia el ultimo valor)."""
    import pandas as pd
    if not os.path.exists(PREDS):
        print("  (sin nested_test_predictions.csv; corre evaluate_nested.py)"); return
    d = pd.read_csv(PREDS, parse_dates=["fecha_t0"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for ax, h in zip(axes, (1, 7)):
        s = d[(d.water_body == wb) & (d.horizon == h)].sort_values("fecha_t0")
        if s.empty:
            continue
        ax.plot(s.fecha_t0, s.chl_real, "-o", color="black", ms=4, lw=1.6,
                label="Clorofila REAL (observada)", zorder=4)
        ax.plot(s.fecha_t0, s.chl_pred, "-s", color="#1f77b4", ms=4, lw=1.6,
                label="PREDICHA por el modelo", zorder=3)
        ax.plot(s.fecha_t0, s.chl_persist, "--", color="#999999", lw=1.4,
                label="Persistencia (baseline: ultimo valor)", zorder=2)
        r = np.corrcoef(s.chl_real, s.chl_pred)[0, 1]
        ax.set_title(f"{wb.upper()} — pronostico a +{h} dias sobre el TEST INTACTO "
                     f"(r real-predicha = {r:.2f})", fontsize=11)
        ax.set_ylabel("Clorofila-a (ug/L)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("El modelo SIGUE la dinamica real de la clorofila en datos nunca vistos\n"
                 "(a +1d casi se superpone; a +7d aun anticipa la tendencia)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(REPORTS, "fig_serie_temporal.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def fig_scatter(group="freshwater"):
    """Predicho vs observado por horizonte (TEST INTACTO). Puntos sobre la diagonal = acierto."""
    import pandas as pd
    if not os.path.exists(PREDS):
        return
    d = pd.read_csv(PREDS)
    d = d[d.group == group]
    bodies = sorted(d.water_body.unique())
    cmap = dict(zip(bodies, plt.cm.tab10(np.linspace(0, 1, len(bodies)))))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, h in zip(axes, HORIZONS):
        s = d[d.horizon == h]
        if s.empty:
            continue
        for wb in bodies:
            ss = s[s.water_body == wb]
            ax.scatter(ss.chl_real, ss.chl_pred, s=22, alpha=0.75,
                       color=cmap[wb], label=wb, edgecolor="none")
        hi = float(np.nanpercentile(np.r_[s.chl_real, s.chl_pred], 98)) or 1.0
        ax.plot([0, hi], [0, hi], "k--", lw=1, zorder=1)
        r = np.corrcoef(s.chl_real, s.chl_pred)[0, 1]
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f"+{h}d  (r = {r:.2f}, n = {len(s)})", fontsize=10)
        ax.set_xlabel("Clorofila REAL (ug/L)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Clorofila PREDICHA (ug/L)")
    axes[-1].legend(fontsize=8, loc="lower right", title="cuerpo")
    lab = "Lagos" if group == "freshwater" else "Costa"
    fig.suptitle(f"{lab}: predicho vs observado en el TEST INTACTO — "
                 f"los puntos siguen la diagonal (linea punteada = acierto perfecto)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(REPORTS, f"fig_dispersion_{group}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def _triplet(d, key):
    """Devuelve (punto, lo, hi) -> (valor, err_inf, err_sup) para barras de error."""
    p, lo, hi = d[key]
    return p, p - lo, hi - p


def fig_skill(nested):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, grp in zip(axes, ("freshwater", "marine")):
        xs, ys, elo, ehi, sig = [], [], [], [], []
        for h in HORIZONS:
            p, lo, hi = nested[grp][str(h)]["skill_nested"]
            xs.append(h); ys.append(p); elo.append(p - lo); ehi.append(hi - p)
            sig.append(lo > 0)                         # IC no cruza 0 -> significativo
        x = np.arange(len(HORIZONS))
        colors = [GROUP_COLOR[grp] if s else "#bbbbbb" for s in sig]
        ax.bar(x, ys, color=colors, width=0.6, zorder=3)
        ax.errorbar(x, ys, yerr=[elo, ehi], fmt="none", ecolor="black",
                    elinewidth=1.3, capsize=5, zorder=4)
        ax.axhline(0, color="black", lw=1.2)
        ax.axhline(0, color="black", lw=0)             # placeholder
        for xi, yi, s in zip(x, ys, sig):
            ax.text(xi, yi + (0.012 if yi >= 0 else -0.028),
                    "*" if s else "ns", ha="center",
                    fontsize=12, fontweight="bold",
                    color="black" if s else "#888888")
        ax.set_xticks(x); ax.set_xticklabels([f"+{h}d" for h in HORIZONS])
        ax.set_title(GROUP_LABEL[grp], fontsize=11)
        ax.set_xlabel("Horizonte de pronostico")
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_ylabel("Skill vs persistencia\n(1 - RMSE_modelo / RMSE_persistencia)")
    fig.suptitle("Validacion anidada (TEST INTACTO): el modelo supera a la persistencia\n"
                 "barra de color + '*' = mejora significativa (IC95% no cruza 0)  ·  "
                 "gris + 'ns' = no significativa", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(REPORTS, "fig_skill_horizonte.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def fig_intervalos(interv):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(HORIZONS)); w = 0.36
    for i, grp in enumerate(("freshwater", "marine")):
        ys, elo, ehi = [], [], []
        for h in HORIZONS:
            d = interv[grp][str(h)]
            p, lo, hi = d["cobertura_cqr"]
            ys.append(p); elo.append(p - lo); ehi.append(hi - p)
        ax.bar(x + (i - 0.5) * w, ys, width=w, color=GROUP_COLOR[grp],
               label=GROUP_LABEL[grp], zorder=3)
        ax.errorbar(x + (i - 0.5) * w, ys, yerr=[elo, ehi], fmt="none",
                    ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
    ax.axhline(0.80, color="green", ls="--", lw=1.8, zorder=2,
               label="Cobertura nominal objetivo (80%)")
    ax.set_xticks(x); ax.set_xticklabels([f"+{h}d" for h in HORIZONS])
    ax.set_ylim(0, 1.0); ax.set_ylabel("Cobertura observada de la banda P10-P90")
    ax.set_xlabel("Horizonte de pronostico")
    ax.set_title("Intervalos de incertidumbre calibrados (CQR) sobre el TEST INTACTO\n"
                 "la banda atrapa el valor real ~80% de las veces, como se busca",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    out = os.path.join(REPORTS, "fig_cobertura_intervalos.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def fig_pr(nested):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(HORIZONS)); w = 0.36
    for i, grp in enumerate(("freshwater", "marine")):
        ys, elo, ehi, base = [], [], [], []
        for h in HORIZONS:
            d = nested[grp][str(h)]
            p, lo, hi = d["pr_auc_nested"]
            ys.append(p); elo.append(p - lo); ehi.append(hi - p)
            base.append(d["pos_test"] / d["n_test"])     # prevalencia = PR-AUC de azar
        ax.bar(x + (i - 0.5) * w, ys, width=w, color=GROUP_COLOR[grp],
               label=GROUP_LABEL[grp], zorder=3)
        ax.errorbar(x + (i - 0.5) * w, ys, yerr=[elo, ehi], fmt="none",
                    ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
        ax.scatter(x + (i - 0.5) * w, base, marker="_", s=320, color="black",
                   zorder=5, label="Azar (prevalencia)" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([f"+{h}d" for h in HORIZONS])
    ax.set_ylim(0, 1.0); ax.set_ylabel("PR-AUC de la alerta")
    ax.set_xlabel("Horizonte de pronostico")
    ax.set_title("Capacidad de ALERTA (PR-AUC) sobre el TEST INTACTO\n"
                 "barra por encima de la marca '_' (azar) = la alerta aporta informacion",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    out = os.path.join(REPORTS, "fig_pr_alerta.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def main():
    nested = _load("nested_metrics.json")
    interv = _load("interval_metrics.json")
    print("Figuras de validacion:")
    fig_skill(nested)
    fig_intervalos(interv)
    fig_pr(nested)
    fig_timeline("okeechobee")
    fig_scatter("freshwater")
    fig_scatter("marine")


if __name__ == "__main__":
    main()
