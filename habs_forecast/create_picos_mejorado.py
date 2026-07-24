"""
Versión mejorada del gráfico de "picos anuales".
Problema del original: usaba la MEDIA como barra, que aplana los eventos
(en sitios marinos la media se mantiene ~5 ug/L aunque haya picos de 50-65).

Esta versión pone el PICO (máximo anual) como protagonista, más el P90,
la media como línea base y el rango de variación sombreado. Así la
magnitud de las floraciones se ve, que es lo que importa para HABs.

Salida: entregables/clima_nutrientes_tesis/fig1b_picos_anuales_mejorado.{png,pdf}
Autor: Sistema HABs Forecast
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = Path("entregables") / "clima_nutrientes_tesis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_CONFIG = {
    'cajon':      {'color': '#E63946', 'nombre': 'El Cajón',          'group': 'freshwater'},
    'yojoa':      {'color': '#F4A261', 'nombre': 'Lago Yojoa',        'group': 'freshwater'},
    'okeechobee': {'color': '#264653', 'nombre': 'Lake Okeechobee',   'group': 'freshwater'},
    'fonseca':    {'color': '#2A9D8F', 'nombre': 'Golfo de Fonseca',  'group': 'marine'},
    'tampa_bay':  {'color': '#457B9D', 'nombre': 'Tampa Bay',         'group': 'marine'},
}
# Orden: dulces primero, marinos al final
ORDER = ['cajon', 'yojoa', 'okeechobee', 'fonseca', 'tampa_bay']

THRESH = {'freshwater': 20.0, 'marine': 10.0}  # umbral HAB clorofila-a (ug/L)


def load():
    chl = pd.read_csv(ARTIFACTS_DIR / "targets" / "combined_target.csv")
    chl['fecha'] = pd.to_datetime(chl['fecha'])
    chl['year'] = chl['fecha'].dt.year
    era5 = pd.read_csv(ARTIFACTS_DIR / "state_series" / "era5_daily.csv")
    era5['fecha'] = pd.to_datetime(era5['fecha'])
    era5['year'] = era5['fecha'].dt.year
    return chl, era5


def main():
    chl, era5 = load()

    # Crear gráficos individuales para cada sitio
    for site in ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        cfg = SITE_CONFIG[site]
        color = cfg['color']
        thr = THRESH[cfg['group']]

        d = chl[chl['water_body'] == site]
        g = d.groupby('year')['chl_ugl'].agg(
            media='mean',
            p90=lambda s: s.quantile(0.90),
            maximo='max',
        ).reset_index()
        years = g['year'].values
        x = np.arange(len(years))

        # Rango de variación: banda de media -> máximo (SIN ETIQUETA EN LEYENDA)
        ax.vlines(x, g['media'], g['maximo'], color=color, alpha=0.25, linewidth=14)

        # Barra = PICO anual (protagonista)
        ax.bar(x, g['maximo'], width=0.60, color=color, alpha=0.85,
               label='Pico anual (max)', zorder=3, edgecolor='black', linewidth=0.8)

        # P90 y media como marcadores
        ax.plot(x, g['p90'], 's', color='black', markersize=9, zorder=4,
                markerfacecolor='white', markeredgewidth=2, label='P90')
        ax.plot(x, g['media'], 'o', color='black', markersize=8, zorder=4,
                label='Media')

        # Umbral HAB
        ax.axhline(y=thr, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
                   label=f'Umbral HAB ({thr:.0f} μg/L)')

        # Precipitación anual (contexto) en eje secundario
        pr = era5[era5['water_body'] == site].groupby('year')['precipitation'].sum().reset_index()
        pr = pr[pr['year'].isin(years)]
        if len(pr):
            ax2 = ax.twinx()
            ax2.plot(x[:len(pr)], pr['precipitation'], color='steelblue', linewidth=2.2,
                     marker='^', markersize=7, alpha=0.60, linestyle=':')
            ax2.set_ylabel('Precipitación anual (mm)', color='steelblue', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
            ax2.grid(False)

        # Etiqueta del valor del pico sobre cada barra
        for xi, mx in zip(x, g['maximo']):
            ax.text(xi, mx, f'{mx:.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color, zorder=5)

        tag = 'AGUA DULCE' if cfg['group'] == 'freshwater' else 'MARINO'
        ax.set_title(f"Picos Anuales de Clorofila-a: {cfg['nombre']} ({tag})", 
                    fontweight='bold', fontsize=15, color=color, pad=15)
        ax.set_ylabel('Clorofila-a (μg/L)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Año', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([int(y) for y in years], fontsize=11)
        ax.grid(True, axis='y', alpha=0.35, which='major')
        ax.grid(True, axis='y', alpha=0.15, which='minor', linestyle=':')
        
        # Mejorar eje Y con más valores
        max_val = g['maximo'].max()
        if max_val <= 30:
            y_max = int(np.ceil(max_val / 5) * 5) + 5
            yticks = np.arange(0, y_max + 1, 2)
        elif max_val <= 60:
            y_max = int(np.ceil(max_val / 10) * 10) + 10
            yticks = np.arange(0, y_max + 1, 5)
        elif max_val <= 120:
            y_max = int(np.ceil(max_val / 10) * 10) + 10
            yticks = np.arange(0, y_max + 1, 10)
        else:
            y_max = int(np.ceil(max_val / 20) * 20) + 20
            yticks = np.arange(0, y_max + 1, 20)
        
        ax.set_ylim(0, y_max)
        ax.set_yticks(yticks)
        ax.yaxis.set_minor_locator(plt.MultipleLocator((yticks[1] - yticks[0]) / 2) if len(yticks) > 1 else plt.AutoMinorLocator())
        
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95, frameon=True, shadow=True, ncol=1)

        plt.tight_layout()
        filename = f"fig1c_picos_anuales_{site}"
        png = OUTPUT_DIR / f"{filename}.png"
        pdf = OUTPUT_DIR / f"{filename}.pdf"
        plt.savefig(png, bbox_inches='tight', dpi=300)
        plt.savefig(pdf, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Guardado: {filename}")
    
    print("✓ Figura de picos anuales guardada: 5 archivos individuales")


if __name__ == "__main__":
    main()
