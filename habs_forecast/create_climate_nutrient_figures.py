"""
Script para generar gráficos de relaciones clima-nutrientes-clorofila
Análisis de factores causales de florecimientos algales
Autor: Sistema HABs Forecast
Fecha: 2026-07-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuración de estilo
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'

# Directorios
ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = Path("entregables") / "clima_nutrientes_tesis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración por sitio
SITE_CONFIG = {
    'cajon': {'color': '#E63946', 'nombre': 'El Cajón'},
    'yojoa': {'color': '#F4A261', 'nombre': 'Lago Yojoa'},
    'fonseca': {'color': '#2A9D8F', 'nombre': 'Golfo de Fonseca'},
    'okeechobee': {'color': '#264653', 'nombre': 'Lake Okeechobee'},
    'tampa_bay': {'color': '#457B9D', 'nombre': 'Tampa Bay'}
}

def load_data():
    """Carga todos los datos necesarios"""
    # Clorofila
    chl = pd.read_csv(ARTIFACTS_DIR / "targets" / "combined_target.csv")
    chl['fecha'] = pd.to_datetime(chl['fecha'])
    chl['year'] = chl['fecha'].dt.year
    chl['month'] = chl['fecha'].dt.month
    
    # Nutrientes
    nutrients = pd.read_csv(ARTIFACTS_DIR / "targets" / "nutrients_daily.csv")
    nutrients['fecha'] = pd.to_datetime(nutrients['fecha'])
    nutrients['year'] = nutrients['fecha'].dt.year
    nutrients['month'] = nutrients['fecha'].dt.month
    
    # Clima (ERA5)
    era5 = pd.read_csv(ARTIFACTS_DIR / "state_series" / "era5_daily.csv")
    era5['fecha'] = pd.to_datetime(era5['fecha'])
    era5['year'] = era5['fecha'].dt.year
    era5['month'] = era5['fecha'].dt.month
    
    return chl, nutrients, era5


def _boxplot_style():
    """Estilo de caja y bigote consistente con fig5: mediana negra gruesa,
    bigotes/caps negros y outliers como círculos huecos."""
    return dict(
        patch_artist=True,
        showfliers=True,
        whis=(5, 95),  # bigotes al P5-P95 (no 1.5*IQR): rango real, menos nube de outliers
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        boxprops=dict(edgecolor='black', linewidth=1.0),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='none',
                        markeredgecolor='black', alpha=0.5),
    )


def _fill_boxes(bp, color, box_alpha=0.7):
    """Colorea el relleno de las cajas (identifica el sitio / la variable)."""
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(box_alpha)


def _dibujar_simbologia(ax):
    """Dibuja un box-and-whisker esquemático con etiquetas para leer la figura."""
    ax.clear()
    stats = [{'med': 5.0, 'q1': 3.6, 'q3': 6.6, 'whislo': 2.0, 'whishi': 9.0,
              'fliers': [11.5], 'label': ''}]
    style = _boxplot_style()
    style.pop('whis', None)          # bxp usa estadísticos ya calculados
    style.pop('showfliers', None)
    bp = ax.bxp(stats, positions=[0], widths=0.5, showfliers=True, **style)
    for patch in bp['boxes']:
        patch.set_facecolor('#B0B0B0')
        patch.set_alpha(0.7)

    ann = dict(fontsize=10.5, va='center', ha='left',
               arrowprops=dict(arrowstyle='-', color='gray', lw=1.1))
    ax.annotate('Mediana', xy=(0.25, 5.0), xytext=(0.85, 5.0), **ann)
    ax.annotate('Caja: Q1–Q3\n(50% central de los datos)',
                xy=(0.25, 4.1), xytext=(0.85, 3.2), **ann)
    ax.annotate('Bigote: percentil 5–95', xy=(0.0, 9.0), xytext=(0.55, 10.4), **ann)
    ax.annotate('Puntos: valores fuera\nde P5–P95 (picos)',
                xy=(0.0, 11.5), xytext=(0.55, 12.6), **ann)

    ax.set_title('Simbología', fontweight='bold', fontsize=14, pad=8)
    ax.set_xlim(-0.6, 3.4)
    ax.set_ylim(0.5, 14.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.02, -0.02,
            'Eje Y en escala logarítmica · cada panel tiene su propia escala\n'
            'El color identifica el sitio',
            transform=ax.transAxes, fontsize=9.5, style='italic',
            color='#444444', va='top')


def fig1_picos_anuales_integrado(chl, nutrients, era5):
    """Figura 1: Distribución anual de clorofila-a por sitio (caja y bigote) - VERSIÓN INDIVIDUAL"""
    
    all_sites = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']

    # Crear figura individual para cada sitio (más legible)
    for site in all_sites:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        color = SITE_CONFIG[site]['color']
        nombre = SITE_CONFIG[site]['nombre']

        site_chl = chl[chl['water_body'] == site]
        years = sorted(site_chl['year'].unique())
        xpos = list(range(len(years)))
        chl_groups = [site_chl[site_chl['year'] == y]['chl_ugl'].dropna().values
                      for y in years]

        bp = ax.boxplot(chl_groups, positions=xpos, widths=0.55,
                        manage_ticks=False, **_boxplot_style())
        _fill_boxes(bp, color)

        ax.set_title(f'Distribución Anual de Clorofila-a: {nombre}', 
                    fontweight='bold', fontsize=14, color=color, pad=10)
        ax.set_ylabel('Clorofila-a (μg/L)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Año', fontsize=12, fontweight='bold')
        ax.set_xticks(xpos)
        ax.set_xticklabels([int(y) for y in years])
        ax.set_xlim(-0.6, len(years) - 0.4)
        
        # Escala log con etiquetas específicas (no notación científica)
        all_vals = site_chl['chl_ugl'].dropna().values
        lo = max(0.8, np.percentile(all_vals, 2) * 0.8)
        hi = all_vals.max() * 1.25
        ax.set_yscale('log')
        ax.set_ylim(lo, hi)
        
        # Desactivar notación científica y usar formato de números reales
        from matplotlib.ticker import ScalarFormatter, FixedLocator
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        
        # Establecer ticks específicos con más detalle según el rango de datos
        if hi <= 20:
            yticks = [1, 2, 3, 5, 7, 10, 15, 20]
        elif hi <= 50:
            yticks = [1, 2, 5, 10, 15, 20, 30, 40, 50]
        elif hi <= 100:
            yticks = [1, 2, 5, 10, 20, 30, 50, 70, 100]
        else:
            yticks = [1, 5, 10, 20, 50, 100, 150, 200]
        
        # Filtrar solo los ticks que están en el rango visible
        yticks_visible = [t for t in yticks if lo <= t <= hi]
        ax.set_yticks(yticks_visible)
        
        # Formatear etiquetas para que sean enteros legibles
        ax.set_yticklabels([str(int(t)) if t >= 1 else f'{t:.1f}' for t in yticks_visible])
        
        ax.grid(True, alpha=0.3, axis='y', which='major')
        
        plt.tight_layout()
        filename = f"fig1a_boxplot_anual_{site}"
        plt.savefig(OUTPUT_DIR / f"{filename}.png", bbox_inches='tight', dpi=300)
        plt.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches='tight')
        plt.close()
        print(f"  ✓ Guardado: {filename}")
    
    print("✓ Figura 1 guardada: distribución anual (5 archivos individuales)")


def fig2_relacion_lluvia_clorofila(chl, era5):
    """Figura 2: Relación entre época de lluvias y clorofila por sitio"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    sites = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    
    for idx, site in enumerate(sites):
        ax = axes[idx]
        
        # Merge clorofila con precipitación
        site_chl = chl[chl['water_body'] == site][['fecha', 'chl_ugl', 'month']].copy()
        site_era5 = era5[era5['water_body'] == site][['fecha', 'precipitation']].copy()
        
        # Agregar por mes
        monthly_chl = site_chl.groupby('month')['chl_ugl'].mean().reset_index()
        monthly_precip = site_era5.groupby(site_era5['fecha'].dt.month)['precipitation'].mean().reset_index()
        monthly_precip.columns = ['month', 'precipitation']
        
        # Crear twin axis
        ax2 = ax.twinx()
        
        # Barras de precipitación
        bars = ax.bar(monthly_precip['month'], monthly_precip['precipitation'],
                     color='steelblue', alpha=0.4, label='Precipitación (mm/día)')
        
        # Línea de clorofila
        line = ax2.plot(monthly_chl['month'], monthly_chl['chl_ugl'],
                       color=SITE_CONFIG[site]['color'], linewidth=3,
                       marker='o', markersize=8, label='Clorofila-a')
        
        # Área sombreada para época de lluvias (identificar meses >umbral)
        precip_threshold = monthly_precip['precipitation'].quantile(0.75)
        rainy_months = monthly_precip[monthly_precip['precipitation'] > precip_threshold]['month'].values
        
        for month in rainy_months:
            ax.axvspan(month - 0.5, month + 0.5, alpha=0.15, color='blue', zorder=0)
        
        # Etiquetas
        month_names = ['E', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax.set_xlabel('Mes', fontweight='bold')
        ax.set_ylabel('Precipitación (mm/día)', fontweight='bold', color='steelblue')
        ax2.set_ylabel('Clorofila-a (μg/L)', fontweight='bold', 
                      color=SITE_CONFIG[site]['color'])
        ax.set_title(f"{SITE_CONFIG[site]['nombre']}", fontweight='bold', fontsize=12)
        
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor=SITE_CONFIG[site]['color'])
        
        # Leyenda combinada
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # Correlación
        merged = pd.merge(monthly_chl, monthly_precip, on='month')
        corr = merged['chl_ugl'].corr(merged['precipitation'])
        ax.text(0.98, 0.02, f'Correlación: {corr:.3f}',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # Ocultar último subplot
    axes[-1].axis('off')
    
    plt.suptitle('Relación entre Época de Lluvias y Concentración de Clorofila',
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_lluvia_clorofila.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig2_lluvia_clorofila.pdf", bbox_inches='tight')
    print("✓ Figura 2 guardada: Relación lluvia-clorofila")
    plt.close()


def fig3_scatter_nutrientes_clorofila(chl, nutrients):
    """Figura 3: Relación nutrientes vs clorofila (scatter plots)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sites_with_nutrients = ['okeechobee', 'tampa_bay']
    
    for idx, site in enumerate(sites_with_nutrients):
        ax = axes[idx]
        
        # Merge por fecha
        site_chl = chl[chl['water_body'] == site][['fecha', 'chl_ugl']].copy()
        site_nut = nutrients[nutrients['water_body'] == site][['fecha', 'tp_mgl']].copy()
        
        merged = pd.merge(site_chl, site_nut, on='fecha', how='inner')
        
        # Scatter plot
        scatter = ax.scatter(merged['tp_mgl'], merged['chl_ugl'],
                            c=SITE_CONFIG[site]['color'], alpha=0.6, s=60,
                            edgecolors='black', linewidth=0.5)
        
        # Línea de tendencia
        if len(merged) > 2:
            z = np.polyfit(merged['tp_mgl'], merged['chl_ugl'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(merged['tp_mgl'].min(), merged['tp_mgl'].max(), 100)
            ax.plot(x_trend, p(x_trend), "--", color='red', linewidth=2, 
                   label=f'Tendencia: y={z[0]:.1f}x+{z[1]:.1f}')
        
        # Correlación
        corr = merged['tp_mgl'].corr(merged['chl_ugl'])
        r2 = corr ** 2
        
        # Etiquetas
        ax.set_xlabel('Fósforo Total (mg/L)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Clorofila-a (μg/L)', fontweight='bold', fontsize=12)
        ax.set_title(f"{SITE_CONFIG[site]['nombre']}\nn={len(merged)}", 
                    fontweight='bold', fontsize=13)
        
        # Estadísticas
        stats_text = f'Correlación (r): {corr:.3f}\nR²: {r2:.3f}\np-value: '
        from scipy import stats as sp_stats
        _, pval = sp_stats.pearsonr(merged['tp_mgl'], merged['chl_ugl'])
        stats_text += f'{pval:.4f}'
        
        ax.text(0.05, 0.95, stats_text,
               transform=ax.transAxes, va='top', ha='left',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Relación entre Fósforo Total y Clorofila-a',
                fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_nutrientes_clorofila.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig3_nutrientes_clorofila.pdf", bbox_inches='tight')
    print("✓ Figura 3 guardada: Relación nutrientes-clorofila")
    plt.close()


def _nota_correlaciones(ax):
    """Recuadro que explica por qué sitios del mismo tipo (lagos / costas) no
    muestran las correlaciones parecidas que uno esperaría."""
    ax.clear()
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                               facecolor='#FFF8E1', edgecolor='#D9B44A',
                               linewidth=1.6, zorder=0))
    ax.text(0.5, 0.955, '¿Por qué sitios del mismo tipo\nno dan resultados similares?',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12.5, fontweight='bold', color='#6b5310')

    cuerpo = (
        'Se esperaría que los lagos (Cajón, Yojoa,\n'
        'Okeechobee) entre sí —y las costas (Fonseca,\n'
        'Tampa) entre sí— tuvieran correlaciones\n'
        'parecidas por ser ambientes similares.\n'
        'Pero no ocurre:\n'
        '\n'
        '•  Lagos: driver dominante distinto en c/u\n'
        '      – Cajón → lluvia (+0.30)\n'
        '      – Yojoa → viento (+0.20)\n'
        '      – Okeechobee → temperatura (+0.27)\n'
        '•  Costas:\n'
        '      – Fonseca → casi nulo (todo |r|<0.06)\n'
        '      – Tampa → viento (+0.22)\n'
        '\n'
        '¿Por qué? El "tipo" de cuerpo de agua no\n'
        'define la respuesta al clima. Pesan más los\n'
        'factores locales: profundidad y mezcla,\n'
        'morfología, fuentes de nutrientes, latitud/\n'
        'estacionalidad e hidrodinámica, y la\n'
        'cantidad y época de datos disponibles.\n'
        '\n'
        'Además todas las correlaciones son débiles\n'
        '(|r|<0.3): el clima por sí solo no explica la\n'
        'clorofila.  Correlación ≠ causa.'
    )
    ax.text(0.06, 0.87, cuerpo, transform=ax.transAxes, ha='left', va='top',
            fontsize=8.7, color='#2b2b2b', linespacing=1.35)


def fig4_heatmap_correlaciones(chl, era5):
    """Figura 4: Correlaciones climáticas - VERSIÓN INDIVIDUAL POR SITIO"""
    
    sites = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    
    # Variables climáticas de interés
    climate_vars = {
        'temp_air_2m': 'Temperatura (°C)',
        'precipitation': 'Precipitación (mm)',
        'solar_radiation': 'Radiación Solar (J/m²)',
        'wind_speed_10m': 'Velocidad Viento (m/s)'
    }
    
    # Crear figura individual para cada sitio (más legible)
    for site in sites:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Merge datos
        site_chl = chl[chl['water_body'] == site][['fecha', 'chl_ugl']].copy()
        site_era5 = era5[era5['water_body'] == site].copy()
        
        merged = pd.merge(site_chl, site_era5, on='fecha', how='inner')
        
        if len(merged) < 10:
            ax.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{SITE_CONFIG[site]['nombre']}", fontweight='bold')
            plt.close()
            continue
        
        # Calcular correlaciones
        correlations = []
        for var in climate_vars.keys():
            if var in merged.columns:
                corr = merged['chl_ugl'].corr(merged[var])
                correlations.append(corr)
            else:
                correlations.append(0)
        
        # Crear barras horizontales
        y_pos = np.arange(len(climate_vars))
        colors = ['green' if x > 0 else 'red' for x in correlations]
        
        bars = ax.barh(y_pos, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Línea en cero
        ax.axvline(x=0, color='black', linewidth=1.5)
        
        # Etiquetas
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(climate_vars.values()), fontsize=12)
        ax.set_xlabel('Correlación con Clorofila-a', fontweight='bold', fontsize=12)
        ax.set_title(f"Correlaciones Climáticas: {SITE_CONFIG[site]['nombre']}", 
                    fontweight='bold', fontsize=14, pad=10)
        ax.set_xlim([-1, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir valores
        for i, (bar, val) in enumerate(zip(bars, correlations)):
            if abs(val) > 0.01:
                ax.text(val, i, f' {val:.3f}', va='center',
                       ha='left' if val > 0 else 'right', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        filename = f"fig4a_correlaciones_{site}"
        plt.savefig(OUTPUT_DIR / f"{filename}.png", bbox_inches='tight', dpi=300)
        plt.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches='tight')
        plt.close()
        print(f"  ✓ Guardado: {filename}")
    
    print("✓ Figura 4 guardada: correlaciones climáticas (5 archivos individuales)")


def fig5_eventos_extremos_vs_clima(chl, era5):
    """Figura 5: Eventos extremos de clorofila y condiciones climáticas"""
    
    # Definir umbrales de eventos extremos (percentil 90)
    threshold_freshwater = 50  # μg/L
    threshold_marine = 15  # μg/L
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Grupos por tipo de ambiente
    freshwater_sites = ['cajon', 'yojoa', 'okeechobee']
    marine_sites = ['fonseca', 'tampa_bay']
    
    # Variables climáticas para analizar
    climate_vars = ['temp_air_2m', 'precipitation', 'solar_radiation', 'wind_speed_10m']
    var_labels = ['Temperatura (K)', 'Precipitación (mm)', 'Radiación Solar (J/m²)', 'Viento (m/s)']
    
    for idx, (var, label) in enumerate(zip(climate_vars, var_labels)):
        ax = axes[idx]
        
        # Datos para agua dulce
        fw_extreme = []
        fw_normal = []
        
        for site in freshwater_sites:
            site_chl = chl[chl['water_body'] == site][['fecha', 'chl_ugl']].copy()
            site_era5 = era5[era5['water_body'] == site][['fecha', var]].copy()
            
            merged = pd.merge(site_chl, site_era5, on='fecha', how='inner')
            
            extreme_mask = merged['chl_ugl'] > threshold_freshwater
            fw_extreme.extend(merged[extreme_mask][var].dropna().tolist())
            fw_normal.extend(merged[~extreme_mask][var].dropna().tolist())
        
        # Datos para marino
        m_extreme = []
        m_normal = []
        
        for site in marine_sites:
            site_chl = chl[chl['water_body'] == site][['fecha', 'chl_ugl']].copy()
            site_era5 = era5[era5['water_body'] == site][['fecha', var]].copy()
            
            merged = pd.merge(site_chl, site_era5, on='fecha', how='inner')
            
            extreme_mask = merged['chl_ugl'] > threshold_marine
            m_extreme.extend(merged[extreme_mask][var].dropna().tolist())
            m_normal.extend(merged[~extreme_mask][var].dropna().tolist())
        
        # Boxplots
        data_to_plot = [fw_normal, fw_extreme, m_normal, m_extreme]
        labels_plot = ['Agua Dulce\nNormal', 'Agua Dulce\nExtremo', 
                      'Marino\nNormal', 'Marino\nExtremo']
        colors = ['lightblue', 'red', 'lightgreen', 'orange']
        
        bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label.split("(")[0].strip()}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Test estadístico (t-test)
        from scipy import stats as sp_stats
        if len(fw_normal) > 0 and len(fw_extreme) > 0:
            t_stat, p_val = sp_stats.ttest_ind(fw_normal, fw_extreme)
            ax.text(0.02, 0.98, f'p-value (FW): {p_val:.4f}',
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Condiciones Climáticas: Eventos Normales vs Extremos de Clorofila',
                fontweight='bold', fontsize=15, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_eventos_extremos_clima.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig5_eventos_extremos_clima.pdf", bbox_inches='tight')
    print("✓ Figura 5 guardada: Eventos extremos vs clima")
    plt.close()


def main():
    """Función principal"""
    print("="*60)
    print("Generando figuras: Clima, Nutrientes y Clorofila")
    print("Análisis de factores causales")
    print("="*60)
    print()
    
    # Cargar datos
    print("Cargando datos...")
    chl, nutrients, era5 = load_data()
    print(f"✓ Clorofila: {len(chl)} observaciones")
    print(f"✓ Nutrientes: {len(nutrients)} observaciones")
    print(f"✓ Clima: {len(era5)} observaciones\n")
    
    # Generar figuras
    print("Generando figuras...")
    print()
    
    fig1_picos_anuales_integrado(chl, nutrients, era5)
    fig2_relacion_lluvia_clorofila(chl, era5)
    fig3_scatter_nutrientes_clorofila(chl, nutrients)
    fig4_heatmap_correlaciones(chl, era5)
    fig5_eventos_extremos_vs_clima(chl, era5)
    
    print()
    print("="*60)
    print("✓ Todas las figuras generadas exitosamente")
    print(f"✓ Ubicación: {OUTPUT_DIR}")
    print("="*60)
    print()
    print("Figuras generadas:")
    print("  1. fig1_picos_anuales_integrado - Clorofila, nutrientes y precipitación por año")
    print("  2. fig2_lluvia_clorofila - Relación época de lluvias y clorofila")
    print("  3. fig3_nutrientes_clorofila - Scatter plots nutrientes vs clorofila")
    print("  4. fig4_correlaciones_clima - Correlaciones variables climáticas")
    print("  5. fig5_eventos_extremos_clima - Condiciones climáticas en eventos extremos")
    print()
    print("Formatos: PNG (alta resolución) y PDF (vectorial)")

if __name__ == "__main__":
    main()
