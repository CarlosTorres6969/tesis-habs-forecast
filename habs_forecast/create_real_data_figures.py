"""
Script para generar gráficos de DATOS REALES de clorofila y biomasa algal
Comparación entre sitios de Honduras y Florida
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
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Directorios
ARTIFACTS_DIR = Path("artifacts")
TARGETS_DIR = ARTIFACTS_DIR / "targets"
OUTPUT_DIR = Path("entregables") / "datos_reales_tesis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración por sitio
SITE_CONFIG = {
    'cajon': {'pais': 'Honduras', 'tipo': 'Agua Dulce', 'color': '#E63946', 'nombre': 'El Cajón'},
    'yojoa': {'pais': 'Honduras', 'tipo': 'Agua Dulce', 'color': '#F4A261', 'nombre': 'Lago Yojoa'},
    'fonseca': {'pais': 'Honduras', 'tipo': 'Marino', 'color': '#2A9D8F', 'nombre': 'Golfo de Fonseca'},
    'okeechobee': {'pais': 'Florida', 'tipo': 'Agua Dulce', 'color': '#264653', 'nombre': 'Lake Okeechobee'},
    'tampa_bay': {'pais': 'Florida', 'tipo': 'Marino', 'color': '#457B9D', 'nombre': 'Tampa Bay'}
}

# Umbrales de alerta para HABs (μg/L clorofila-a)
THRESHOLD_FRESHWATER = 20.0  # WHO guideline
THRESHOLD_MARINE = 10.0      # Típico para ambientes marinos

def load_data():
    """Carga los datos de clorofila observada"""
    df = pd.read_csv(TARGETS_DIR / "combined_target.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['year'] = df['fecha'].dt.year
    df['month'] = df['fecha'].dt.month
    
    # Añadir información de configuración
    df['pais'] = df['water_body'].map(lambda x: SITE_CONFIG[x]['pais'])
    df['tipo'] = df['water_body'].map(lambda x: SITE_CONFIG[x]['tipo'])
    df['nombre_sitio'] = df['water_body'].map(lambda x: SITE_CONFIG[x]['nombre'])
    
    return df


def fig1_concentracion_por_sitio(df):
    """Figura 1: Comparación de distribución de clorofila por sitio"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Ordenar por país y tipo
    order = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    df_ordered = df.copy()
    df_ordered['water_body'] = pd.Categorical(df_ordered['water_body'], categories=order, ordered=True)
    df_ordered = df_ordered.sort_values('water_body')
    
    # Crear boxplot con violinplot
    parts = ax.violinplot(
        [df[df['water_body'] == site]['chl_ugl'].values for site in order],
        positions=range(len(order)),
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )
    
    # Colorear violines
    for i, (site, pc) in enumerate(zip(order, parts['bodies'])):
        pc.set_facecolor(SITE_CONFIG[site]['color'])
        pc.set_alpha(0.7)
    
    # Añadir boxplot encima
    bp = ax.boxplot(
        [df[df['water_body'] == site]['chl_ugl'].values for site in order],
        positions=range(len(order)),
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor='white', alpha=0.5),
        medianprops=dict(color='red', linewidth=2),
        showfliers=False
    )
    
    # Líneas de umbral
    ax.axhline(y=THRESHOLD_FRESHWATER, color='red', linestyle='--', linewidth=2, 
               alpha=0.6, label=f'Umbral Agua Dulce ({THRESHOLD_FRESHWATER} μg/L)')
    ax.axhline(y=THRESHOLD_MARINE, color='orange', linestyle='--', linewidth=2,
               alpha=0.6, label=f'Umbral Marino ({THRESHOLD_MARINE} μg/L)')
    
    # Etiquetas
    labels = [f"{SITE_CONFIG[site]['nombre']}\n{SITE_CONFIG[site]['pais']}\n(n={len(df[df['water_body']==site])})" 
              for site in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=11)
    
    ax.set_ylabel('Clorofila-a (μg/L)', fontweight='bold', fontsize=14)
    ax.set_title('Distribución de Concentraciones de Clorofila-a por Cuerpo de Agua\nHonduras vs Florida',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 120)
    
    # Separador visual Honduras vs Florida
    ax.axvline(x=2.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(1, 110, 'HONDURAS', ha='center', fontsize=13, fontweight='bold', color='gray')
    ax.text(3.5, 110, 'FLORIDA', ha='center', fontsize=13, fontweight='bold', color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_concentracion_por_sitio.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig1_concentracion_por_sitio.pdf", bbox_inches='tight')
    print("✓ Figura 1 guardada: Distribución de clorofila por sitio")
    plt.close()


def fig2_series_temporales(df):
    """Figura 2: Series temporales de clorofila por sitio - VERSIÓN INDIVIDUAL"""
    
    order = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    
    for site in order:
        # Crear figura individual para cada sitio
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        site_data = df[df['water_body'] == site].sort_values('fecha')
        
        # Línea principal (SIN PUNTOS - removido marker)
        ax.plot(site_data['fecha'], site_data['chl_ugl'], 
                color=SITE_CONFIG[site]['color'], linewidth=2, alpha=0.8)
        
        # Rellenar área
        ax.fill_between(site_data['fecha'], 0, site_data['chl_ugl'],
                        color=SITE_CONFIG[site]['color'], alpha=0.2)
        
        # Umbral
        threshold = THRESHOLD_FRESHWATER if SITE_CONFIG[site]['tipo'] == 'Agua Dulce' else THRESHOLD_MARINE
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.6,
                  label=f'Umbral HAB ({threshold} μg/L)')
        
        # Resaltar eventos sobre umbral (SIN SCATTER POINTS - removido)
        eventos = site_data[site_data['chl_ugl'] > threshold]
        
        # Títulos y etiquetas
        ax.set_title(f"Serie Temporal de Clorofila-a: {SITE_CONFIG[site]['nombre']} ({SITE_CONFIG[site]['pais']})\n"
                    f"{SITE_CONFIG[site]['tipo']} - n={len(site_data)}",
                    fontweight='bold', fontsize=14, color=SITE_CONFIG[site]['color'], pad=15)
        ax.set_xlabel('Fecha', fontweight='bold', fontsize=12)
        ax.set_ylabel('Clorofila-a (μg/L)', fontsize=12, fontweight='bold')
        
        # Configurar eje Y con más detalle y valores exactos
        max_val = site_data['chl_ugl'].max()
        
        # Determinar rango apropiado y ticks según el máximo
        if max_val <= 30:
            # Rango pequeño: ticks cada 2 unidades
            y_max = int(np.ceil(max_val / 5) * 5) + 5
            yticks = np.arange(0, y_max + 1, 2)
        elif max_val <= 60:
            # Rango medio: ticks cada 5 unidades
            y_max = int(np.ceil(max_val / 10) * 10) + 10
            yticks = np.arange(0, y_max + 1, 5)
        elif max_val <= 120:
            # Rango grande: ticks cada 10 unidades
            y_max = int(np.ceil(max_val / 10) * 10) + 10
            yticks = np.arange(0, y_max + 1, 10)
        else:
            # Rango muy grande: ticks cada 20 unidades
            y_max = int(np.ceil(max_val / 20) * 20) + 20
            yticks = np.arange(0, y_max + 1, 20)
        
        ax.set_ylim(0, y_max)
        ax.set_yticks(yticks)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(yticks[1] - yticks[0]) if len(yticks) > 1 else plt.AutoMinorLocator())
        
        ax.grid(True, alpha=0.3, which='major', axis='both')
        ax.grid(True, alpha=0.15, which='minor', axis='y', linestyle=':')
        ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
        
        # Formato de fechas
        ax.tick_params(axis='x', rotation=45)
        
        # Estadísticas en texto
        mean_val = site_data['chl_ugl'].mean()
        eventos_count = len(eventos)
        perc_eventos = (eventos_count / len(site_data)) * 100
        
        stats_text = f"Media: {mean_val:.1f} μg/L\nMáx: {max_val:.1f} μg/L\nEventos HAB: {eventos_count} ({perc_eventos:.1f}%)"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        filename = f"fig2a_serie_temporal_{site}"
        plt.savefig(OUTPUT_DIR / f"{filename}.png", bbox_inches='tight', dpi=300)
        plt.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches='tight')
        plt.close()
        print(f"  ✓ Guardado: {filename}")
    
    print("✓ Figura 2 guardada: Series temporales (5 archivos individuales)")


def fig3_comparacion_paises(df):
    """Figura 3: Comparación agregada Honduras vs Florida"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Agrupar por país
    honduras = df[df['pais'] == 'Honduras']
    florida = df[df['pais'] == 'Florida']
    
    # 1. Distribución general
    ax1.violinplot([honduras['chl_ugl'], florida['chl_ugl']], 
                    positions=[0, 1], widths=0.7, showmeans=True, showmedians=True)
    ax1.boxplot([honduras['chl_ugl'], florida['chl_ugl']], 
                positions=[0, 1], widths=0.3, patch_artist=True,
                boxprops=dict(facecolor='white', alpha=0.5),
                medianprops=dict(color='red', linewidth=2))
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Honduras\n(n=' + str(len(honduras)) + ')', 
                         'Florida\n(n=' + str(len(florida)) + ')'], fontsize=12)
    ax1.set_ylabel('Clorofila-a (μg/L)', fontweight='bold')
    ax1.set_title('Distribución General por País', fontweight='bold', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Por tipo de ambiente
    agua_dulce = df[df['tipo'] == 'Agua Dulce']
    marino = df[df['tipo'] == 'Marino']
    
    data_by_type = [
        agua_dulce[agua_dulce['pais'] == 'Honduras']['chl_ugl'],
        agua_dulce[agua_dulce['pais'] == 'Florida']['chl_ugl'],
        marino[marino['pais'] == 'Honduras']['chl_ugl'],
        marino[marino['pais'] == 'Florida']['chl_ugl']
    ]
    
    bp = ax2.boxplot(data_by_type, positions=[0, 1, 3, 4], widths=0.6,
                     patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    
    # Colorear por país
    colors = ['#E63946', '#264653', '#2A9D8F', '#457B9D']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xticks([0, 1, 3, 4])
    ax2.set_xticklabels(['Hn\nDulce', 'FL\nDulce', 'Hn\nMarino', 'FL\nMarino'], fontsize=11)
    ax2.set_ylabel('Clorofila-a (μg/L)', fontweight='bold')
    ax2.set_title('Por Tipo de Ambiente', fontweight='bold', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(x=2, color='gray', linestyle=':', alpha=0.5)
    
    # 3. Frecuencia de eventos HAB
    def calc_hab_percentage(data, threshold):
        return (data['chl_ugl'] > threshold).sum() / len(data) * 100
    
    sites_order = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    hab_data = {}
    colors_bar = []
    
    for site in sites_order:
        threshold = THRESHOLD_FRESHWATER if SITE_CONFIG[site]['tipo'] == 'Agua Dulce' else THRESHOLD_MARINE
        hab_data[SITE_CONFIG[site]['nombre']] = calc_hab_percentage(df[df['water_body'] == site], threshold)
        colors_bar.append(SITE_CONFIG[site]['color'])
    bars = ax3.bar(range(len(hab_data)), list(hab_data.values()), color=colors_bar, alpha=0.8)
    ax3.set_xticks(range(len(hab_data)))
    ax3.set_xticklabels(list(hab_data.keys()), rotation=45, ha='right')
    ax3.set_ylabel('% Observaciones > Umbral', fontweight='bold')
    ax3.set_title('Frecuencia de Eventos HAB por Sitio', fontweight='bold', fontsize=13)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(hab_data.values()) * 1.15)
    
    # Añadir valores sobre barras
    for bar, val in zip(bars, hab_data.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Estadísticas comparativas
    ax4.axis('off')
    
    stats_text = "ESTADÍSTICAS COMPARATIVAS\n\n"
    stats_text += "=" * 40 + "\n\n"
    
    stats_text += "HONDURAS:\n"
    stats_text += f"  • Total observaciones: {len(honduras)}\n"
    stats_text += f"  • Media: {honduras['chl_ugl'].mean():.2f} μg/L\n"
    stats_text += f"  • Mediana: {honduras['chl_ugl'].median():.2f} μg/L\n"
    stats_text += f"  • Máximo: {honduras['chl_ugl'].max():.2f} μg/L\n"
    stats_text += f"  • Desv. Est.: {honduras['chl_ugl'].std():.2f} μg/L\n\n"
    
    stats_text += "FLORIDA:\n"
    stats_text += f"  • Total observaciones: {len(florida)}\n"
    stats_text += f"  • Media: {florida['chl_ugl'].mean():.2f} μg/L\n"
    stats_text += f"  • Mediana: {florida['chl_ugl'].median():.2f} μg/L\n"
    stats_text += f"  • Máximo: {florida['chl_ugl'].max():.2f} μg/L\n"
    stats_text += f"  • Desv. Est.: {florida['chl_ugl'].std():.2f} μg/L\n\n"
    
    stats_text += "=" * 40 + "\n\n"
    stats_text += "POR TIPO DE AMBIENTE:\n\n"
    stats_text += f"Agua Dulce (media): {agua_dulce['chl_ugl'].mean():.2f} μg/L\n"
    stats_text += f"Marino (media): {marino['chl_ugl'].mean():.2f} μg/L\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Comparación Honduras vs Florida: Niveles de Clorofila-a',
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_comparacion_paises.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig3_comparacion_paises.pdf", bbox_inches='tight')
    print("✓ Figura 3 guardada: Comparación por países")
    plt.close()


def fig4_estacionalidad(df):
    """Figura 4: Patrones estacionales por sitio"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    order = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    for idx, site in enumerate(order):
        ax = axes[idx]
        site_data = df[df['water_body'] == site].copy()
        
        # Agrupar por mes
        monthly = site_data.groupby('month')['chl_ugl'].agg(['mean', 'std', 'count']).reset_index()
        
        # Gráfico de barras con error bars
        ax.bar(monthly['month'], monthly['mean'], 
               color=SITE_CONFIG[site]['color'], alpha=0.7,
               yerr=monthly['std'], capsize=5, error_kw={'linewidth': 1.5})
        
        # Línea de tendencia
        ax.plot(monthly['month'], monthly['mean'], 
                color=SITE_CONFIG[site]['color'], linewidth=2, marker='o', markersize=8)
        
        # Umbral
        threshold = THRESHOLD_FRESHWATER if SITE_CONFIG[site]['tipo'] == 'Agua Dulce' else THRESHOLD_MARINE
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Mes', fontweight='bold')
        ax.set_ylabel('Clorofila-a (μg/L)', fontweight='bold')
        ax.set_title(f"{SITE_CONFIG[site]['nombre']}\n{SITE_CONFIG[site]['pais']}",
                    fontweight='bold', fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Número de observaciones por mes (anotación)
        for _, row in monthly.iterrows():
            if row['count'] > 0:
                ax.text(row['month'], -3, f"n={int(row['count'])}", 
                       ha='center', fontsize=8, color='gray')
    
    # Ocultar el último subplot
    axes[-1].axis('off')
    
    fig.suptitle('Variabilidad Estacional de Clorofila-a por Sitio',
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_estacionalidad.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig4_estacionalidad.pdf", bbox_inches='tight')
    print("✓ Figura 4 guardada: Estacionalidad")
    plt.close()


def fig5_heatmap_intensidad(df):
    """Figura 5: Mapa de calor de intensidad de florecimientos"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Crear matriz de año x sitio con promedio de clorofila
    pivot = df.pivot_table(values='chl_ugl', index='year', 
                           columns='water_body', aggfunc='mean')
    
    # Ordenar columnas
    order = ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']
    pivot = pivot[order]
    
    # Renombrar columnas para el gráfico
    pivot.columns = [SITE_CONFIG[col]['nombre'] for col in pivot.columns]
    
    # Crear heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Clorofila-a (μg/L)'},
                linewidths=0.5, linecolor='gray', ax=ax,
                vmin=0, vmax=40, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_xlabel('Cuerpo de Agua', fontweight='bold', fontsize=13)
    ax.set_ylabel('Año', fontweight='bold', fontsize=13)
    ax.set_title('Intensidad Promedio de Florecimientos Algales por Año y Sitio',
                fontweight='bold', fontsize=15, pad=15)
    
    # Separador Honduras vs Florida
    ax.axvline(x=3, color='blue', linewidth=3, alpha=0.7)
    ax.text(1.5, -0.8, 'HONDURAS', ha='center', fontsize=12, 
            fontweight='bold', color='gray', transform=ax.get_xaxis_transform())
    ax.text(4, -0.8, 'FLORIDA', ha='center', fontsize=12,
            fontweight='bold', color='gray', transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_heatmap_intensidad.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig5_heatmap_intensidad.pdf", bbox_inches='tight')
    print("✓ Figura 5 guardada: Heatmap de intensidad")
    plt.close()


def crear_tabla_resumen(df):
    """Crear tabla resumen en formato markdown"""
    
    tabla_md = "# Tabla Resumen: Estadísticas de Clorofila-a por Sitio\n\n"
    tabla_md += "## Datos Observados de Biomasa Algal\n\n"
    tabla_md += "| Sitio | País | Tipo | N | Media (μg/L) | Mediana (μg/L) | Máx (μg/L) | Desv. Est. | % > Umbral HAB |\n"
    tabla_md += "|-------|------|------|---|--------------|----------------|------------|------------|----------------|\n"
    
    for site in ['cajon', 'yojoa', 'fonseca', 'okeechobee', 'tampa_bay']:
        site_data = df[df['water_body'] == site]
        threshold = THRESHOLD_FRESHWATER if SITE_CONFIG[site]['tipo'] == 'Agua Dulce' else THRESHOLD_MARINE
        perc_hab = (site_data['chl_ugl'] > threshold).sum() / len(site_data) * 100
        
        tabla_md += f"| {SITE_CONFIG[site]['nombre']} | "
        tabla_md += f"{SITE_CONFIG[site]['pais']} | "
        tabla_md += f"{SITE_CONFIG[site]['tipo']} | "
        tabla_md += f"{len(site_data)} | "
        tabla_md += f"{site_data['chl_ugl'].mean():.2f} | "
        tabla_md += f"{site_data['chl_ugl'].median():.2f} | "
        tabla_md += f"{site_data['chl_ugl'].max():.2f} | "
        tabla_md += f"{site_data['chl_ugl'].std():.2f} | "
        tabla_md += f"{perc_hab:.1f}% |\n"
    
    tabla_md += "\n\n## Resumen por País\n\n"
    tabla_md += "| País | N Total | Media (μg/L) | Mediana (μg/L) | Máx (μg/L) |\n"
    tabla_md += "|------|---------|--------------|----------------|------------|\n"
    
    for pais in ['Honduras', 'Florida']:
        pais_data = df[df['pais'] == pais]
        tabla_md += f"| {pais} | "
        tabla_md += f"{len(pais_data)} | "
        tabla_md += f"{pais_data['chl_ugl'].mean():.2f} | "
        tabla_md += f"{pais_data['chl_ugl'].median():.2f} | "
        tabla_md += f"{pais_data['chl_ugl'].max():.2f} |\n"
    
    tabla_md += "\n\n## Resumen por Tipo de Ambiente\n\n"
    tabla_md += "| Tipo | N Total | Media (μg/L) | Mediana (μg/L) | Máx (μg/L) |\n"
    tabla_md += "|------|---------|--------------|----------------|------------|\n"
    
    for tipo in ['Agua Dulce', 'Marino']:
        tipo_data = df[df['tipo'] == tipo]
        tabla_md += f"| {tipo} | "
        tabla_md += f"{len(tipo_data)} | "
        tabla_md += f"{tipo_data['chl_ugl'].mean():.2f} | "
        tabla_md += f"{tipo_data['chl_ugl'].median():.2f} | "
        tabla_md += f"{tipo_data['chl_ugl'].max():.2f} |\n"
    
    tabla_md += "\n\n## Notas\n\n"
    tabla_md += f"- **Umbral HAB Agua Dulce:** {THRESHOLD_FRESHWATER} μg/L (WHO guideline)\n"
    tabla_md += f"- **Umbral HAB Marino:** {THRESHOLD_MARINE} μg/L (típico para ambientes costeros)\n"
    tabla_md += f"- **Periodo de datos:** {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}\n"
    tabla_md += f"- **Total de observaciones:** {len(df)}\n"
    
    with open(OUTPUT_DIR / "TABLA_COMPARATIVA.md", 'w', encoding='utf-8') as f:
        f.write(tabla_md)
    
    print("✓ Tabla resumen guardada: TABLA_COMPARATIVA.md")


def main():
    """Función principal"""
    print("="*60)
    print("Generando figuras de DATOS REALES de clorofila")
    print("Comparación: Honduras vs Florida")
    print("="*60)
    print()
    
    # Cargar datos
    print("Cargando datos...")
    df = load_data()
    print(f"✓ Cargados {len(df)} observaciones de {df['water_body'].nunique()} sitios\n")
    
    # Generar figuras
    print("Generando figuras...")
    print()
    
    fig1_concentracion_por_sitio(df)
    fig2_series_temporales(df)
    fig3_comparacion_paises(df)
    fig4_estacionalidad(df)
    fig5_heatmap_intensidad(df)
    crear_tabla_resumen(df)
    
    print()
    print("="*60)
    print("✓ Todas las figuras generadas exitosamente")
    print(f"✓ Ubicación: {OUTPUT_DIR}")
    print("="*60)
    print()
    print("Figuras generadas:")
    print("  1. fig1_concentracion_por_sitio - Distribución por cuerpo de agua")
    print("  2. fig2_series_temporales - Series temporales de cada sitio")
    print("  3. fig3_comparacion_paises - Comparación Honduras vs Florida")
    print("  4. fig4_estacionalidad - Patrones estacionales")
    print("  5. fig5_heatmap_intensidad - Intensidad por año y sitio")
    print("  6. TABLA_COMPARATIVA.md - Estadísticas detalladas")
    print()
    print("Formatos: PNG (alta resolución) y PDF (vectorial)")

if __name__ == "__main__":
    main()
