"""
Script para generar gráficos comparativos de freshwater vs marine para tesis
Autor: Sistema HABs Forecast
Fecha: 2026-07-07
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
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
REPORTS_DIR = ARTIFACTS_DIR / "reports"
OUTPUT_DIR = Path("entregables") / "tesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colores consistentes
COLORS = {
    'freshwater': '#2E86AB',  # Azul
    'marine': '#06A77D'       # Verde azulado
}

def load_data():
    """Carga todos los datos necesarios"""
    with open(REPORTS_DIR / "nested_metrics.json") as f:
        nested_metrics = json.load(f)
    
    with open(REPORTS_DIR / "interval_metrics.json") as f:
        interval_metrics = json.load(f)
    
    feature_importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    shap_importance = pd.read_csv(REPORTS_DIR / "shap_importance.csv")
    
    forecast_verification = pd.read_csv(REPORTS_DIR / "forecast_verification_summary.csv")
    
    return nested_metrics, interval_metrics, feature_importance, shap_importance, forecast_verification


def fig1_skill_comparison(nested_metrics):
    """Figura 1: Comparación de Skill Score por horizonte y ambiente"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [1, 3, 5, 7]
    x = np.arange(len(horizons))
    width = 0.35
    
    # Extraer skill scores (media de nested CV)
    freshwater_skill = []
    marine_skill = []
    freshwater_err = []
    marine_err = []
    
    for h in horizons:
        fw_scores = nested_metrics['freshwater'][str(h)]['skill_nested']
        m_scores = nested_metrics['marine'][str(h)]['skill_nested']
        
        freshwater_skill.append(np.mean(fw_scores))
        marine_skill.append(np.mean(m_scores))
        freshwater_err.append(np.std(fw_scores))
        marine_err.append(np.std(m_scores))
    
    # Crear barras
    bars1 = ax.bar(x - width/2, freshwater_skill, width, 
                   label='Agua Dulce', color=COLORS['freshwater'],
                   yerr=freshwater_err, capsize=5, alpha=0.9)
    bars2 = ax.bar(x + width/2, marine_skill, width,
                   label='Marino', color=COLORS['marine'],
                   yerr=marine_err, capsize=5, alpha=0.9)
    
    ax.set_xlabel('Horizonte de Predicción (días)', fontweight='bold')
    ax.set_ylabel('Skill Score', fontweight='bold')
    ax.set_title('Comparación de Skill Score: Ambientes de Agua Dulce vs Marinos', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(freshwater_skill + marine_skill) * 1.2])
    
    # Añadir valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_skill_comparison.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig1_skill_comparison.pdf", bbox_inches='tight')
    print("✓ Figura 1 guardada: Comparación de Skill Score")
    plt.close()


def fig2_prauc_comparison(nested_metrics):
    """Figura 2: Comparación de PR-AUC (detección de alertas) por horizonte y ambiente"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [1, 3, 5, 7]
    x = np.arange(len(horizons))
    width = 0.35
    
    freshwater_prauc = []
    marine_prauc = []
    freshwater_err = []
    marine_err = []
    
    for h in horizons:
        fw_scores = nested_metrics['freshwater'][str(h)]['pr_auc_nested']
        m_scores = nested_metrics['marine'][str(h)]['pr_auc_nested']
        
        freshwater_prauc.append(np.mean(fw_scores))
        marine_prauc.append(np.mean(m_scores))
        freshwater_err.append(np.std(fw_scores))
        marine_err.append(np.std(m_scores))
    
    bars1 = ax.bar(x - width/2, freshwater_prauc, width,
                   label='Agua Dulce', color=COLORS['freshwater'],
                   yerr=freshwater_err, capsize=5, alpha=0.9)
    bars2 = ax.bar(x + width/2, marine_prauc, width,
                   label='Marino', color=COLORS['marine'],
                   yerr=marine_err, capsize=5, alpha=0.9)
    
    ax.set_xlabel('Horizonte de Predicción (días)', fontweight='bold')
    ax.set_ylabel('PR-AUC (Precision-Recall)', fontweight='bold')
    ax.set_title('Capacidad de Detección de Alertas: Agua Dulce vs Marino',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Añadir línea de referencia para clasificador aleatorio
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Clasificador aleatorio')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_prauc_comparison.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig2_prauc_comparison.pdf", bbox_inches='tight')
    print("✓ Figura 2 guardada: Comparación de PR-AUC")
    plt.close()


def fig3_interval_coverage(interval_metrics):
    """Figura 3: Cobertura de intervalos de predicción"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [1, 3, 5, 7]
    x = np.arange(len(horizons))
    width = 0.35
    
    freshwater_cov = []
    marine_cov = []
    freshwater_err = []
    marine_err = []
    
    for h in horizons:
        fw_cov = interval_metrics['freshwater'][str(h)]['cobertura_cqr']
        m_cov = interval_metrics['marine'][str(h)]['cobertura_cqr']
        
        freshwater_cov.append(np.mean(fw_cov))
        marine_cov.append(np.mean(m_cov))
        freshwater_err.append(np.std(fw_cov))
        marine_err.append(np.std(m_cov))
    
    bars1 = ax.bar(x - width/2, freshwater_cov, width,
                   label='Agua Dulce', color=COLORS['freshwater'],
                   yerr=freshwater_err, capsize=5, alpha=0.9)
    bars2 = ax.bar(x + width/2, marine_cov, width,
                   label='Marino', color=COLORS['marine'],
                   yerr=marine_err, capsize=5, alpha=0.9)
    
    # Línea objetivo (80% de cobertura nominal)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label='Cobertura Nominal (80%)')
    
    ax.set_xlabel('Horizonte de Predicción (días)', fontweight='bold')
    ax.set_ylabel('Cobertura de Intervalos', fontweight='bold')
    ax.set_title('Cobertura de Intervalos de Predicción (CQR): Comparación por Ambiente',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc='lower left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_interval_coverage.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig3_interval_coverage.pdf", bbox_inches='tight')
    print("✓ Figura 3 guardada: Cobertura de intervalos")
    plt.close()


def fig4_feature_count_comparison(feature_importance):
    """Figura 4: Número de características seleccionadas por ambiente"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [1, 3, 5, 7]
    x = np.arange(len(horizons))
    width = 0.35
    
    freshwater_counts = []
    marine_counts = []
    
    for h in horizons:
        fw_count = len(feature_importance[(feature_importance['group'] == 'freshwater') & 
                                          (feature_importance['horizon'] == h)])
        m_count = len(feature_importance[(feature_importance['group'] == 'marine') & 
                                         (feature_importance['horizon'] == h)])
        freshwater_counts.append(fw_count)
        marine_counts.append(m_count)
    
    bars1 = ax.bar(x - width/2, freshwater_counts, width,
                   label='Agua Dulce', color=COLORS['freshwater'], alpha=0.9)
    bars2 = ax.bar(x + width/2, marine_counts, width,
                   label='Marino', color=COLORS['marine'], alpha=0.9)
    
    ax.set_xlabel('Horizonte de Predicción (días)', fontweight='bold')
    ax.set_ylabel('Número de Características', fontweight='bold')
    ax.set_title('Complejidad del Modelo: Número de Características por Ambiente',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_feature_count.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig4_feature_count.pdf", bbox_inches='tight')
    print("✓ Figura 4 guardada: Número de características")
    plt.close()


def fig5_top_features_comparison(shap_importance):
    """Figura 5: Top 10 características más importantes por ambiente (SHAP values)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Agrupar por ambiente y sumar importancia SHAP a través de horizontes
    fw_agg = shap_importance[shap_importance['group'] == 'freshwater'].groupby('feature')['mean_abs_shap'].sum().sort_values(ascending=False)
    m_agg = shap_importance[shap_importance['group'] == 'marine'].groupby('feature')['mean_abs_shap'].sum().sort_values(ascending=False)
    
    # Top 10
    fw_top10 = fw_agg.head(10)
    m_top10 = m_agg.head(10)
    
    # Agua Dulce
    ax1.barh(range(len(fw_top10)), fw_top10.values, color=COLORS['freshwater'], alpha=0.9)
    ax1.set_yticks(range(len(fw_top10)))
    ax1.set_yticklabels(fw_top10.index)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importancia SHAP (Acumulada)', fontweight='bold')
    ax1.set_title('Agua Dulce\nTop 10 Características', fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Añadir valores
    for i, v in enumerate(fw_top10.values):
        ax1.text(v, i, f' {v:.3f}', va='center', fontsize=9)
    
    # Marino
    ax2.barh(range(len(m_top10)), m_top10.values, color=COLORS['marine'], alpha=0.9)
    ax2.set_yticks(range(len(m_top10)))
    ax2.set_yticklabels(m_top10.index)
    ax2.invert_yaxis()
    ax2.set_xlabel('Importancia SHAP (Acumulada)', fontweight='bold')
    ax2.set_title('Marino\nTop 10 Características', fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(m_top10.values):
        ax2.text(v, i, f' {v:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Características Más Importantes por Ambiente (Valores SHAP)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_top_features_shap.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig5_top_features_shap.pdf", bbox_inches='tight')
    print("✓ Figura 5 guardada: Top características SHAP")
    plt.close()


def fig6_forecast_verification(forecast_verification):
    """Figura 6: Métricas de verificación de pronósticos operacionales"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    horizons = [1, 3, 5, 7]
    x = np.arange(len(horizons))
    width = 0.35
    
    # Extraer métricas
    fw_data = forecast_verification[forecast_verification['group'] == 'freshwater']
    m_data = forecast_verification[forecast_verification['group'] == 'marine']
    
    # MAE
    fw_mae = [fw_data[fw_data['horizon'] == h]['MAE'].values[0] for h in horizons]
    m_mae = [m_data[m_data['horizon'] == h]['MAE'].values[0] for h in horizons]
    
    ax1.bar(x - width/2, fw_mae, width, label='Agua Dulce', color=COLORS['freshwater'], alpha=0.9)
    ax1.bar(x + width/2, m_mae, width, label='Marino', color=COLORS['marine'], alpha=0.9)
    ax1.set_ylabel('MAE (μg/L)', fontweight='bold')
    ax1.set_title('Error Absoluto Medio', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # POD (Probability of Detection)
    fw_pod = [fw_data[fw_data['horizon'] == h]['POD'].values[0] for h in horizons]
    m_pod = [m_data[m_data['horizon'] == h]['POD'].values[0] for h in horizons]
    
    ax2.bar(x - width/2, fw_pod, width, label='Agua Dulce', color=COLORS['freshwater'], alpha=0.9)
    ax2.bar(x + width/2, m_pod, width, label='Marino', color=COLORS['marine'], alpha=0.9)
    ax2.set_ylabel('POD', fontweight='bold')
    ax2.set_title('Probabilidad de Detección', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    # FAR (False Alarm Rate)
    fw_far = [fw_data[fw_data['horizon'] == h]['FAR'].values[0] for h in horizons]
    m_far = [m_data[m_data['horizon'] == h]['FAR'].values[0] for h in horizons]
    
    ax3.bar(x - width/2, fw_far, width, label='Agua Dulce', color=COLORS['freshwater'], alpha=0.9)
    ax3.bar(x + width/2, m_far, width, label='Marino', color=COLORS['marine'], alpha=0.9)
    ax3.set_xlabel('Horizonte de Predicción (días)', fontweight='bold')
    ax3.set_ylabel('FAR', fontweight='bold')
    ax3.set_title('Tasa de Falsas Alarmas', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons)
    ax3.set_ylim([0, 0.6])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
    
    # F1 Score
    fw_f1 = [fw_data[fw_data['horizon'] == h]['F1'].values[0] for h in horizons]
    m_f1 = [m_data[m_data['horizon'] == h]['F1'].values[0] for h in horizons]
    
    ax4.bar(x - width/2, fw_f1, width, label='Agua Dulce', color=COLORS['freshwater'], alpha=0.9)
    ax4.bar(x + width/2, m_f1, width, label='Marino', color=COLORS['marine'], alpha=0.9)
    ax4.set_xlabel('Horizonte de Predicción (días)', fontweight='bold')
    ax4.set_ylabel('F1 Score', fontweight='bold')
    ax4.set_title('F1 Score (Balance POD-FAR)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(horizons)
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    plt.suptitle('Verificación de Pronósticos Operacionales: Agua Dulce vs Marino',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_forecast_verification.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig6_forecast_verification.pdf", bbox_inches='tight')
    print("✓ Figura 6 guardada: Verificación de pronósticos")
    plt.close()


def fig7_feature_category_comparison(feature_importance):
    """Figura 7: Distribución de tipos de características por ambiente"""
    
    def categorize_feature(feat):
        """Categoriza características por tipo"""
        if feat.startswith('chl_') or feat == 'log_chl_t0':
            return 'Autorregresivas'
        elif feat.startswith('B') or feat in ['NDCI', 'CI_red', 'FAI', 'turbidity']:
            return 'Espectrales'
        elif feat in ['temp_air_2m', 'solar_radiation', 'precipitation', 
                      'wind_speed_10m', 'surface_pressure'] or '_roll7' in feat:
            return 'Climáticas (ERA5)'
        elif feat in ['tp_context', 'water_temp', 'do_mgl', 'ph', 
                      'turbidity_insitu', 'spec_cond', 'secchi', 'ammonia']:
            return 'In-situ'
        else:
            return 'Otras'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Procesar ambientes por separado
    for ax, group, color in [(ax1, 'freshwater', COLORS['freshwater']), 
                             (ax2, 'marine', COLORS['marine'])]:
        
        # Obtener características únicas para este ambiente
        features = feature_importance[feature_importance['group'] == group]['feature'].unique()
        categories = [categorize_feature(f) for f in features]
        
        # Contar por categoría
        from collections import Counter
        cat_counts = Counter(categories)
        
        # Crear gráfico de torta
        labels = list(cat_counts.keys())
        sizes = list(cat_counts.values())
        colors_pie = plt.cm.Set3(range(len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90,
                                           textprops={'fontsize': 10})
        
        # Hacer texto más legible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        title = 'Agua Dulce' if group == 'freshwater' else 'Marino'
        ax.set_title(f'{title}\nDistribución de Tipos de Características',
                    fontweight='bold', pad=15)
    
    plt.suptitle('Comparación de Tipos de Características por Ambiente',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_feature_categories.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig7_feature_categories.pdf", bbox_inches='tight')
    print("✓ Figura 7 guardada: Categorías de características")
    plt.close()


def fig8_performance_summary_radar(nested_metrics, forecast_verification):
    """Figura 8: Gráfico radar comparativo de métricas consolidadas"""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Métricas a comparar (normalizadas 0-1)
    categories = ['Skill Score', 'PR-AUC', 'POD', 'Precisión', 'F1 Score']
    N = len(categories)
    
    # Calcular promedios a través de horizontes
    fw_skill = np.mean([np.mean(nested_metrics['freshwater'][str(h)]['skill_nested']) 
                        for h in [1, 3, 5, 7]])
    m_skill = np.mean([np.mean(nested_metrics['marine'][str(h)]['skill_nested']) 
                       for h in [1, 3, 5, 7]])
    
    fw_prauc = np.mean([np.mean(nested_metrics['freshwater'][str(h)]['pr_auc_nested']) 
                        for h in [1, 3, 5, 7]])
    m_prauc = np.mean([np.mean(nested_metrics['marine'][str(h)]['pr_auc_nested']) 
                       for h in [1, 3, 5, 7]])
    
    fw_ver = forecast_verification[forecast_verification['group'] == 'freshwater']
    m_ver = forecast_verification[forecast_verification['group'] == 'marine']
    
    fw_values = [
        fw_skill,
        fw_prauc,
        fw_ver['POD'].mean(),
        fw_ver['precision'].mean(),
        fw_ver['F1'].mean()
    ]
    
    m_values = [
        m_skill,
        m_prauc,
        m_ver['POD'].mean(),
        m_ver['precision'].mean(),
        m_ver['F1'].mean()
    ]
    
    # Ángulos para el radar
    angles = [n / float(N) * 2 * pi for n in range(N)]
    fw_values += fw_values[:1]
    m_values += m_values[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, fw_values, 'o-', linewidth=2, label='Agua Dulce', 
            color=COLORS['freshwater'])
    ax.fill(angles, fw_values, alpha=0.25, color=COLORS['freshwater'])
    
    ax.plot(angles, m_values, 'o-', linewidth=2, label='Marino',
            color=COLORS['marine'])
    ax.fill(angles, m_values, alpha=0.25, color=COLORS['marine'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True, shadow=True)
    plt.title('Resumen de Rendimiento: Agua Dulce vs Marino\n(Promedio a través de horizontes)',
              size=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig8_performance_radar.png", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig8_performance_radar.pdf", bbox_inches='tight')
    print("✓ Figura 8 guardada: Gráfico radar de rendimiento")
    plt.close()


def main():
    """Función principal para generar todas las figuras"""
    print("="*60)
    print("Generando figuras comparativas para tesis")
    print("Ambientes: Agua Dulce vs Marino")
    print("="*60)
    print()
    
    # Cargar datos
    print("Cargando datos...")
    nested_metrics, interval_metrics, feature_importance, shap_importance, forecast_verification = load_data()
    print("✓ Datos cargados exitosamente\n")
    
    # Generar figuras
    print("Generando figuras...")
    print()
    
    fig1_skill_comparison(nested_metrics)
    fig2_prauc_comparison(nested_metrics)
    fig3_interval_coverage(interval_metrics)
    fig4_feature_count_comparison(feature_importance)
    fig5_top_features_comparison(shap_importance)
    fig6_forecast_verification(forecast_verification)
    fig7_feature_category_comparison(feature_importance)
    fig8_performance_summary_radar(nested_metrics, forecast_verification)
    
    print()
    print("="*60)
    print("✓ Todas las figuras generadas exitosamente")
    print(f"✓ Ubicación: {OUTPUT_DIR}")
    print("="*60)
    print()
    print("Figuras generadas:")
    print("  1. fig1_skill_comparison - Comparación de Skill Score")
    print("  2. fig2_prauc_comparison - Capacidad de detección de alertas")
    print("  3. fig3_interval_coverage - Cobertura de intervalos de predicción")
    print("  4. fig4_feature_count - Complejidad del modelo")
    print("  5. fig5_top_features_shap - Características más importantes")
    print("  6. fig6_forecast_verification - Verificación de pronósticos")
    print("  7. fig7_feature_categories - Distribución de tipos de características")
    print("  8. fig8_performance_radar - Resumen de rendimiento (radar)")
    print()
    print("Formatos: PNG (alta resolución) y PDF (vectorial)")

if __name__ == "__main__":
    main()
