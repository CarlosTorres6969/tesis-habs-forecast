import os, glob, re, rasterio, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = r'C:\Users\JC\Desktop\Tesis'
OUT = os.path.join(BASE, 'validacion_mapas', 'visual')
os.makedirs(OUT, exist_ok=True)

# Mapa de calor personalizado: azul -> amarillo -> rojo
colors = [(0, 0, 0.8), (0, 0.4, 1), (0.2, 0.8, 0.2), (1, 1, 0), (1, 0.6, 0), (0.8, 0, 0)]
cmap_hab = LinearSegmentedColormap.from_list('hab', colors, N=256)

def rgb_from_s2(bands):
    """Componer RGB real desde bandas S2: B4=Rojo, B3=Verde, B2=Azul.
    Estiramiento global + gamma para colores naturales."""
    r, g, b = bands[2], bands[1], bands[0]
    rgb = np.stack([r, g, b], axis=-1)
    valido = (bands > 0).any(axis=0)
    if valido.sum() > 100:
        todos = rgb[valido]
        lo = np.percentile(todos, 0.5)
        hi = np.percentile(todos, 99.5)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = rgb ** 0.85
    rgb[~valido] = [1, 1, 1]
    return rgb

def load_hab_map(path, st=1):
    """Cargar mapa de probabilidad HAB"""
    with rasterio.open(path) as s:
        hab = s.read(1)
    hab = np.clip(hab, 0, 1)
    if st > 1:
        hab = hab[::st, ::st]
    return hab

def find_matching_pairs(region, top_n=4):
    """Encontrar fechas con mas y menos HAB para una region"""
    import pandas as pd
    dfs = []
    for fname in ['resumen.csv', 'resumen_honduras.csv']:
        p = os.path.join(BASE, 'mapas_finales', fname)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True)
    df_reg = df[df['region'] == region].copy()
    if len(df_reg) == 0:
        return []
    # Top N mas altos y mas bajos
    top = df_reg.nlargest(top_n, 'hab_pct')
    bot = df_reg.nsmallest(top_n, 'hab_pct')
    return pd.concat([top, bot]).drop_duplicates(subset='date')

regions = ['Okeechobee', 'TampaBay', 'Lago de Yojoa', 'Golfo_Fonseca', 'Cajon']

for region in regions:
    print(f'\n=== {region} ===')
    pairs = find_matching_pairs(region, top_n=3)
    if len(pairs) == 0:
        print('  Sin datos de resumen')
        continue
    
    for _, row in pairs.iterrows():
        date_str = row['date']
        hab_pct = row['hab_pct']
        
        # Buscar el GeoTIFF original que coincide con esta fecha
        img_dir = os.path.join(BASE, 'imagenes', region)
        date_glob = date_str.replace('-', '')
        tifs = sorted(glob.glob(os.path.join(img_dir, '**', f'*{date_glob}*.tif'), recursive=True))
        
        if len(tifs) == 0:
            tifs = sorted(glob.glob(os.path.join(img_dir, '**', f'*{date_str}*.tif'), recursive=True))
        
        if len(tifs) == 0:
            print(f'  {date_str} (HAB={hab_pct:.1f}%): No se encontro GeoTIFF original')
            continue
        
        # Elegir el tile con MAS cobertura (menos NoData)
        best_tif, best_coverage = None, 0
        for t in tifs:
            try:
                with rasterio.open(t) as s:
                    b_test = s.read()
                    valido = (b_test > 0).any(axis=0)
                    cov = valido.sum() / valido.size
                    if cov > best_coverage:
                        best_coverage = cov
                        best_tif = t
            except:
                continue
        
        tif_path = best_tif if best_tif else tifs[0]
        basename = os.path.basename(tif_path)
        
        # Buscar el mapa HAB correspondiente
        hab_path = os.path.join(BASE, 'mapas_finales', basename.replace('.tif', '_hab_final.tif'))
        if not os.path.exists(hab_path):
            print(f'  {date_str} (HAB={hab_pct:.1f}%): No se encontro mapa HAB')
            continue
        
        print(f'  {date_str} (HAB={hab_pct:.1f}%): {basename}')
        
        # Cargar imagenes
        # stride=2 para maxima calidad visual
        st = 2
        try:
            with rasterio.open(tif_path) as s:
                bands = s.read()
                if st > 1:
                    bands = bands[:, ::st, ::st]
            
            rgb = rgb_from_s2(bands)
            hab = load_hab_map(hab_path, 1)
            
            # Redimensionar HAB al tamano de RGB si es necesario (interpolacion cubica)
            from scipy.ndimage import zoom
            if hab.shape != rgb.shape[:2]:
                zy = rgb.shape[0] / hab.shape[0]
                zx = rgb.shape[1] / hab.shape[1]
                hab = zoom(hab, (zy, zx), order=3)
            
            # Auto-crop: eliminar bordes sin dato
            valido = (bands > 0).any(axis=0)
            if valido.sum() > 0 and valido.sum() < valido.size * 0.95:
                rows = np.any(valido, axis=1)
                cols = np.any(valido, axis=0)
                y0, y1 = np.where(rows)[0][[0, -1]]
                x0, x1 = np.where(cols)[0][[0, -1]]
                rgb = rgb[y0:y1+1, x0:x1+1]
                hab = hab[y0:y1+1, x0:x1+1]
                print(f'    Crop: ({y0}:{y1+1}, {x0}:{x1+1})')
            
            # Estadisticas
            agua_px = rgb.shape[0] * rgb.shape[1]
            prob_media = float(hab.mean())
            prob_max = float(hab.max())
            prob_p95 = float(np.percentile(hab, 95))
            hab_px_05 = int((hab > 0.5).sum())
            hab_px_01 = int((hab > 0.1).sum())
            es_honduras = region in ('Lago de Yojoa', 'Golfo_Fonseca', 'Cajon')
            es_florida = region in ('Okeechobee', 'TampaBay')
            
            # Crear figura
            fig, axes = plt.subplots(1, 3, figsize=(30, 12))
            fig.suptitle(f'{region}  |  {date_str}  |  Riesgo HAB: {hab_pct:.1f}% del agua en riesgo  |  Prob media: {prob_media:.2f}  |  Prob max: {prob_max:.2f}',
                         fontsize=14, fontweight='bold', y=0.98)
            
            # ---------- Panel 1: RGB ----------
            axes[0].imshow(rgb)
            axes[0].set_title('IMAGEN SATELITAL (RGB: B4-B3-B2)', fontsize=13, fontweight='bold')
            axes[0].axis('off')
            # Leyenda RGB
            if es_honduras:
                leg_rgb = (
                    'QUE SIGNIFICAN LOS COLORES:\n'
                    '  Verde intenso  = Alta clorofila O vegetacion\n'
                    '                   acuatica (lirios, jacinto)\n'
                    '  Verde tenue    = Sedimentos ligeros o biofilm\n'
                    '  Azul oscuro    = Agua clara (sin floracion)\n'
                    '  Marron         = Sedimentos en suspension\n'
                    '  Blanco         = Sin dato satelital\n'
                    '  \n'
                    '  NOTA: En Honduras el verde puede ser\n'
                    '  vegetacion acuatica, NO siempre es algas.\n'
                    '  Modelo: XGBoost con adaptacion de dominio.'
                )
            else:
                leg_rgb = (
                    'QUE SIGNIFICAN LOS COLORES:\n'
                    '  Verde intenso  = Alta clorofila (floracion\n'
                    '                   algal probable)\n'
                    '  Verde tenue    = Baja clorofila o sedimentos\n'
                    '  Azul oscuro    = Agua clara (sin floracion)\n'
                    '  Marron         = Sedimentos en suspension\n'
                    '  Blanco         = Sin dato satelital\n'
                    '  \n'
                    '  Modelo: HABNet1Dv2 (entrenado en Florida).'
                )
            axes[0].text(0.02, 0.02, leg_rgb, transform=axes[0].transAxes, fontsize=10,
                        color='black', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # ---------- Panel 2: Mapa de Probabilidad ----------
            im = axes[1].imshow(hab, cmap=cmap_hab, vmin=0, vmax=1, aspect='equal')
            axes[1].set_title('MAPA DE RIESGO HAB', fontsize=13, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Riesgo HAB (0-1)')
            if es_honduras:
                leg_hab = (
                    'ESCALA DE RIESGO HAB:\n'
                    '  0-10%   = Sin riesgo (agua clara)\n'
                    '  10-20%  = Riesgo bajo\n'
                    '  20-50%  = Riesgo moderado\n'
                    '  >50%    = Riesgo alto\n'
                    '  \n'
                    '  NOTA: En Honduras las probabilidades\n'
                    '  son sistematicamente bajas (p95 ~18%).\n'
                    '  Un valor de 15% ya puede indicar\n'
                    '  condicion de riesgo para esta region.\n'
                    '  Modelo XGBoost (DA): AUC=0.98'
                )
            else:
                leg_hab = (
                    'ESCALA DE RIESGO HAB:\n'
                    '  0-10%   = Sin riesgo (agua clara)\n'
                    '  10-30%  = Riesgo bajo\n'
                    '  30-50%  = Riesgo moderado\n'
                    '  50-75%  = Riesgo alto\n'
                    '  >75%    = Riesgo muy alto\n'
                    '  \n'
                    '  Modelo HABNet1Dv2: AUC=0.99\n'
                    '  Correlacion con clorofila: r=0.78'
                )
            axes[1].text(0.02, 0.02, leg_hab, transform=axes[1].transAxes, fontsize=10,
                        color='black', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # ---------- Panel 3: Superposicion ----------
            axes[2].imshow(rgb)
            hab_masked = np.ma.masked_where(hab < 0.5, hab)
            axes[2].imshow(hab_masked, cmap='Reds', alpha=0.6, vmin=0.5, vmax=1)
            axes[2].set_title('ZONAS DE ALTO RIESGO (riesgo > 50%)', fontsize=13, fontweight='bold')
            axes[2].axis('off')
            if es_honduras:
                leg_over = (
                    f'ESTADISTICAS:\n'
                    f'  Area analizada: {agua_px:,} pixeles\n'
                    f'  Prob media: {prob_media:.2f} | Max: {prob_max:.2f}\n'
                    f'  Riesgo alto (>50%): {hab_pct:.1f}% del agua\n'
                    f'  \n'
                    f'  ROJO = Zona de alto riesgo HAB\n'
                    f'  Modelo XGBoost con adaptacion de dominio\n'
                    f'  Probabilidades bajas vs Florida es NORMAL.\n'
                    f'  Umbral sugerido para Honduras: >10%\n'
                    f'  Validar visualmente con imagen satelital.'
                )
            else:
                leg_over = (
                    f'ESTADISTICAS:\n'
                    f'  Area analizada: {agua_px:,} pixeles\n'
                    f'  Prob media: {prob_media:.2f} | Max: {prob_max:.2f}\n'
                    f'  Riesgo alto (>50%): {hab_pct:.1f}% del agua\n'
                    f'  \n'
                    f'  ROJO = Zona de alto riesgo HAB\n'
                    f'  Modelo entrenado en Florida (AUC=0.99)\n'
                    f'  Correlacion positiva con clorofila\n'
                    f'  Validacion: {region} 2023-2026'
                )
            axes[2].text(0.02, 0.02, leg_over, transform=axes[2].transAxes, fontsize=10,
                        color='black', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # Guardar
            safe_name = f'{region}_{date_str}.png'
            plt.subplots_adjust(top=0.92, wspace=0.05)
            plt.savefig(os.path.join(OUT, safe_name), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'    -> {safe_name}')
            
        except Exception as e:
            print(f'    Error: {e}')
            continue

print(f'\nCompletado. Graficos en: {OUT}')
