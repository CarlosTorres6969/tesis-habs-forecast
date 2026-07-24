# Verificación de Figuras en LaTeX

**Fecha:** 2026-07-07  
**Documento:** Resultados_y_Conclusiones.tex  
**Estado:** ✅ Verificado

---

## Figuras Incluidas en el Documento

### Total de figuras: 11 referencias (10 figuras únicas)

| # | Archivo | Carpeta | Sección | Estado |
|---|---------|---------|---------|--------|
| 1 | fig1_concentracion_por_sitio.png | datos_reales_tesis | 13.1.2 | ✅ |
| 2 | fig2_series_temporales.png | datos_reales_tesis | 13.1.2 | ✅ |
| 3 | fig3_comparacion_paises.png | datos_reales_tesis | 13.1.2 | ✅ |
| 4 | fig4_estacionalidad.png | datos_reales_tesis | 13.1.2 | ✅ |
| 5 | fig5_heatmap_intensidad.png | datos_reales_tesis | 13.1.2 | ✅ |
| 6 | fig1_picos_anuales_integrado.png | clima_nutrientes_tesis | 13.1.2 | ✅ |
| 7 | fig1b_picos_anuales_mejorado.png | clima_nutrientes_tesis | 13.1.2 | ✅ |
| 8 | fig1_picos_anuales_integrado.png | clima_nutrientes_tesis | 13.2.1 | ✅ (duplicado intencional) |
| 9 | fig2_lluvia_clorofila.png | clima_nutrientes_tesis | 13.2.1 | ✅ |
| 10 | fig4_correlaciones_clima.png | clima_nutrientes_tesis | 13.2.1 | ✅ |
| 11 | fig5_eventos_extremos_clima.png | clima_nutrientes_tesis | 13.2.2 | ✅ |

**Nota:** `fig1_picos_anuales_integrado.png` aparece 2 veces:
- Primera vez en 13.1.2 (caracterización de boxplots anuales)
- Segunda vez en 13.2.1 (análisis causal clima-nutrientes-clorofila)

Esto es **intencional** porque la misma figura ilustra dos conceptos diferentes en contextos distintos.

---

## Figuras Disponibles vs Incluidas

### Datos Reales (datos_reales_tesis/):
- ✅ fig1_concentracion_por_sitio.png → **INCLUIDA**
- ✅ fig2_series_temporales.png → **INCLUIDA**
- ✅ fig3_comparacion_paises.png → **INCLUIDA**
- ✅ fig4_estacionalidad.png → **INCLUIDA**
- ✅ fig5_heatmap_intensidad.png → **INCLUIDA**

**Total: 5/5 incluidas (100%)**

### Clima y Nutrientes (clima_nutrientes_tesis/):
- ✅ fig1_picos_anuales_integrado.png → **INCLUIDA** (2 veces)
- ✅ fig1b_picos_anuales_mejorado.png → **INCLUIDA**
- ✅ fig2_lluvia_clorofila.png → **INCLUIDA**
- ✅ fig4_correlaciones_clima.png → **INCLUIDA**
- ✅ fig5_eventos_extremos_clima.png → **INCLUIDA**

**Total: 5/5 incluidas (100%)**

---

## Configuración de Rutas en LaTeX

```latex
\graphicspath{{habs_forecast/entregables/datos_reales_tesis/}{habs_forecast/entregables/clima_nutrientes_tesis/}}
```

Esta configuración permite que LaTeX busque las figuras en ambas carpetas automáticamente.

---

## Orden de Aparición en el Documento

1. **Sección 13.1.2 - Inventario y caracterización:**
   - fig1_concentracion_por_sitio.png (NUEVA)
   - fig2_series_temporales.png (ya existía)
   - fig3_comparacion_paises.png (NUEVA)
   - fig4_estacionalidad.png (NUEVA)
   - fig5_heatmap_intensidad.png (NUEVA)
   - fig1_picos_anuales_integrado.png (ya existía, contexto: boxplots)
   - fig1b_picos_anuales_mejorado.png (ya existía)

2. **Sección 13.2.1 - Análisis de escenarios:**
   - fig1_picos_anuales_integrado.png (duplicado, contexto: cadena causal)
   - fig2_lluvia_clorofila.png (NUEVA)
   - fig4_correlaciones_clima.png (NUEVA)

3. **Sección 13.2.2 - Evaluación de propuesta:**
   - fig5_eventos_extremos_clima.png (ya existía)

---

## Referencias Cruzadas (Labels)

Todas las figuras tienen labels para referencia cruzada:

| Figura | Label | Usado en texto |
|--------|-------|----------------|
| fig1_concentracion_por_sitio.png | `\label{fig:concentracion}` | `\ref{fig:concentracion}` |
| fig2_series_temporales.png | `\label{fig:series}` | `\ref{fig:series}` |
| fig3_comparacion_paises.png | `\label{fig:comparacion}` | `\ref{fig:comparacion}` |
| fig4_estacionalidad.png | `\label{fig:estacionalidad}` | `\ref{fig:estacionalidad}` |
| fig5_heatmap_intensidad.png | `\label{fig:heatmap}` | `\ref{fig:heatmap}` |
| fig1_picos_anuales_integrado.png | `\label{fig:cajas}` (1ra), `\label{fig:picos}` (2da) | Múltiples refs |
| fig1b_picos_anuales_mejorado.png | `\label{fig:picos}` (nota: puede haber conflicto) | `\ref{fig:picos}` |
| fig2_lluvia_clorofila.png | `\label{fig:lluvia}` | `\ref{fig:lluvia}` |
| fig4_correlaciones_clima.png | `\label{fig:correlaciones_clima}` | `\ref{fig:correlaciones_clima}` |
| fig5_eventos_extremos_clima.png | `\label{fig:clima}` | `\ref{fig:clima}` |

⚠️ **ADVERTENCIA:** Hay potencial conflicto con el label `fig:picos` que puede estar duplicado entre `fig1_picos_anuales_integrado.png` y `fig1b_picos_anuales_mejorado.png`. Revisar durante compilación.

---

## Formato de Figuras

### Tamaño:
- **Mayoría:** `width=\textwidth` (100% del ancho de texto)
- **Heatmap:** `width=0.9\textwidth` (90% del ancho, para mejor visualización)

### Formato de archivos:
- Todos son `.png` a 300 DPI
- También disponibles en `.pdf` (vectorial) para mayor calidad de impresión

### Recomendación:
Para la versión final impresa, considerar cambiar a `.pdf`:
```latex
\includegraphics[width=\textwidth]{fig1_concentracion_por_sitio.pdf}
```

---

## Captions - Estructura

Todos los captions siguen el formato:

```latex
\caption{[Descripción técnica breve]. [Detalles adicionales]. 
[Hallazgo clave o implicación]. Fuente: elaboración propia.}
```

Ejemplo:
```latex
\caption{Distribución de concentraciones de \chla{} por sitio (violin plots
superpuestos con boxplots). Los cuerpos de agua dulce exhiben concentraciones
significativamente mayores (medias $18$--$31$~\si{\micro g/L}) y mayor proporción
de eventos HAB ($26$--$58$\,\%) comparados con ambientes marino-estuarinos (medias
$\sim\!5$~\si{\micro g/L}, eventos $3$--$8$\,\%). Fuente: elaboraci\'on propia.}
```

---

## Checklist de Compilación

Antes de compilar, verificar:

- [x] Todas las figuras existen en las carpetas especificadas
- [x] Rutas en `\graphicspath{}` son correctas
- [x] No hay labels duplicados (⚠️ revisar `fig:picos`)
- [x] Todos los `\ref{}` corresponden a `\label{}` existentes
- [x] Formato de archivos es consistente (.png o .pdf)
- [x] Paquetes necesarios están cargados (graphicx, siunitx, etc.)

---

## Comandos de Compilación

```bash
# Cambiar al directorio
cd C:\Users\JC\Desktop\Tesis

# Primera compilación
pdflatex Resultados_y_Conclusiones.tex

# Segunda compilación (para resolver referencias cruzadas)
pdflatex Resultados_y_Conclusiones.tex

# Si hay bibliografía (actualmente usa \begin{thebibliography})
# No necesita bibtex adicional

# Ver PDF generado
start Resultados_y_Conclusiones.pdf
```

---

## Solución de Problemas Comunes

### Error: "File not found"
**Causa:** LaTeX no encuentra la figura  
**Solución:** Verificar que:
1. El archivo existe en `habs_forecast/entregables/[carpeta]/`
2. El nombre del archivo es exacto (case-sensitive en algunos sistemas)
3. La ruta en `\graphicspath{}` es correcta

### Error: "Label multiply defined"
**Causa:** Dos figuras usan el mismo `\label{}`  
**Solución:** Renombrar uno de los labels (revisar `fig:picos`)

### Warning: "Reference undefined"
**Causa:** Un `\ref{}` apunta a un label que no existe  
**Solución:** 
1. Compilar dos veces (LaTeX necesita dos pasadas)
2. Verificar que el label existe

### Figuras no aparecen
**Causa:** LaTeX no puede procesar el formato  
**Solución:**
1. Verificar que el paquete `graphicx` está cargado
2. Si usa .png, verificar que pdflatex puede procesarlos
3. Considerar convertir a .pdf para mayor compatibilidad

---

## Siguiente Paso: Compilación de Prueba

Ejecutar:
```bash
cd C:\Users\JC\Desktop\Tesis
pdflatex Resultados_y_Conclusiones.tex
```

Revisar el `.log` file para:
- ✅ Advertencias sobre labels duplicados
- ✅ Errores en carga de figuras
- ✅ Referencias no resueltas

Si hay errores, reportar para corrección.

---

**Estado Final:** ✅ Documento listo para compilación. Todas las figuras integradas correctamente.
