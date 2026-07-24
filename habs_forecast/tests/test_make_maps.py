"""test_make_maps.py — GUARDA DE REGRESION de build_map_figure (mapas).

Invariante critico: la figura tiene DOS responsabilidades separadas —
  (a) los STATS de retorno (chl_mean, pct_alert, pct_elev, n_water_px), que describen TODO el
      cuerpo de agua, y
  (b) el DISPLAY (recorte/encuadre, suavizado, color), que solo afecta como se ve la figura.
El encuadre recorta al componente de agua mayor; si los stats se calcularan sobre el grid
YA recortado, el agua fuera del cuadro quedaria excluida y los valores de retorno cambiarian
con las banderas de visualizacion (bug real corregido). Este test fija que NO ocurra:
los stats deben ser IDENTICOS con cualquier combinacion de gradient_focus/focus_water.

Es un test de integracion: necesita un raster S2 real y el modelo entrenado del cuerpo. Si
falta cualquiera de los dos (p.ej. CI sin datos), se OMITE (skip), no falla.
"""
import glob
import os

import pytest

# make_maps arrastra rasterio/matplotlib/torch (deps pesadas, ausentes en el CI minimo).
# Igual que el resto del test, si no estan disponibles el modulo se OMITE (skip) en vez de
# romper la coleccion de pytest.
pytest.importorskip("rasterio")
pytest.importorskip("matplotlib")
pytest.importorskip("torch")

import config as C
from make_maps import KEY2FOLDER, build_map_figure, _scene_pixels

# Cuerpo pequeno y rapido (embalse angosto, pocos pixeles) para no cargar escenas enormes.
_BODY = "cajon"
_H = 3                       # permite probar la desagregacion espacial heuristica


def _find_buildable_scene(wb, limit=14):
    """Devuelve la ruta de la escena S2 mas liviana del cuerpo que construya sin error
    (suficiente agua valida + modelo disponible), o None si ninguna sirve / no hay datos."""
    folder = KEY2FOLDER.get(wb)
    if folder is None:
        return None
    tifs = sorted(
        (t for t in glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True)
         if not os.path.basename(t).startswith("LS_")),
        key=os.path.getsize,
    )
    for t in tifs[:limit]:
        try:
            fig, _ = build_map_figure(wb, _H, t, None)   # t0=None -> no depende de datos de target
        except ValueError:
            continue                                     # poca agua/escena no apta -> siguiente
        except (FileNotFoundError, OSError):
            return None                                  # sin modelo entrenado -> no verificable aqui
        else:
            import matplotlib.pyplot as plt
            plt.close(fig)
            return t
    return None


@pytest.fixture(scope="module")
def scene():
    path = _find_buildable_scene(_BODY)
    if path is None:
        pytest.skip(f"sin escena/modelo verificable para {_BODY} (esperado en CI sin datos)")
    return path


_STAT_KEYS = ("chl_mean", "pct_alert", "pct_elev", "n_water_px")


def _stats(path, **flags):
    import matplotlib.pyplot as plt
    fig, stats = build_map_figure(_BODY, _H, path, None, **flags)
    plt.close(fig)
    return stats


def test_stats_independientes_del_encuadre(scene):
    """Los valores de retorno NO deben cambiar con las banderas de VISUALIZACION: describen
    todo el cuerpo, no la region recortada. (Guarda directa del bug del recorte.)"""
    base = _stats(scene, gradient_focus=False, focus_water=False)
    for gf, fw in [(True, False), (False, True), (True, True)]:
        s = _stats(scene, gradient_focus=gf, focus_water=fw)
        for k in _STAT_KEYS:
            assert s[k] == base[k], (
                f"stat '{k}' cambio con gradient_focus={gf}, focus_water={fw}: "
                f"{s[k]} != {base[k]} (el recorte/flags no debe afectar los stats)"
            )


def test_stats_sobre_todo_el_cuerpo(scene):
    """n_water_px debe contar TODA el agua de la mascara (no la recortada), reescalada al
    equivalente en resolucion nativa cuando la escena se leyo decimada (factor > 1)."""
    _, water, decim = _scene_pixels(scene)
    stats = _stats(scene)
    assert stats["n_water_px"] == int(int(water.sum()) * decim * decim)


def test_stats_bien_formados(scene):
    """Contrato de salida: claves presentes, porcentajes en [0, 100], umbrales positivos."""
    s = _stats(scene)
    for k in ("chl_mean", "pct_alert", "pct_elev", "thr", "thr_elev", "n_water_px", "spatial_mode"):
        assert k in s, f"falta la clave '{k}' en stats"
    assert 0.0 <= s["pct_alert"] <= 100.0
    assert 0.0 <= s["pct_elev"] <= 100.0
    assert s["thr"] > 0 and s["thr_elev"] > 0
    assert s["thr_elev"] <= s["thr"]          # orden biologico: elevada < floracion
    assert s["chl_mean"] >= 0


def test_mapa_muestra_gradiente_heuristico(scene):
    s = _stats(scene, gradient_focus=True)
    assert s["spatial_mode"] == "heuristic"
