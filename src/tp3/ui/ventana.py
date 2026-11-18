"""Arma la ventana del TP3: un visor de napari con solapas organizadas por función.

    Datos         base HRF, ROIs de medición, regiones persistentes
    Calidad       medición en ROIs, diagnóstico, estimación de ruido, métricas
    Preproceso    escala de grises, ruido, deconvolución, notch, fondo, realce, degradaciones
    Descriptores  descriptores por lotes, máscaras vasculares, feature maps por píxel
    Segmentar     vasos con ML clásico y con U-Net
    Clasificar    sano / glaucoma / RD con ML clásico y con CNN
    Resultados    registro y CSV, resultados de descriptores, ML vs DL, robustez

Cada solapa reúne herramientas por la tarea que resuelven, sin importar en qué
trabajo se implementaron. El informe de texto y el progreso de los cálculos
por lotes quedan en un panel fijo debajo del visor.

Para agregar una herramienta nueva, reemplazar el `pendiente(...)` de su
sección por el widget correspondiente.
"""

import os
import sys
from pathlib import Path

from tp3 import rutas


def _precargar_fuentes_vispy():
    """Carga OpenSans desde bytes: evita rutas incompatibles con FreeType en Windows.

    Sin esto, el texto de las capas Shapes/Points puede fallar con rutas que
    tienen caracteres no ASCII.
    """
    from io import BytesIO
    from freetype import Face, FT_FACE_FLAG_SCALABLE
    from vispy.util.fonts import _freetype
    from vispy.util.fonts._vispy_fonts import _get_vispy_font_filename

    for bold, italic in ((False, False), (True, False), (False, True), (True, True)):
        path = Path(_get_vispy_font_filename("OpenSans", bold, italic))
        font = Face(BytesIO(path.read_bytes()))
        if not font.face_flags & FT_FACE_FLAG_SCALABLE:
            raise RuntimeError("La fuente OpenSans instalada no es escalable.")
        _freetype._font_dict[f"OpenSans-{bold}-{italic}"] = font


def _panel_inferior(informe, estado_lotes):
    """Informe de texto a la izquierda; progreso y registro de lotes a la derecha."""
    from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QWidget

    w = QWidget()
    w.setMaximumHeight(220)
    lay = QHBoxLayout(w)
    lay.setContentsMargins(4, 0, 4, 0)
    for titulo, contenido, peso in (("Informe", informe, 3), ("Cálculos por lotes", estado_lotes, 2)):
        caja = QGroupBox(titulo)
        QVBoxLayout(caja).addWidget(contenido)
        lay.addWidget(caja, peso)
    return w


def crear_ventana(show=True):
    """Devuelve (viewer, solapas). No llama a napari.run()."""
    # Las rutas relativas de las herramientas de preprocesamiento
    # (resultados/tp1/...) se resuelven contra la raíz del proyecto.
    os.chdir(rutas.RAIZ)
    if sys.platform == "win32":
        try:
            _precargar_fuentes_vispy()
        except Exception as e:  # noqa: BLE001  sólo afecta al texto de las capas
            print(f"Aviso: no se pudieron precargar las fuentes de vispy ({e}).")

    import napari
    from qtpy.QtWidgets import QTabWidget

    from tp3.ui.secciones import pagina, pendiente

    viewer = napari.Viewer(title="PAIByB · TP3 · Alvarez Taboada / Cobián", show=show)

    # Herramientas de preprocesamiento y calidad (toman napari.current_viewer()).
    from tp1 import ui as pre

    s = {k: b.native for k, b in pre.secciones.items()}

    # Descriptores por lotes. El panel queda oculto: sólo se reparten sus páginas.
    from tp2.batch_ui import Panel as PanelDescriptores

    descriptores = PanelDescriptores(viewer, rutas.ESPACIO_TP2)
    d = descriptores.separar_secciones()

    solapas = QTabWidget()
    solapas.setUsesScrollButtons(True)
    paginas = [
        ("Datos", "Datos y regiones", [
            ("Base HRF", pendiente("carga de imágenes, gold standard y FOV como capas; desbalance por imagen y grupo (Act. 1).")),
            ("ROIs de medición", s["rois"]),
            ("Regiones persistentes", d["rois"]),
        ]),
        ("Calidad", "Calidad de imagen", [
            ("Medición en ROIs (media / varianza / MAD)", s["medicion"]),
            ("Diagnóstico de ruido y artefactos", s["diagnostico"]),
            ("Estimación de ruido", s["ruido"]),
            ("Métricas de calidad (antes / después)", s["metricas"]),
        ]),
        ("Preproceso", "Preprocesamiento y degradaciones", [
            ("Escala de grises y recorte", s["gris"]),
            ("Reducción de ruido", s["filtro"]),
            ("Restauración (deconvolución)", s["deconvolucion"]),
            ("Artefactos periódicos (notch)", s["notch"]),
            ("Corrección de fondo", s["fondo"]),
            ("Realce", s["realce"]),
            ("Degradaciones controladas", pendiente("ruido, artefacto periódico, desenfoque e iluminación no uniforme con parámetros documentados (Act. 8).")),
        ]),
        ("Descriptores", "Descriptores y feature maps", [
            ("Cálculo por lotes", d["lotes"]),
            ("Máscaras vasculares", d["mascaras"]),
            ("Feature maps por píxel", pendiente("stack de intensidad, Gauss, gradiente, LoG/DoG, Gabor, LBP y estadísticos locales (Act. 3).")),
        ]),
        ("Segmentar", "Segmentación de vasos", [
            ("ML clásico (k-NN, SVM, Random Forest)", pendiente("entrenamiento con máscara completa o trazos, mapa de probabilidad y máscara final (Act. 3).")),
            ("Deep Learning (U-Net)", pendiente("carga de pesos, inferencia por parches, probabilidad y máscara final (Act. 5).")),
        ]),
        ("Clasificar", "Clasificación sano / glaucoma / RD", [
            ("ML clásico (k-NN, SVM, Random Forest)", pendiente("vector de descriptores por imagen y validación cruzada (Act. 4).")),
            ("Deep Learning (CNN)", pendiente("CNN desde cero y preentrenada, mapas de activación (Act. 6).")),
        ]),
        ("Resultados", "Registro, resultados y comparación", [
            ("Registro del pipeline y CSV", s["registro"]),
            ("Resultados de descriptores", d["resultados"]),
            ("Comparación ML vs DL", pendiente("métricas en test, overlays TP/FP/FN, incertidumbre y eficiencia en datos (Act. 7).")),
            ("Robustez frente a degradaciones", pendiente("tabla modelo vs degradación, con y sin preprocesamiento (Act. 8).")),
        ]),
    ]
    for i, (nombre, ayuda, secciones) in enumerate(paginas):
        solapas.addTab(pagina(secciones), nombre)
        solapas.setTabToolTip(i, ayuda)

    dock = viewer.window.add_dock_widget(solapas, area="right", name="TP3")
    dock.setMinimumWidth(520)
    viewer.window.add_dock_widget(
        _panel_inferior(pre.informe.native, d["estado"]), area="bottom", name="Informe"
    )
    # Referencia al panel oculto: mantiene vivos sus métodos conectados a los botones.
    viewer.window._panel_descriptores = descriptores
    return viewer, solapas
