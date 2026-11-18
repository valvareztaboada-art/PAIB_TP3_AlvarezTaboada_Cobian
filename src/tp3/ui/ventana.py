"""Arma la ventana del TP3: un visor de napari con solapas organizadas por función.

    Datos         base HRF, pool de retinas, ROIs de medición
    Calidad       medición en ROIs, diagnóstico, estimación de ruido, métricas
    Preproceso    escala de grises, ruido, deconvolución, notch, fondo, realce, degradaciones
    Descriptores  descriptores por ROI, segmentación vascular asistida, feature maps por píxel
    Segmentar     vasos con ML clásico y con U-Net
    Clasificar    sano / glaucoma / RD con ML clásico y con CNN
    Resultados    registro y CSV, archivos de resultados, ML vs DL, robustez

Cada solapa reúne herramientas por la tarea que resuelven. Todas trabajan
sobre las capas cargadas en el visor (imágenes HRF, pool o resultados
intermedios). El informe de texto queda fijo debajo del visor.

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


def _boton(texto, accion):
    from qtpy.QtWidgets import QPushButton

    b = QPushButton(texto)
    b.clicked.connect(accion)
    return b


def _abrir_carpeta(ruta):
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices

    ruta.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))


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

    # Herramientas de calidad y preprocesamiento (toman napari.current_viewer()).
    from tp1 import ui as pre

    s = {k: b.native for k, b in pre.secciones.items()}

    def informar(texto):
        pre.informe.value = texto

    # Herramientas de descriptores sobre cualquier imagen y ROI del visor.
    from tp2.ui import HerramientaDescriptores, HerramientaSegmentacion, cargar_pool

    descriptores = HerramientaDescriptores(viewer, rutas.RESULTADOS_DESCRIPTORES, informar)
    segmentacion = HerramientaSegmentacion(viewer, informar)

    def _cargar_pool():
        try:
            n = cargar_pool(viewer, rutas.POOL)
            informar(f"POOL DE RETINAS\n{n} imágenes agregadas desde {rutas.POOL.relative_to(rutas.RAIZ)} (hashes verificados).")
        except Exception as e:  # noqa: BLE001
            informar(f"POOL DE RETINAS · ERROR\n{e}")

    solapas = QTabWidget()
    solapas.setUsesScrollButtons(True)
    paginas = [
        ("Datos", "Datos y regiones", [
            ("Base HRF", pendiente("carga de imágenes, gold standard y FOV como capas; desbalance por imagen y grupo (Act. 1).")),
            ("Pool de retinas (conjunto externo)", _boton("Cargar las 13 retinas de los TPs anteriores", _cargar_pool)),
            ("ROIs de medición", s["rois"]),
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
            ("Descriptores por ROI", descriptores.widget),
            ("Segmentación vascular asistida (Frangi)", segmentacion.widget),
            ("Feature maps por píxel", pendiente("stack de intensidad, Gauss, gradiente, LoG/DoG, Gabor, LBP y estadísticos locales (Act. 3).")),
            ("Descriptores por imagen HRF", pendiente("densidad, esqueleto, calibre, tortuosidad y textura en ROIs peripapilar, macular y periférica (Act. 4).")),
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
            ("Archivos de resultados", _boton("Abrir la carpeta resultados/", lambda: _abrir_carpeta(rutas.RESULTADOS))),
            ("Comparación ML vs DL", pendiente("métricas en test, overlays TP/FP/FN, incertidumbre y eficiencia en datos (Act. 7).")),
            ("Robustez frente a degradaciones", pendiente("tabla modelo vs degradación, con y sin preprocesamiento (Act. 8).")),
        ]),
    ]
    for i, (nombre, ayuda, secciones) in enumerate(paginas):
        solapas.addTab(pagina(secciones), nombre)
        solapas.setTabToolTip(i, ayuda)

    dock = viewer.window.add_dock_widget(solapas, area="right", name="TP3")
    dock.setMinimumWidth(520)
    pre.informe.native.setMaximumHeight(220)
    viewer.window.add_dock_widget(pre.informe.native, area="bottom", name="Informe")
    # Referencias a las herramientas: mantienen vivos los métodos conectados a sus botones.
    viewer.window._herramientas_tp3 = dict(descriptores=descriptores, segmentacion=segmentacion)
    return viewer, solapas
