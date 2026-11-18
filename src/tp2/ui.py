"""Herramientas de descriptores sobre cualquier imagen y ROI del visor.

Interfaz liviana sobre tp2.engine.run (cálculo) y tp2.storage (guardado
reproducible). Igual que las herramientas de calidad y preprocesamiento,
trabaja con las capas que estén cargadas: imágenes HRF, el pool de retinas
o resultados intermedios.

    HerramientaDescriptores  intensidad, forma, estructura vascular, LBP,
                             Gabor, Wavelet, multiescala, puntos y radiómica
                             sobre una ROI (Shapes, Labels o imagen completa)
    HerramientaSegmentacion  máscara vascular asistida con Frangi, como capa
                             Labels editable para medir forma y estructura

Las funciones puras (parametros_por_defecto, mascara_de_roi, imagen_gris)
no dependen de Qt y se prueban en tests/test_tp2_ui.py.
"""

import json
import traceback

import numpy as np

from . import engine, keypoints, measure, radiomics, structure, texture, storage
from .data import grayscale

# Familias de descriptores: clave del motor -> (nombre visible, función cuyos
# argumentos por defecto se ofrecen como parámetros editables).
ACTIVIDADES = {
    "01_intensidad": ("Intensidad (y corrección por fondo)", measure.intensity),
    "02_geometria": ("Forma de la máscara de vasos o de la ROI", None),
    "03_estructura": ("Esqueleto y estructura vascular", structure.vascular),
    "04_lbp": ("Textura · LBP", texture.lbp_features),
    "04_gabor": ("Textura · banco de Gabor", texture.gabor_bank),
    "04_wavelet": ("Textura · Wavelet", texture.wavelet_features),
    "05_multiescala": ("Multiescala · Gauss, LoG y DoG", texture.multiscale),
    "06_puntos": ("Puntos de interés y correspondencias", keypoints.detect),
    "07_radiomica": ("Radiómica · GLCM", radiomics.extract_radiomics),
}
_NO_PARAMETROS = {"image", "mask", "background"}

OPCION_COMPLETA = "Imagen completa"
OPCION_NINGUNA = "(ninguna)"


# ------------------------------------------------------------------
# Funciones puras
# ------------------------------------------------------------------

def parametros_por_defecto(actividad):
    """Parámetros editables de una familia, tomados de la firma de su función."""
    funcion = ACTIVIDADES[actividad][1]
    if funcion is None:
        return {}
    p = {k: v for k, v in engine.resolved(funcion, {}).items() if k not in _NO_PARAMETROS}
    if actividad == "06_puntos":
        p.update(ratio=0.75, mutual=True, tolerance=3)
    # JSON no tiene tuplas: se muestran como listas.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}


def imagen_gris(datos):
    """2D float en la escala original; RGB/RGBA -> luminancia (misma regla que la conversión a grises)."""
    return grayscale(datos)


def mascara_de_roi(forma, capa=None, indice=-1):
    """Máscara bool de la ROI.

    capa None -> imagen completa. Shapes -> figura `indice` (-1 = unión de todas).
    Labels -> etiqueta `indice` (-1 = toda etiqueta > 0).
    """
    if capa is None:
        return np.ones(forma, bool)
    tipo = type(capa).__name__
    if tipo == "Shapes":
        if len(capa.data) == 0:
            raise ValueError(f"La capa {capa.name} no tiene figuras.")
        mascaras = np.asarray(capa.to_masks(mask_shape=forma), bool)
        if indice < 0:
            return mascaras.any(axis=0)
        if indice >= len(mascaras):
            raise ValueError(f"La capa {capa.name} tiene {len(mascaras)} figuras (índices 0 a {len(mascaras) - 1}).")
        return mascaras[indice]
    if tipo == "Labels":
        datos = np.asarray(capa.data)
        if datos.shape != tuple(forma):
            raise ValueError(f"La capa {capa.name} mide {datos.shape} y la imagen {tuple(forma)}.")
        return datos > 0 if indice < 0 else datos == indice
    raise ValueError(f"La capa {capa.name} no es Shapes ni Labels.")


def resumen(resultado, max_filas=40):
    """Texto para el informe: familia, parámetros efectivos y características."""
    lineas = [
        f"DESCRIPTORES · {ACTIVIDADES[resultado['activity']][0]}",
        f"Imagen: {resultado['context'].get('imagen_id', '')}   ROI: {resultado['context'].get('roi_id', '')}",
        f"Píxeles en la ROI: {int(resultado['mask'].sum())}",
        "Parámetros: " + json.dumps(resultado["params"], ensure_ascii=False, default=str),
        "",
    ]
    for i, (k, v) in enumerate(resultado["features"].items()):
        if i == max_filas:
            lineas.append(f"... y {len(resultado['features']) - max_filas} más (ver CSV al guardar)")
            break
        lineas.append(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    return "\n".join(lineas)


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------

def _capas(viewer, tipos):
    return [c for c in viewer.layers if type(c).__name__ in tipos]


def _llenar(combo, nombres, extra=()):
    actual = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(list(extra) + list(nombres))
    if actual in [combo.itemText(i) for i in range(combo.count())]:
        combo.setCurrentText(actual)
    combo.blockSignals(False)


class HerramientaDescriptores:
    """Calcula una familia de descriptores sobre imagen + ROI y la guarda de forma reproducible."""

    def __init__(self, viewer, carpeta_resultados, informar):
        from qtpy.QtWidgets import (
            QCheckBox, QComboBox, QFormLayout, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
        )

        self.viewer, self.carpeta, self.informar = viewer, carpeta_resultados, informar
        self.pendiente = None

        self.widget = QWidget()
        lay = QVBoxLayout(self.widget)
        form = QFormLayout()
        lay.addLayout(form)

        self.imagen = QComboBox()
        self.roi = QComboBox()
        self.indice = QSpinBox()
        self.indice.setRange(-1, 9999)
        self.indice.setValue(-1)
        self.indice.setToolTip("Shapes: índice de figura · Labels: valor de etiqueta · -1 = todas")
        self.actividad = QComboBox()
        for clave, (nombre, _) in ACTIVIDADES.items():
            self.actividad.addItem(nombre, clave)
        self.vasos = QComboBox()
        self.vasos.setToolTip("Para forma y estructura: Labels con la máscara de vasos (si no, se usa la ROI)")
        self.fondo = QComboBox()
        self.fondo.setToolTip("Para intensidad: ROI de fondo para la corrección (índice -1 = todas sus figuras)")
        self.imagen_b = QComboBox()
        self.imagen_b.setToolTip("Para puntos de interés: segunda imagen con la que buscar correspondencias")
        self.parametros = QPlainTextEdit()
        self.parametros.setMaximumHeight(130)
        self.mapas = QCheckBox("Agregar mapas de respuesta como capas")

        form.addRow("Imagen", self.imagen)
        form.addRow("ROI", self.roi)
        form.addRow("Figura / etiqueta", self.indice)
        form.addRow("Familia", self.actividad)
        form.addRow("Máscara de vasos", self.vasos)
        form.addRow("ROI de fondo", self.fondo)
        form.addRow("Segunda imagen", self.imagen_b)
        form.addRow("Parámetros (JSON)", self.parametros)
        lay.addWidget(self.mapas)

        b_calcular = QPushButton("Calcular")
        b_guardar = QPushButton("Guardar medición (CSV + registro reproducible)")
        b_defecto = QPushButton("Restablecer parámetros por defecto")
        for b in (b_calcular, b_guardar, b_defecto):
            lay.addWidget(b)
        b_calcular.clicked.connect(lambda: self._seguro(self.calcular))
        b_guardar.clicked.connect(lambda: self._seguro(self.guardar))
        b_defecto.clicked.connect(self._cargar_defecto)
        self.actividad.currentIndexChanged.connect(self._cambio_actividad)

        for evento in (viewer.layers.events.inserted, viewer.layers.events.removed):
            evento.connect(self.refrescar)
        self.refrescar()
        self._cambio_actividad()

    # -- estado de la interfaz
    def refrescar(self, event=None):
        imagenes = [c.name for c in _capas(self.viewer, {"Image"})]
        rois = [c.name for c in _capas(self.viewer, {"Shapes", "Labels"})]
        labels = [c.name for c in _capas(self.viewer, {"Labels"})]
        _llenar(self.imagen, imagenes)
        _llenar(self.imagen_b, imagenes, (OPCION_NINGUNA,))
        _llenar(self.roi, rois, (OPCION_COMPLETA,))
        _llenar(self.vasos, labels, (OPCION_NINGUNA,))
        _llenar(self.fondo, rois, (OPCION_NINGUNA,))

    def _cambio_actividad(self):
        clave = self.actividad.currentData()
        self.vasos.setEnabled(clave in ("02_geometria", "03_estructura"))
        self.fondo.setEnabled(clave == "01_intensidad")
        self.imagen_b.setEnabled(clave == "06_puntos")
        self._cargar_defecto()

    def _cargar_defecto(self):
        p = parametros_por_defecto(self.actividad.currentData())
        self.parametros.setPlainText(json.dumps(p, indent=1, ensure_ascii=False))

    def _seguro(self, accion):
        try:
            accion()
        except Exception as e:  # noqa: BLE001  el error se muestra en el informe
            self.informar(f"DESCRIPTORES · ERROR\n{e}\n\n{traceback.format_exc(limit=2)}")

    # -- cálculo
    def _capa(self, nombre):
        return None if nombre in (OPCION_COMPLETA, OPCION_NINGUNA, "") else self.viewer.layers[nombre]

    def calcular(self):
        capa_img = self._capa(self.imagen.currentText())
        if capa_img is None:
            raise ValueError("No hay imágenes en el visor.")
        img = imagen_gris(capa_img.data)
        capa_roi = self._capa(self.roi.currentText())
        roi = mascara_de_roi(img.shape, capa_roi, self.indice.value())
        clave = self.actividad.currentData()
        params = json.loads(self.parametros.toPlainText() or "{}")

        extra = {}
        if clave == "01_intensidad" and self._capa(self.fondo.currentText()) is not None:
            extra["background"] = mascara_de_roi(img.shape, self._capa(self.fondo.currentText()), -1)
        if clave in ("02_geometria", "03_estructura") and self._capa(self.vasos.currentText()) is not None:
            extra["vessels"] = mascara_de_roi(img.shape, self._capa(self.vasos.currentText()), -1) & roi
        if clave == "06_puntos":
            capa_b = self._capa(self.imagen_b.currentText())
            if capa_b is None:
                raise ValueError("Elegir una segunda imagen para los puntos de interés.")
            extra["image_b"] = imagen_gris(capa_b.data)

        nombre_roi = OPCION_COMPLETA if capa_roi is None else f"{capa_roi.name}#{self.indice.value()}"
        contexto = dict(
            imagen_id=capa_img.name,
            variante=str(capa_img.metadata.get("variant", "")) if capa_img.metadata else "",
            roi_id=nombre_roi,
            capa_imagen=capa_img.name,
            capa_region=None if capa_roi is None else capa_roi.name,
            indice_region=self.indice.value(),
        )
        self.pendiente = engine.run(clave, img, roi, params, contexto, **extra)
        self.informar(resumen(self.pendiente) + "\n\nSin guardar: usar «Guardar medición».")
        self._mostrar(self.pendiente, capa_img)

    def _mostrar(self, r, capa_img):
        nombre = f"{capa_img.name} · {r['activity']}"
        for clave, puntos in r["points"].items():
            if len(puntos):
                self.viewer.add_points(np.asarray(puntos), name=f"{nombre} · {clave}", size=6, face_color="yellow")
        if self.mapas.isChecked():
            for clave, mapa in r["maps"].items():
                mapa = np.asarray(mapa, float)
                if mapa.shape == capa_img.data.shape[:2]:
                    self.viewer.add_image(mapa, name=f"{nombre} · {clave}", colormap="magma", visible=False)

    def guardar(self):
        if self.pendiente is None:
            raise ValueError("Primero calcular una medición.")
        identidad = storage.save_measurement(self.carpeta, self.pendiente)
        self.informar(
            resumen(self.pendiente)
            + f"\n\nGuardada como {identidad}: fila en {self.carpeta.name}/{self.pendiente['activity']}.csv "
            f"y registro con la imagen, la ROI y los parámetros en {self.carpeta.name}/mediciones/{identidad}/."
        )


class HerramientaSegmentacion:
    """Máscara vascular asistida (Frangi + umbral) como capa Labels editable."""

    def __init__(self, viewer, informar):
        from qtpy.QtWidgets import (
            QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
        )

        self.viewer, self.informar = viewer, informar
        self.widget = QWidget()
        lay = QVBoxLayout(self.widget)
        form = QFormLayout()
        lay.addLayout(form)
        self.imagen = QComboBox()
        self.roi = QComboBox()
        self.indice = QSpinBox()
        self.indice.setRange(-1, 9999)
        self.indice.setValue(-1)
        self.sigmas = QLineEdit("1, 2, 3")
        self.umbral = QLineEdit("")
        self.umbral.setPlaceholderText("vacío = Otsu dentro de la ROI")
        self.min_obj = QSpinBox()
        self.min_obj.setRange(0, 100000)
        self.min_obj.setValue(20)
        self.cierre = QDoubleSpinBox()
        self.cierre.setRange(0, 10)
        self.cierre.setDecimals(0)
        form.addRow("Imagen", self.imagen)
        form.addRow("ROI", self.roi)
        form.addRow("Figura / etiqueta", self.indice)
        form.addRow("Escalas Frangi σ (px)", self.sigmas)
        form.addRow("Umbral de respuesta", self.umbral)
        form.addRow("Objeto mínimo (px)", self.min_obj)
        form.addRow("Radio de cierre (px)", self.cierre)
        boton = QPushButton("Segmentar vasos (capa Labels editable)")
        lay.addWidget(boton)
        boton.clicked.connect(self._segmentar)
        for evento in (viewer.layers.events.inserted, viewer.layers.events.removed):
            evento.connect(self.refrescar)
        self.refrescar()

    def refrescar(self, event=None):
        _llenar(self.imagen, [c.name for c in _capas(self.viewer, {"Image"})])
        _llenar(self.roi, [c.name for c in _capas(self.viewer, {"Shapes", "Labels"})], (OPCION_COMPLETA,))

    def _segmentar(self):
        try:
            nombre = self.imagen.currentText()
            if not nombre:
                raise ValueError("No hay imágenes en el visor.")
            capa = self.viewer.layers[nombre]
            img = imagen_gris(capa.data)
            capa_roi = None if self.roi.currentText() == OPCION_COMPLETA else self.viewer.layers[self.roi.currentText()]
            roi = mascara_de_roi(img.shape, capa_roi, self.indice.value())
            sigmas = tuple(float(s) for s in self.sigmas.text().replace(";", ",").split(",") if s.strip())
            umbral = float(self.umbral.text()) if self.umbral.text().strip() else None
            mascara, _, t = structure.segment_vessels(
                img, roi, sigmas=sigmas, threshold=umbral, min_size=self.min_obj.value(),
                closing=int(self.cierre.value()),
            )
            self.viewer.add_labels(mascara.astype(np.uint8), name=f"vasos · {nombre}", opacity=0.6)
            self.informar(
                "SEGMENTACIÓN VASCULAR ASISTIDA (Frangi)\n"
                f"Imagen: {nombre}   σ = {sigmas}   umbral = {t:.4g}{' (Otsu)' if umbral is None else ''}\n"
                f"Objeto mínimo = {self.min_obj.value()} px   cierre = {int(self.cierre.value())} px\n"
                f"Píxeles de vaso: {int(mascara.sum())} de {int(roi.sum())} en la ROI\n\n"
                "La capa Labels se puede corregir con el pincel antes de medir forma o estructura."
            )
        except Exception as e:  # noqa: BLE001
            self.informar(f"SEGMENTACIÓN · ERROR\n{e}\n\n{traceback.format_exc(limit=2)}")


def cargar_pool(viewer, carpeta):
    """Agrega como capas las retinas de los TPs anteriores, verificando sus hashes."""
    from .data import read_image, sha256_file

    manifiesto = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    n = 0
    for item in manifiesto["images"]:
        ruta = carpeta / item["path"]
        if sha256_file(ruta) != item["sha256"]:
            raise ValueError(f"Archivo modificado: {item['path']}")
        if item["layer_name"] not in viewer.layers:
            viewer.add_image(
                read_image(ruta, item.get("crop")), name=item["layer_name"], metadata=dict(item),
                colormap="gray", contrast_limits=(0, 255), visible=(n == 0),
            )
            n += 1
    return n
