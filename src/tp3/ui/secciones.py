"""Piezas de interfaz comunes: secciones plegables, páginas desplazables y avisos."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class Seccion(QWidget):
    """Encabezado clicable que muestra u oculta su contenido."""

    def __init__(self, titulo, contenido, abierta=False, parent=None):
        super().__init__(parent)
        self.boton = QToolButton()
        self.boton.setText(titulo)
        self.boton.setCheckable(True)
        self.boton.setChecked(abierta)
        self.boton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.boton.setStyleSheet("QToolButton { font-weight: bold; text-align: left; padding: 4px; }")
        self.boton.toggled.connect(self._alternar)

        self.contenido = contenido
        linea = QFrame()
        linea.setFrameShape(QFrame.HLine)
        linea.setFrameShadow(QFrame.Sunken)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.boton)
        lay.addWidget(contenido)
        lay.addWidget(linea)
        self._alternar(abierta)

    def _alternar(self, abierta):
        self.boton.setArrowType(Qt.DownArrow if abierta else Qt.RightArrow)
        self.contenido.setVisible(abierta)


def pagina(secciones):
    """Solapa desplazable con una lista de (título, widget); abre la primera sección."""
    cuerpo = QWidget()
    lay = QVBoxLayout(cuerpo)
    lay.setContentsMargins(4, 4, 4, 4)
    for i, (titulo, widget) in enumerate(secciones):
        lay.addWidget(Seccion(titulo, widget, abierta=(i == 0)))
    lay.addStretch()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(cuerpo)
    return scroll


def pendiente(texto):
    """Marcador para funciones del TP3 que todavía no están implementadas."""
    etiqueta = QLabel(f"Pendiente: {texto}")
    etiqueta.setWordWrap(True)
    etiqueta.setStyleSheet("color: gray; font-style: italic; padding: 4px;")
    return etiqueta
