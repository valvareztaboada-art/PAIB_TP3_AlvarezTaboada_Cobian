"""Interfaz de la solapa TP1 (preprocesamiento y evaluación de calidad).

Es el widget del TP1 sin cambios funcionales. Las funciones de cálculo están
en tp1.core. El módulo toma el visor ya abierto (napari.current_viewer()),
así que debe importarse después de crear el Viewer. Expone:

    secciones  -> dict de bloques funcionales (ROIs, diagnóstico, filtros,
                  fondo, realce, métricas, registro...); la ventana del
                  TP3 los reparte en solapas por función
    informe    -> panel de texto del informe, se acopla abajo
"""

from pathlib import Path
from datetime import datetime

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    LineEdit,
    TextEdit,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from napari.layers import Image, Shapes
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from .core import *  # noqa: F401,F403  funciones puras del TP1

# ============================================================
# 4. VISOR Y ESTADO
# ============================================================

viewer = napari.current_viewer()
if viewer is None:
    raise RuntimeError("Crear el napari.Viewer antes de importar tp1.ui (lo hace widget_tp3.py).")

registros_pendientes = []

informe = TextEdit(value="INFORME\nTodavía no se realizó ningún cálculo.")
informe.native.setReadOnly(True)
informe.native.setMinimumHeight(180)
estado_csv = Label(value="CSV\nNo hay resultados pendientes.")


def agregar_pendientes(registros):
    """Suma filas a las pendientes de guardar en el CSV."""
    registros_pendientes.extend(registros)
    estado_csv.value = (
        f"CSV\n{len(registros_pendientes)} fila(s) pendiente(s) de guardado."
    )


# ============================================================
# 5. CAPA DE ROIs (base de la clase, para Actividades 1 y 2)
#    Cada ROI puede llevar una etiqueta con el segmento que representa
#    (fondo, vaso, disco óptico...). Se muestra sobre la imagen y se
#    guarda en la columna "etiqueta_roi" del CSV.
# ============================================================

PALETA_ROIS = np.array([
    [1.00, 0.20, 0.20, 1.00],   # rojo
    [0.10, 0.80, 1.00, 1.00],   # celeste
    [0.20, 0.90, 0.35, 1.00],   # verde
    [1.00, 0.75, 0.10, 1.00],   # naranja
    [0.75, 0.35, 1.00, 1.00],   # violeta
    [1.00, 0.35, 0.75, 1.00],   # magenta
])


def obtener_capa_roi():
    if "ROI" not in viewer.layers:
        return None
    capa = viewer.layers["ROI"]
    if not isinstance(capa, Shapes):
        return None
    return capa


def etiqueta_roi(roi_layer, indice):
    """Etiqueta (segmento) de una ROI, o "" si no tiene."""
    if roi_layer is None or "etiqueta" not in roi_layer.features:
        return ""
    valor = roi_layer.features["etiqueta"].iloc[indice]
    return "" if valor is None or str(valor) == "nan" else str(valor)


def nombre_roi(roi_layer, indice):
    """'ROI 2' o 'ROI 2 (fondo)'."""
    etiqueta = etiqueta_roi(roi_layer, indice)
    return f"ROI {indice + 1}" + (f" ({etiqueta})" if etiqueta else "")


def actualizar_rois(event=None):
    """Colores por ROI y rótulo visible 'ROI n (etiqueta)'."""
    roi_layer = obtener_capa_roi()
    if roi_layer is None:
        return
    n = len(roi_layer.data)
    if n == 0:
        return
    colores_borde = np.array(
        [PALETA_ROIS[i % len(PALETA_ROIS)] for i in range(n)]
    )
    colores_cara = colores_borde.copy()
    colores_cara[:, 3] = 0.12
    roi_layer.edge_color = colores_borde
    roi_layer.face_color = colores_cara
    roi_layer.edge_width = 3

    features = roi_layer.features.copy()
    if "etiqueta" not in features:
        features["etiqueta"] = ""
    features["etiqueta"] = features["etiqueta"].fillna("").astype(str)
    features["rotulo"] = [
        f"ROI {i + 1}" + (f" ({e})" if e else "")
        for i, e in enumerate(features["etiqueta"])
    ]
    roi_layer.features = features


boton_crear_roi = PushButton(text="Crear capa de ROIs")


@boton_crear_roi.clicked.connect
def crear_capa_rois():
    roi_layer = obtener_capa_roi()

    if roi_layer is None:
        roi_layer = viewer.add_shapes(
            name="ROI",
            edge_width=3,
            features={"etiqueta": np.array([], dtype=object),
                      "rotulo": np.array([], dtype=object)},
            feature_defaults={"etiqueta": "", "rotulo": ""},
            text={"string": "{rotulo}", "color": "white", "size": 9,
                  "anchor": "upper_left"},
        )
        roi_layer.events.data.connect(actualizar_rois)
        mensaje = "Capa ROI creada.\nDibujá una o más regiones."
    else:
        mensaje = "La capa ROI ya existe.\nSe volvió a activar."

    viewer.layers.selection.active = roi_layer
    for widget in (widget_medicion, widget_gris, widget_diagnostico,
                   widget_ruido, widget_filtro, widget_deconvolucion,
                   widget_notch, widget_fondo, widget_realce,
                   widget_metricas):
        widget.roi.value = roi_layer
    roi_layer.mode = "add_rectangle"

    informe.value = f"INFORME\n{mensaje}"


etiqueta_nueva = LineEdit(value="fondo", label="Etiqueta (segmento)")
boton_etiquetar = PushButton(text="Asignar etiqueta a las ROIs seleccionadas")


@boton_etiquetar.clicked.connect
def etiquetar_rois():
    roi_layer = obtener_capa_roi()
    try:
        if roi_layer is None or len(roi_layer.data) == 0:
            raise ValueError("Primero creá la capa de ROIs y dibujá alguna ROI.")
        seleccionadas = sorted(roi_layer.selected_data)
        if not seleccionadas:
            raise ValueError("Seleccioná una o más ROIs con la herramienta Select.")

        actualizar_rois()   # asegura que existan las columnas
        features = roi_layer.features.copy()
        etiqueta = etiqueta_nueva.value.strip()
        for i in seleccionadas:
            features.loc[i, "etiqueta"] = etiqueta
        roi_layer.features = features
        actualizar_rois()

        informe.value = (
            "INFORME\nEtiqueta asignada:\n"
            + "\n".join(nombre_roi(roi_layer, i) for i in seleccionadas)
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo etiquetar:\n{error}"


# ============================================================
# 6. HERRAMIENTA DE MEDICIÓN (Actividades 1 y 2)
#    Media / varianza / desvío / MAD sobre la imagen completa,
#    la ROI seleccionada o todas las ROIs (una fila del CSV por región).
# ============================================================

@magicgui(
    call_button="Medir estadísticas",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada", "Todas las ROIs"],
        "label": "Región a analizar",
    },
    excluir_marco={"label": "Quitar marco (imagen completa)"},
    registrar={"label": "Agregar al CSV"},
)
def widget_medicion(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        img = a_escala_de_grises(imagen.data)

        # (region, etiqueta, pixeles)
        if modo == "Imagen completa":
            datos = quitar_marco(img)[0] if excluir_marco else img
            region = "Imagen completa" + (" (sin marco)" if excluir_marco else "")
            regiones = [(region, "", datos)]
        else:
            if roi is None:
                raise ValueError("Creá primero una capa de ROIs.")
            if modo == "ROI seleccionada":
                indices = [obtener_indice_roi_seleccionada(roi)]
            else:
                if len(roi.data) == 0:
                    raise ValueError("Dibujá al menos una ROI.")
                indices = range(len(roi.data))
            origen = origen_capa(imagen)
            regiones = [
                (f"ROI {i + 1}", etiqueta_roi(roi, i),
                 obtener_datos_roi(img, roi, i, origen))
                for i in indices
            ]

        bloques = []
        registros = []
        for region, etiqueta, datos in regiones:
            est = estadisticas_robustas(datos)
            titulo_region = region + (f" ({etiqueta})" if etiqueta else "")
            bloques.append(
                f"{titulo_region}\n"
                f"  n = {est['n']}   media = {est['media']:.3f}\n"
                f"  varianza = {est['varianza']:.3f}   desvío = {est['desviacion']:.3f}\n"
                f"  mediana = {est['mediana']:.3f}   MAD = {est['mad']:.3f}"
                f"   σ_MAD = {est['sigma_mad']:.3f}"
            )
            registros.append(crear_registro(
                imagen=imagen.name,
                actividad="Medición",
                region=region,
                etiqueta_roi=etiqueta,
                metodo="Estadísticas de la región",
                n_px=est["n"],
                media=est["media"],
                desvio=est["desviacion"],
                mediana=est["mediana"],
                sigma_mad=est["sigma_mad"],
                resultados={"varianza": est["varianza"], "mad": est["mad"]},
            ))

        if registrar:
            agregar_pendientes(registros)

        informe.value = (
            "INFORME - MEDICIÓN\n"
            f"Capa: {imagen.name}\n\n"
            + "\n\n".join(bloques)
            + (f"\n\n{len(registros)} fila(s) agregada(s) al CSV pendiente."
               if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo calcular:\n{error}"


# ============================================================
# 6.1 CONVERSIÓN A ESCALA DE GRISES (paso previo a la Act. 1)
#     Las Retina*.png vienen como RGBA con R = G = B: se pasan a 2D.
#     Sobre la imagen completa o sobre el recorte de la ROI.
# ============================================================

def origen_capa(capa):
    """Desplazamiento (fila, columna) de una capa en el mundo."""
    return np.asarray(capa.translate, dtype=float)[-2:]


def agregar_o_actualizar_capa(nombre, datos, **kwargs):
    """Si la capa existe le cambia los datos; si no, la crea."""
    if nombre in viewer.layers:
        capa = viewer.layers[nombre]
        capa.data = datos
        for clave, valor in kwargs.items():
            setattr(capa, clave, valor)
        return capa
    return viewer.add_image(datos, name=nombre, **kwargs)


def region_en_gris(imagen, modo, roi, excluir_marco):
    """
    Devuelve (imagen_gris_2D, origen, descripcion) de la región pedida,
    con origen en coordenadas del mundo de napari.
    """
    gris = a_escala_de_grises(imagen.data)
    origen = origen_capa(imagen)

    if modo == "Imagen completa":
        region, desplazamiento = gris, (0, 0)
        descripcion = "Imagen completa"
    else:
        if roi is None:
            raise ValueError("Creá primero una capa de ROIs.")
        indice = obtener_indice_roi_seleccionada(roi)
        vertices = np.asarray(roi.data[indice]) - origen
        region, desplazamiento = recortar_roi(gris, vertices)
        descripcion = nombre_roi(roi, indice)

    if excluir_marco:
        region, (df, dc) = quitar_marco(region)
        desplazamiento = (desplazamiento[0] + df, desplazamiento[1] + dc)
        descripcion += " (sin marco)"

    return region, origen + np.asarray(desplazamiento, dtype=float), descripcion


def region_y_etiqueta(modo, roi):
    """('Imagen completa', '') o ('ROI 2', 'fondo'), para las columnas del CSV."""
    if modo == "Imagen completa":
        return "Imagen completa", ""
    indice = obtener_indice_roi_seleccionada(roi)
    return f"ROI {indice + 1}", etiqueta_roi(roi, indice)


@magicgui(
    call_button="Convertir a escala de grises",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a convertir",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
)
def widget_gris(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = False,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        if modo == "Imagen completa":
            nombre = f"{imagen.name}_gris"
        else:
            nombre = f"{imagen.name}_{descripcion.split(' (')[0].replace(' ', '')}_gris"
        capa = agregar_o_actualizar_capa(
            nombre,
            region,
            colormap="gray",
            translate=tuple(origen),
        )
        viewer.layers.selection.active = capa

        informe.value = (
            "INFORME\n"
            f"Capa creada: {nombre}\n"
            f"Región: {descripcion}\n"
            f"Tamaño: {region.shape[0]} x {region.shape[1]} (2D, float)\n"
            f"Rango: {region.min():.1f} - {region.max():.1f}"
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo convertir:\n{error}"


# ============================================================
# 6.2 DIAGNÓSTICO DE LA IMAGEN (Actividad 1)
#     Histogramas, perfiles, estadísticos + MAD, variabilidad local,
#     diferencias, Fourier, autocorrelación y wavelet, más una
#     hipótesis sugerida de la degradación dominante.
# ============================================================

ventanas_figuras = []   # referencias para que Qt no cierre las ventanas


def obtener_capa_perfil():
    if "Perfil" not in viewer.layers:
        return None
    capa = viewer.layers["Perfil"]
    if not isinstance(capa, Shapes):
        return None
    return capa


boton_crear_perfil = PushButton(text="Crear capa de perfil (línea)")


@boton_crear_perfil.clicked.connect
def crear_capa_perfil():
    capa = obtener_capa_perfil()
    if capa is None:
        capa = viewer.add_shapes(
            name="Perfil", shape_type="line", edge_color="yellow", edge_width=2
        )
    viewer.layers.selection.active = capa
    capa.mode = "add_line"
    informe.value = (
        "INFORME\nDibujá una línea en la capa 'Perfil'.\n"
        "Se usa la última línea para el perfil de intensidad."
    )


def perfiles_intensidad(gris_completa, origen, region, origen_region):
    """
    Perfiles de intensidad. Si hay una línea en la capa 'Perfil' se usa;
    si no, se toman la fila y la columna centrales de la región.
    Devuelve una lista de (etiqueta, valores).
    """
    capa = obtener_capa_perfil()
    lineas = []
    if capa is not None:
        lineas = [
            np.asarray(d) for d, tipo in zip(capa.data, capa.shape_type)
            if tipo in ("line", "path") and len(d) >= 2
        ]

    if lineas:
        vertices = lineas[-1] - origen
        valores = profile_line(
            gris_completa, vertices[0], vertices[-1],
            linewidth=1, mode="constant", cval=np.nan,
        )
        return [("línea 'Perfil'", valores)]

    fila, columna = region.shape[0] // 2, region.shape[1] // 2
    f_abs = int(origen_region[0] + fila)
    c_abs = int(origen_region[1] + columna)
    return [
        (f"horizontal (fila {f_abs})", region[fila, :]),
        (f"vertical (columna {c_abs})", region[:, columna]),
    ]


def histogramas_regionales(gris_completa, origen):
    """Píxeles de cada ROI dibujada (para comparar con el global)."""
    roi_layer = obtener_capa_roi()
    if roi_layer is None:
        return []
    resultado = []
    for i in range(len(roi_layer.data)):
        try:
            datos = obtener_datos_roi(gris_completa, roi_layer, i, origen)
        except ValueError:
            continue
        if datos.size > 1:
            resultado.append((i, datos))
    return resultado


def figura_diagnostico(nombre, res, perfiles, regionales):
    """Arma la figura resumen de la Actividad 1 (4 x 4 paneles)."""
    fig = Figure(figsize=(17, 13), layout="constrained")
    ejes = fig.subplots(4, 4)
    img = res["imagen"]
    est = res["estadisticas"]

    # --- Fila 1: imagen, histogramas, perfiles
    ax = ejes[0, 0]
    ax.imshow(img, cmap="gray")
    ax.set_title("Región analizada (gris)")

    ax = ejes[0, 1]
    bins = np.linspace(img.min(), img.max() + 1e-9, 128)
    ax.hist(img.ravel(), bins=bins, density=True, color="0.4", label="global")
    for i, datos in regionales:
        ax.hist(
            datos, bins=bins, density=True, histtype="step", linewidth=1.5,
            color=PALETA_ROIS[i % len(PALETA_ROIS)],
            label=nombre_roi(obtener_capa_roi(), i),
        )
    ax.axvline(est["media"], color="k", linestyle="--", linewidth=1)
    ax.set_title("Histograma global y regional")
    ax.set_xlabel("intensidad")
    ax.legend(fontsize=7)

    ax = ejes[0, 2]
    for etiqueta, valores in perfiles:
        ax.plot(valores, linewidth=0.8, label=etiqueta)
    ax.set_title("Perfiles de intensidad")
    ax.set_xlabel("posición (px)")
    ax.legend(fontsize=7)

    ax = ejes[0, 3]
    ax.axis("off")
    ind = res["indicadores"]
    texto = (
        f"n = {est['n']}\n"
        f"media = {est['media']:.2f}\n"
        f"varianza = {est['varianza']:.2f}\n"
        f"desvío = {est['desviacion']:.2f}\n"
        f"mediana = {est['mediana']:.2f}\n"
        f"MAD = {est['mad']:.2f} (σ_MAD = {est['sigma_mad']:.2f})\n\n"
        f"σ diferencias (std) = {res['diferencias']['sigma_std']:.2f}\n"
        f"σ diferencias (MAD) = {res['diferencias']['sigma_mad']:.2f}\n"
        f"σ wavelet (Donoho) = {res['wavelet']['sigma_donoho']:.2f}\n"
        f"σ wavelet / rango = {ind['sigma_wavelet_sobre_rango']:.3f}\n"
        f"curtosis diferencias = {ind['curtosis_diferencias']:.1f}\n"
        f"impulsos = {100 * ind['fraccion_impulsos']:.2f} %\n"
        f"pico FFT = {ind['pico_fft_prominencia']:.2f} "
        f"(período {ind['pico_fft_periodo_px']:.1f} px)\n"
        f"fondo borde/centro = {ind['fondo_borde_centro']:.2f}\n"
        f"E altas / E medias = {ind['relacion_altas_medias']:.3f}"
    )
    ax.text(0, 1, texto, va="top", family="monospace", fontsize=9)
    ax.set_title(nombre)

    # --- Fila 2: variabilidad local, diferencias, fondo
    ax = ejes[1, 0]
    im = ax.imshow(res["variabilidad"], cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Variabilidad local (desvío)")

    ax = ejes[1, 1]
    im = ax.imshow(np.abs(res["diferencias"]["dx"]), cmap="magma",
                   vmax=np.percentile(np.abs(res["diferencias"]["dx"]), 99))
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("|Diferencia horizontal|")

    ax = ejes[1, 2]
    dx = res["diferencias"]["dx"].ravel()
    dy = res["diferencias"]["dy"].ravel()
    lim = np.ceil(np.percentile(np.abs(np.concatenate([dx, dy])), 99.5))
    bins_d = np.arange(-lim - 0.5, lim + 1.5, 1)   # centrados en enteros
    ax.hist(dx, bins=bins_d, histtype="step", label="dx", density=True)
    ax.hist(dy, bins=bins_d, histtype="step", label="dy", density=True)
    ax.set_yscale("log")
    ax.set_title("Histograma de diferencias")
    ax.legend(fontsize=7)

    ax = ejes[1, 3]
    im = ax.imshow(res["fondo"]["fondo"], cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Fondo (gaussiano de gran escala)")

    # --- Fila 3: Fourier y autocorrelación
    fourier = res["fourier"]
    ax = ejes[2, 0]
    ax.imshow(fourier["log_magnitud"], cmap="gray")
    fila, columna = fourier["pico_posicion"]
    ax.plot(columna, fila, "o", markerfacecolor="none", color="r", markersize=10)
    ax.set_title("log |FFT| (o = pico más prominente)")

    ax = ejes[2, 1]
    ax.semilogy(fourier["frecuencias_radiales"][1:], fourier["psd_radial"][1:])
    ax.set_xlabel("frecuencia (ciclos/px)")
    ax.set_title("Espectro de potencia radial")

    auto = res["autocorrelacion"]
    ax = ejes[2, 2]
    alto, ancho = auto["mapa"].shape
    lag = min(40, alto // 2 - 1, ancho // 2 - 1)
    ax.imshow(
        auto["mapa"][alto // 2 - lag:alto // 2 + lag + 1,
                     ancho // 2 - lag:ancho // 2 + lag + 1],
        cmap="coolwarm", vmin=-1, vmax=1, extent=(-lag, lag, lag, -lag),
    )
    ax.set_title("Autocorrelación (zoom)")

    ax = ejes[2, 3]
    ax.plot(auto["perfil_x"][:lag + 1], label="horizontal")
    ax.plot(auto["perfil_y"][:lag + 1], label="vertical")
    ax.axhline(0.5, color="0.6", linestyle=":")
    ax.set_xlabel("lag (px)")
    ax.set_title(f"Autocorrelación (lag1 = {auto['lag1_x']:.2f} / {auto['lag1_y']:.2f})")
    ax.legend(fontsize=7)

    # --- Fila 4: wavelet
    ondita = res["wavelet"]
    for ax, clave, titulo in [
        (ejes[3, 0], "horizontal_1", "Wavelet detalle H (nivel 1)"),
        (ejes[3, 1], "vertical_1", "Wavelet detalle V (nivel 1)"),
        (ejes[3, 2], "diagonal_1", "Wavelet detalle D (nivel 1)"),
    ]:
        coef = ondita[clave]
        lim = np.percentile(np.abs(coef), 99)
        ax.imshow(coef, cmap="gray", vmin=-lim, vmax=lim)
        ax.set_title(titulo)

    ax = ejes[3, 3]
    niveles = np.arange(1, len(ondita["energias_detalle"]) + 1)
    ax.bar(niveles, ondita["energias_detalle"], color="0.4")
    ax.set_yscale("log")
    ax.set_xticks(niveles)
    ax.set_xlabel("nivel (1 = más fino)")
    ax.set_title("Energía de detalle por nivel")

    for ax in [ejes[0, 0], ejes[1, 0], ejes[1, 1], ejes[1, 3],
               ejes[2, 0], ejes[3, 0], ejes[3, 1], ejes[3, 2]]:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def mostrar_figura(fig, titulo):
    """
    Muestra una figura de matplotlib en una ventana Qt aparte.
    El lienzo va dentro de un QScrollArea con su tamaño real, así la
    ventana puede ser más chica que la pantalla sin cortar nada:
    si no entra, aparecen barras de desplazamiento.
    """
    ventana = QWidget()
    ventana.setWindowTitle(titulo)
    layout = QVBoxLayout(ventana)

    canvas = FigureCanvasQTAgg(fig)
    ancho_px = int(fig.get_figwidth() * fig.dpi)
    alto_px = int(fig.get_figheight() * fig.dpi)
    canvas.setMinimumSize(ancho_px, alto_px)   # conserva el tamaño real

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(canvas)

    layout.addWidget(NavigationToolbar2QT(canvas, ventana))
    layout.addWidget(scroll)

    ventana.resize(1200, 800)   # entra en pantallas de notebook
    ventana.show()
    ventanas_figuras.append(ventana)


@magicgui(
    call_button="Diagnosticar",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a analizar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    ventana={"label": "Ventana variab. local", "min": 3, "max": 51, "step": 2},
    wavelet={"choices": ["haar", "db2", "db4", "sym4"], "label": "Wavelet"},
    niveles={"label": "Niveles wavelet", "min": 1, "max": 6},
    agregar_mapas={"label": "Agregar mapas como capas"},
    guardar_figura={"label": "Guardar figura en resultados/tp1/"},
    registrar={"label": "Agregar al CSV"},
)
def widget_diagnostico(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    ventana: int = 7,
    wavelet: str = "db2",
    niveles: int = 3,
    agregar_mapas: bool = True,
    guardar_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        res = diagnosticar(region, ventana=ventana, wavelet=wavelet, niveles=niveles)

        gris_completa = a_escala_de_grises(imagen.data)
        origen = origen_capa(imagen)
        perfiles = perfiles_intensidad(gris_completa, origen, region, origen_region)
        regionales = histogramas_regionales(gris_completa, origen)

        nombre = f"{imagen.name} - {descripcion}"
        fig = figura_diagnostico(nombre, res, perfiles, regionales)
        mostrar_figura(fig, f"Diagnóstico: {nombre}")

        ruta_figura = ""
        if guardar_figura:
            carpeta = Path("resultados/tp1")
            carpeta.mkdir(parents=True, exist_ok=True)
            sufijo = descripcion.split(" (")[0].replace(" ", "")
            ruta_figura = carpeta / f"diagnostico_{imagen.name}_{sufijo}.png"
            fig.savefig(ruta_figura, dpi=110)

        if agregar_mapas:
            base = f"{imagen.name}_diag"
            traslado = tuple(origen_region)
            agregar_o_actualizar_capa(
                f"{base}_variabilidad", res["variabilidad"],
                colormap="magma", translate=traslado, visible=False,
            )
            agregar_o_actualizar_capa(
                f"{base}_fondo", res["fondo"]["fondo"],
                colormap="viridis", translate=traslado, visible=False,
            )
            agregar_o_actualizar_capa(
                f"{base}_fft_log", res["fourier"]["log_magnitud"],
                colormap="gray", visible=False,
            )
            agregar_o_actualizar_capa(
                f"{base}_autocorrelacion", res["autocorrelacion"]["mapa"],
                colormap="twilight_shifted", visible=False,
            )

        est = res["estadisticas"]
        ind = res["indicadores"]

        if registrar:
            region_fila, etiqueta = region_y_etiqueta(modo, roi)
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="1 - Diagnóstico",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo="Diagnóstico",
                parametros={
                    "tamano": f"{region.shape[0]}x{region.shape[1]}",
                    "sin_marco": excluir_marco,
                    "ventana": ventana,
                    "wavelet": wavelet,
                    "niveles": niveles,
                },
                n_px=est["n"],
                media=est["media"],
                desvio=est["desviacion"],
                mediana=est["mediana"],
                sigma_mad=est["sigma_mad"],
                resultados=ind,
                observaciones=str(ruta_figura),
            )])

        informe.value = (
            "INFORME - DIAGNÓSTICO\n"
            f"Capa: {imagen.name}\n"
            f"Región: {descripcion} ({region.shape[0]}x{region.shape[1]})\n"
            f"Media: {est['media']:.2f}   Varianza: {est['varianza']:.2f}\n"
            f"Desvío: {est['desviacion']:.2f}   Mediana: {est['mediana']:.2f}   "
            f"MAD: {est['mad']:.2f}\n\n"
            "INDICADORES\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}" for clave, valor in ind.items()
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
            + ("\nFigura: " + str(ruta_figura) if ruta_figura else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo diagnosticar:\n{error}"


# ============================================================
# 6.3 ESTIMACIÓN DE RUIDO (Actividad 2)
#     Estrategia principal elegida por el usuario + las demás que se
#     puedan calcular con los datos disponibles, para comparar.
# ============================================================

# Última σ estimada: la Act. 3 la puede copiar como parámetro (a pedido).
ultimo_sigma_ruido = {}

REFERENCIA_LIMPIA = "Referencia sin ruido (σ = std(A-B))"
ADQUISICION_INDEPENDIENTE = "Adquisición independiente (σ = std(A-B)/√2)"


def figura_ruido(nombre, res):
    """Figura de la Actividad 2 (2 x 2 paneles)."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    metodo = res["metodo"]
    principal = res["principal"]
    sigma = res["sigma"]

    # --- Histograma de la muestra usada vs. N(centro, σ)
    ax = ejes[0, 0]
    muestra = principal["muestra"]
    centro = principal["centro"]
    ancho = max(5 * sigma, 3.0)
    if np.allclose(muestra, np.round(muestra)):
        paso = max(1, int(round(sigma / 3)))
        bins = np.arange(np.floor(centro - ancho) - 0.5, centro + ancho + 1, paso)
    else:
        bins = np.linspace(centro - ancho, centro + ancho, 61)
    ax.hist(muestra, bins=bins, density=True, color="0.6", label="datos")
    if sigma > 0:
        x = np.linspace(bins[0], bins[-1], 300)
        ax.plot(
            x, np.exp(-0.5 * ((x - centro) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
            "r", label=f"N({centro:.1f}, σ={sigma:.2f})",
        )
    if "media_fija" in principal:
        ax.axvline(principal["media_fija"], color="b", linestyle="--",
                   label=f"media fija = {principal['media_fija']:g}")
    ax.set_title(f"{metodo}: muestra usada vs. gaussiana")
    ax.legend(fontsize=7)

    # --- Comparación de estrategias
    ax = ejes[0, 1]
    nombres, valores, colores = [], [], []
    for nombre_m, r in res["resultados"].items():
        if "sigma" in r:
            nombres.append(nombre_m.replace(" entre ", "\nentre ").replace(" ", "\n", 1))
            valores.append(r["sigma"])
            colores.append("tab:red" if nombre_m == metodo else "0.5")
    ax.bar(range(len(valores)), valores, color=colores)
    ax.set_xticks(range(len(valores)))
    ax.set_xticklabels(nombres, fontsize=7)
    ax.set_ylabel("σ estimado")
    ax.set_title("Comparación de estrategias (rojo = elegida)")

    # --- Media vs. varianza de las ROIs (dependencia con la señal)
    ax = ejes[1, 0]
    multiples = res["resultados"]["Múltiples ROIs"]
    if "rois" in multiples:
        for indice, _, media, varianza, _ in multiples["rois"]:
            ax.plot(media, varianza, "o",
                    color=PALETA_ROIS[indice % len(PALETA_ROIS)])
            ax.annotate(nombre_roi(obtener_capa_roi(), indice),
                        (media, varianza), fontsize=7)
        aj = multiples["ajuste"]
        if aj is not None:
            medias = np.array([f[2] for f in multiples["rois"]])
            x = np.linspace(medias.min(), medias.max(), 50)
            ax.plot(x, aj["a"] + aj["b"] * x, "k--",
                    label=f"var = {aj['a']:.1f} + {aj['b']:.3f}·μ (R²={aj['r2']:.2f})")
            ax.legend(fontsize=7)
        ax.set_xlabel("media de la ROI")
        ax.set_ylabel("varianza de la ROI")
    else:
        ax.text(0.5, 0.5, "Dibujá 2 o más ROIs homogéneas\na distintas intensidades",
                ha="center", va="center")
    ax.set_title("Varianza vs. media por ROI")

    # --- Mapa del residuo de ruido
    ax = ejes[1, 1]
    if metodo == "Diferencias entre adquisiciones":
        mapa, titulo_mapa = principal["mapa"], "Diferencia entre adquisiciones (A - B)"
    elif "mapa" in res["resultados"]["Coeficientes wavelet"]:
        mapa = res["resultados"]["Coeficientes wavelet"]["mapa"]
        titulo_mapa = "Coeficientes wavelet HH1 (nivel 1)"
    else:
        mapa, titulo_mapa = None, ""
    if mapa is not None:
        lim = max(np.percentile(np.abs(mapa - np.median(mapa)), 99), 1e-9)
        im = ax.imshow(mapa, cmap="gray", vmin=np.median(mapa) - lim,
                       vmax=np.median(mapa) + lim)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(titulo_mapa)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(nombre)
    return fig


def clave_csv(texto):
    """'Dark-field / flat-field' -> 'dark_field_flat_field' (para claves y capas)."""
    reemplazos = str.maketrans("áéíóúñ /-", "aeioun___")
    clave = texto.lower().translate(reemplazos)
    while "__" in clave:
        clave = clave.replace("__", "_")
    return clave.strip("_")


@magicgui(
    call_button="Estimar ruido",
    metodo={"choices": METODOS_RUIDO, "label": "Estrategia"},
    modo={
        "choices": ["ROI seleccionada", "Imagen completa"],
        "label": "Región (dif. / wavelet)",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    usar_media_fija={"label": "Usar media fija (no la observada)"},
    media_fija={"label": "Media fija", "min": -1e6, "max": 1e6, "step": 1.0},
    segunda={"label": "Otra adquisición"},
    tipo_segunda={
        "choices": [REFERENCIA_LIMPIA, ADQUISICION_INDEPENDIENTE],
        "label": "La otra imagen es",
    },
    wavelet={"choices": ["haar", "db2", "db4", "sym4"], "label": "Wavelet"},
    ver_figura={"label": "Mostrar figura"},
    guardar_figura={"label": "Guardar figura en resultados/tp1/"},
    registrar={"label": "Agregar al CSV"},
)
def widget_ruido(
    imagen: Image,
    metodo: str = "ROI homogénea",
    modo: str = "ROI seleccionada",
    roi: Shapes = None,
    excluir_marco: bool = True,
    usar_media_fija: bool = False,
    media_fija: float = 0.0,
    segunda: Image = None,
    tipo_segunda: str = REFERENCIA_LIMPIA,
    wavelet: str = "db2",
    ver_figura: bool = True,
    guardar_figura: bool = False,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        gris = a_escala_de_grises(imagen.data)
        origen = origen_capa(imagen)
        notas = []
        motivos = {}

        # ROI seleccionada: píxeles del polígono y su rectángulo
        datos_roi = recorte_roi = None
        region_roi, etiqueta = "", ""
        try:
            indice = obtener_indice_roi_seleccionada(roi)
            datos_roi = obtener_datos_roi(gris, roi, indice, origen)
            recorte_roi, _ = recortar_roi(gris, np.asarray(roi.data[indice]) - origen)
            region_roi, etiqueta = f"ROI {indice + 1}", etiqueta_roi(roi, indice)
        except ValueError as error:
            motivos["ROI"] = str(error)

        # Región rectangular para diferencias locales / wavelet / adquisiciones
        modo_efectivo = modo
        if modo == "ROI seleccionada" and datos_roi is None:
            modo_efectivo = "Imagen completa"
            notas.append(
                "No hay ROI seleccionada: diferencias y wavelet se calcularon "
                "sobre la imagen completa."
            )
        region, origen_region, descripcion = region_en_gris(
            imagen, modo_efectivo, roi, excluir_marco
        )

        # Misma región en la otra adquisición
        region_b = None
        if segunda is None:
            motivos["segunda"] = "Elegí otra adquisición en 'Otra adquisición'."
        elif segunda is imagen:
            motivos["segunda"] = "La otra adquisición es la misma capa."
        else:
            gris_b = a_escala_de_grises(segunda.data)
            if gris_b.shape != gris.shape:
                motivos["segunda"] = (
                    f"'{segunda.name}' tiene otro tamaño ({gris_b.shape})."
                )
            else:
                f0, c0 = np.round(origen_region - origen).astype(int)
                alto, ancho = region.shape
                region_b = gris_b[f0:f0 + alto, c0:c0 + ancho]

        referencia_limpia = tipo_segunda == REFERENCIA_LIMPIA
        res = estimar_ruido(
            region,
            metodo=metodo,
            datos_roi=datos_roi,
            recorte_roi=recorte_roi,
            lista_rois=histogramas_regionales(gris, origen),
            segunda=region_b,
            referencia_limpia=referencia_limpia,
            media_fija=media_fija if usar_media_fija else None,
            wavelet=wavelet,
            motivos=motivos,
        )
        principal = res["principal"]
        sigma = res["sigma"]

        imagen_util = quitar_marco(gris)[0] if excluir_marco else gris
        rango = np.percentile(imagen_util, 99) - np.percentile(imagen_util, 1)
        roi_layer = obtener_capa_roi()

        # --- Región y etiqueta de la fila principal
        if metodo in ("ROI homogénea", "MAD"):
            region_fila = region_roi
            etiqueta_fila = etiqueta
            donde = f"{nombre_roi(roi_layer, indice)} ({principal['n']} px)"
        elif metodo == "Múltiples ROIs":
            indices = [f[0] for f in principal["rois"]]
            region_fila = ", ".join(f"ROI {i + 1}" for i in indices)
            etiquetas = [etiqueta_roi(roi_layer, i) for i in indices]
            etiqueta_fila = ", ".join(sorted({e for e in etiquetas if e}))
            donde = ", ".join(nombre_roi(roi_layer, i) for i in indices)
        else:
            region_fila, etiqueta_fila = region_y_etiqueta(modo_efectivo, roi)
            donde = f"{descripcion} ({region.shape[0]}x{region.shape[1]})"
        if metodo == "Diferencias entre adquisiciones":
            donde = f"{imagen.name} - {segunda.name}, {donde}"

        # --- Valores propios de la estrategia
        valores = {"sigma_sobre_rango": sigma / max(rango, 1e-12)}
        detalle = []
        if metodo == "ROI homogénea":
            valores["snr_media_sobre_sigma"] = principal["snr"]
            detalle.append(
                f"Media observada = {principal['media']:.2f}   "
                f"SNR = media/σ = {principal['snr']:.1f}"
            )
        elif metodo == "MAD":
            valores["mad"] = principal["mad"]
            detalle.append(
                f"Mediana = {principal['mediana']:.2f}   MAD = {principal['mad']:.2f}"
            )
        elif metodo == "Múltiples ROIs":
            valores["cv_sigma_entre_rois"] = principal["cv_sigma"]
            for i, n, media, _, desvio in principal["rois"]:
                detalle.append(
                    f"{nombre_roi(roi_layer, i)}: n={n}  media={media:.2f}  σ={desvio:.2f}"
                )
            detalle.append(f"CV de σ entre ROIs = {principal['cv_sigma']:.3f}")
            aj = principal["ajuste"]
            if aj is not None:
                valores.update(var_vs_media_a=aj["a"], var_vs_media_b=aj["b"],
                               var_vs_media_r2=aj["r2"])
                detalle.append(
                    f"Ajuste varianza = a + b·media: a = {aj['a']:.3f}, "
                    f"b = {aj['b']:.4f}, R² = {aj['r2']:.3f}"
                )
        elif metodo == "Diferencias entre adquisiciones":
            valores.update(sigma_std=principal["sigma_std"],
                           offset_medio=principal["offset"],
                           correlacion=principal["correlacion"])
            detalle.append(
                f"σ (MAD) = {sigma:.2f}   σ (std) = {principal['sigma_std']:.2f}   "
                f"offset medio A-B = {principal['offset']:+.2f}   "
                f"correlación = {principal['correlacion']:.3f}"
            )
        elif metodo == "Diferencias locales":
            valores.update(sigma_std=principal["sigma_std"],
                           curtosis=principal["curtosis"])
            detalle.append(
                f"σ (MAD) = {sigma:.2f}   σ (std) = {principal['sigma_std']:.2f}   "
                f"curtosis = {principal['curtosis']:.2f}"
            )

        if "media_fija" in principal:
            valores.update(media_fija=principal["media_fija"],
                           delta=principal["delta"],
                           sigma_alrededor_media_fija=principal["sigma_media_fija"])
            nivel = "media" if metodo == "ROI homogénea" else "mediana"
            detalle.append(
                f"Media fija μ0 = {principal['media_fija']:g}:   "
                f"Δ = {nivel} - μ0 = {principal['delta']:+.2f}   "
                f"dispersión alrededor de μ0 = {principal['sigma_media_fija']:.2f}"
            )

        valores.update(res["complementarios"])
        for nombre_m, r in res["resultados"].items():
            if nombre_m != metodo and "sigma" in r:
                valores[f"sigma_por_{clave_csv(nombre_m)}"] = r["sigma"]

        ultimo_sigma_ruido.update(
            valor=sigma, imagen=imagen.name, metodo=metodo, region=donde
        )

        # --- Figura
        ruta_figura = ""
        if ver_figura or guardar_figura:
            nombre_fig = f"Ruido: {imagen.name} - {metodo}"
            fig = figura_ruido(nombre_fig, res)
            if ver_figura:
                mostrar_figura(fig, nombre_fig)
            if guardar_figura:
                carpeta = Path("resultados/tp1")
                carpeta.mkdir(parents=True, exist_ok=True)
                ruta_figura = carpeta / f"ruido_{imagen.name}_{clave_csv(metodo)}.png"
                fig.savefig(ruta_figura, dpi=110)

        # --- Filas del CSV
        if registrar:
            parametros = {"sin_marco": excluir_marco}
            if usar_media_fija and metodo in ("ROI homogénea", "MAD"):
                parametros["media_fija"] = media_fija
            if metodo == "Coeficientes wavelet":
                parametros["wavelet"] = wavelet
            if metodo == "Diferencias entre adquisiciones":
                parametros["otra_adquisicion"] = segunda.name
                parametros["tipo"] = "referencia" if referencia_limpia else "independiente"

            estadisticas = {}
            if metodo in ("ROI homogénea", "MAD"):
                est = estadisticas_robustas(datos_roi)
                estadisticas = dict(
                    n_px=est["n"], media=est["media"], desvio=est["desviacion"],
                    mediana=est["mediana"], sigma_mad=est["sigma_mad"],
                )

            filas = []
            if metodo == "Múltiples ROIs":
                # una fila por ROI, para comparar niveles e intensidades
                for i, n, media, _, desvio in principal["rois"]:
                    est = estadisticas_robustas(
                        obtener_datos_roi(gris, roi_layer, i, origen)
                    )
                    filas.append(crear_registro(
                        imagen=imagen.name,
                        actividad="2 - Estimación de ruido",
                        region=f"ROI {i + 1}",
                        etiqueta_roi=etiqueta_roi(roi_layer, i),
                        metodo="Múltiples ROIs (cada ROI)",
                        n_px=n, media=media, desvio=desvio,
                        mediana=est["mediana"], sigma_mad=est["sigma_mad"],
                        estimacion_ruido=desvio,
                    ))
            filas.append(crear_registro(
                imagen=imagen.name,
                actividad="2 - Estimación de ruido",
                region=region_fila,
                etiqueta_roi=etiqueta_fila,
                metodo=metodo,
                parametros=parametros,
                estimacion_ruido=sigma,
                resultados=valores,
                observaciones=str(ruta_figura),
                **estadisticas,
            ))
            agregar_pendientes(filas)

        comparacion = []
        for nombre_m, r in res["resultados"].items():
            marca = "→" if nombre_m == metodo else " "
            valor = f"{r['sigma']:.3f}" if "sigma" in r else f"no aplica ({r['error']})"
            comparacion.append(f"{marca} {nombre_m}: {valor}")

        complementarios = [
            f"  {clave} = {formatear_valor(valor)}"
            for clave, valor in res["complementarios"].items()
        ]

        informe.value = (
            "INFORME - ESTIMACIÓN DE RUIDO\n"
            f"Capa: {imagen.name}\n"
            f"Estrategia: {metodo}\n"
            f"Región: {donde}\n"
            f"σ = {sigma:.3f}   (σ / rango p1-p99 = {100 * valores['sigma_sobre_rango']:.2f}%)\n"
            + "".join(f"{d}\n" for d in detalle)
            + "\nCOMPARACIÓN DE ESTRATEGIAS (σ)\n"
            + "\n".join(comparacion)
            + "\n\nOTROS VALORES\n"
            + "\n".join(complementarios)
            + ("\n\nNOTAS\n" + "\n".join(f"- {n}" for n in notas) if notas else "")
            + (f"\n\n{len(filas)} fila(s) agregada(s) al CSV pendiente."
               if registrar else "")
            + ("\nFigura: " + str(ruta_figura) if ruta_figura else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo estimar el ruido:\n{error}"


# ============================================================
# 6.4 REDUCCIÓN DE RUIDO (Actividad 3)
#     Se elige el filtro y sus parámetros; el resultado es una capa
#     nueva (entrada de los pasos siguientes). El panel muestra solo
#     los parámetros del método elegido.
# ============================================================

# Parámetros que usa cada método (el resto se oculta en el panel)
PARAMETROS_FILTRO = {
    "Mediana": ["ventana_mediana", "forma_mediana", "umbral_impulso"],
    "Gaussiano": ["sigma_espacial"],
    "Bilateral": ["sigma_color", "sigma_espacial"],
    "Difusión anisotrópica": ["iteraciones", "kappa", "gamma", "funcion_conduccion"],
    "Non-Local Means": ["h", "sigma_ruido", "tamano_parche", "distancia_busqueda"],
    "Variación Total": ["peso_tv"],
    "Wavelet thresholding": [
        "sigma_ruido", "wavelet", "umbral", "modo_umbral", "niveles_wavelet",
    ],
    "Wiener local": ["ventana", "sigma_ruido"],
    "BM3D": ["sigma_ruido", "etapas_bm3d"],
}
TODOS_PARAMETROS_FILTRO = sorted(
    {nombre for nombres in PARAMETROS_FILTRO.values() for nombre in nombres}
)


def escala_intensidad(datos):
    """Máximo del tipo de dato (255 en uint8): pasa intensidades a [0, 1]."""
    datos = np.asarray(datos)
    if np.issubdtype(datos.dtype, np.integer):
        return float(np.iinfo(datos.dtype).max)
    return 1.0 if np.nanmax(datos) <= 1 else float(np.nanmax(datos))


def figura_filtro(nombre, original, filtrada):
    """Original, filtrada, residuo (original - filtrada) y perfil central."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    ax = ejes[0, 1]
    ax.imshow(filtrada, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Filtrada (misma escala de grises)")

    ax = ejes[1, 0]
    residuo = original - filtrada
    lim = max(np.percentile(np.abs(residuo), 99), 1e-9)
    im = ax.imshow(residuo, cmap="gray", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Residuo = original - filtrada (lo que se quitó)")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 1]
    fila = original.shape[0] // 2
    ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
    ax.plot(filtrada[fila], color="tab:red", linewidth=1.0, label="filtrada")
    ax.set_xlabel("columna (px)")
    ax.set_title(f"Perfil horizontal (fila central {fila})")
    ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


def medidas_filtrado(region, filtrada, origen_region, roi):
    """
    σ wavelet antes/después y, si hay una ROI seleccionada que cae entera
    dentro de la región, su media y desvío antes/después.
    Devuelve (antes, despues, nombre_de_la_roi_medida).
    """
    antes = {"sigma_wavelet": descomposicion_wavelet(region, "db2", 1)["sigma_donoho"]}
    despues = {"sigma_wavelet": descomposicion_wavelet(filtrada, "db2", 1)["sigma_donoho"]}

    try:
        indice = obtener_indice_roi_seleccionada(roi)
    except ValueError:
        return antes, despues, ""

    en_region = [i for i, _ in rois_en_arreglo(region, origen_region, roi)]
    if indice not in en_region:
        return antes, despues, ""

    datos_antes = dict(rois_en_arreglo(region, origen_region, roi))[indice]
    datos_despues = dict(rois_en_arreglo(filtrada, origen_region, roi))[indice]
    antes.update(media_roi=float(datos_antes.mean()),
                 desvio_roi=float(datos_antes.std(ddof=1)))
    despues.update(media_roi=float(datos_despues.mean()),
                   desvio_roi=float(datos_despues.std(ddof=1)))
    return antes, despues, nombre_roi(roi, indice)


boton_usar_sigma = PushButton(text="Copiar σ de la última estimación de ruido")


@boton_usar_sigma.clicked.connect
def copiar_ultimo_sigma():
    if not ultimo_sigma_ruido:
        informe.value = "INFORME\nTodavía no se estimó el ruido (Calidad › Estimación de ruido)."
        return
    widget_filtro.sigma_ruido.value = round(float(ultimo_sigma_ruido["valor"]), 3)
    informe.value = (
        "INFORME\nσ copiado al parámetro 'σ del ruido':\n"
        f"σ = {ultimo_sigma_ruido['valor']:.3f} ({ultimo_sigma_ruido['metodo']}, "
        f"{ultimo_sigma_ruido['imagen']}, {ultimo_sigma_ruido['region']})"
    )


@magicgui(
    call_button="Aplicar filtro",
    metodo={"choices": METODOS_FILTRO, "label": "Filtro"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a filtrar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    sigma_espacial={"label": "σ espacial (px)", "min": 0.1, "max": 20.0, "step": 0.1},
    sigma_color={"label": "σ color (intensidad)", "min": 0.1, "max": 1000.0, "step": 0.5},
    iteraciones={"label": "Iteraciones", "min": 1, "max": 500},
    kappa={"label": "κ (intensidad)", "min": 0.1, "max": 1000.0, "step": 0.5},
    gamma={"label": "γ (paso, ≤ 0.25)", "min": 0.01, "max": 0.25, "step": 0.01},
    funcion_conduccion={
        "choices": ["exponencial", "cuadrática"],
        "label": "Función de conducción",
    },
    h={"label": "h (intensidad)", "min": 0.1, "max": 1000.0, "step": 0.5},
    sigma_ruido={
        "label": "σ del ruido (0 = auto)", "min": 0.0, "max": 1000.0, "step": 0.1,
    },
    tamano_parche={"label": "Tamaño de parche (px)", "min": 3, "max": 21},
    distancia_busqueda={"label": "Distancia de búsqueda (px)", "min": 1, "max": 30},
    peso_tv={"label": "Peso TV", "min": 0.001, "max": 5.0, "step": 0.005},
    wavelet={"choices": ["haar", "db2", "db4", "sym4", "sym8", "coif1"], "label": "Wavelet"},
    umbral={"choices": ["BayesShrink", "VisuShrink"], "label": "Umbral"},
    modo_umbral={"choices": ["soft", "hard"], "label": "Umbralado"},
    niveles_wavelet={"label": "Niveles (0 = auto)", "min": 0, "max": 8},
    ventana={"label": "Ventana (px)", "min": 3, "max": 31, "step": 2},
    ventana_mediana={"label": "Ventana de la mediana (px)", "min": 3, "max": 31, "step": 2},
    forma_mediana={"choices": ["Cuadrada", "Disco"], "label": "Forma de la ventana"},
    umbral_impulso={
        "label": "Umbral de impulso (0 = todos)", "min": 0.0, "max": 255.0, "step": 1.0,
    },
    etapas_bm3d={
        "choices": ["Completo (2 etapas)", "Solo umbral duro (1 etapa)"],
        "label": "Etapas BM3D",
    },
    agregar_residuo={"label": "Agregar capa de residuo"},
    factor_barrido={"label": "Factor del barrido", "min": 1.1, "max": 10.0, "step": 0.1},
    ver_figura={"label": "Mostrar figura comparativa"},
    registrar={"label": "Agregar al CSV"},
)
def widget_filtro(
    imagen: Image,
    metodo: str = "Gaussiano",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    sigma_espacial: float = 1.0,
    sigma_color: float = 20.0,
    iteraciones: int = 15,
    kappa: float = 20.0,
    gamma: float = 0.2,
    funcion_conduccion: str = "exponencial",
    h: float = 10.0,
    sigma_ruido: float = 0.0,
    tamano_parche: int = 5,
    distancia_busqueda: int = 6,
    peso_tv: float = 0.05,
    wavelet: str = "db2",
    umbral: str = "BayesShrink",
    modo_umbral: str = "soft",
    niveles_wavelet: int = 0,
    ventana: int = 5,
    ventana_mediana: int = 3,
    forma_mediana: str = "Cuadrada",
    umbral_impulso: float = 0.0,
    etapas_bm3d: str = "Completo (2 etapas)",
    agregar_residuo: bool = False,
    factor_barrido: float = 2.0,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        escala = escala_intensidad(imagen.data)

        todos = dict(
            sigma_espacial=sigma_espacial, sigma_color=sigma_color,
            iteraciones=iteraciones, kappa=kappa, gamma=gamma,
            funcion_conduccion=funcion_conduccion, h=h, sigma_ruido=sigma_ruido,
            tamano_parche=tamano_parche, distancia_busqueda=distancia_busqueda,
            peso_tv=peso_tv, wavelet=wavelet, umbral=umbral,
            modo_umbral=modo_umbral, niveles_wavelet=niveles_wavelet,
            ventana=ventana, ventana_mediana=ventana_mediana,
            forma_mediana=forma_mediana, umbral_impulso=umbral_impulso,
            etapas_bm3d=etapas_bm3d,
        )
        parametros = {k: todos[k] for k in PARAMETROS_FILTRO[metodo]}

        inicio = datetime.now()
        filtrada = reducir_ruido(region, metodo, escala=escala, **todos)
        segundos = (datetime.now() - inicio).total_seconds()

        # --- Capas nuevas (en la posición de la región original)
        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, filtrada, colormap="gray",
            translate=tuple(origen_region),
            contrast_limits=tuple(imagen.contrast_limits),
        )
        residuo = region - filtrada
        if agregar_residuo:
            lim = max(float(np.percentile(np.abs(residuo), 99)), 1e-9)
            agregar_o_actualizar_capa(
                f"{nombre_capa}_residuo", residuo, colormap="gray",
                translate=tuple(origen_region), contrast_limits=(-lim, lim),
                visible=False,
            )
        viewer.layers.selection.active = capa

        # --- Valores antes / después (sin interpretar)
        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        antes, despues, roi_medida = medidas_filtrado(region, filtrada, origen_region, roi)

        resultados = {
            "tiempo_s": segundos,
            "residuo_media": float(residuo.mean()),
            "residuo_desvio": float(residuo.std()),
            "capa_resultado": nombre_capa,
        }
        if roi_medida:
            resultados["roi_medida"] = roi_medida

        ruta_figura = ""
        if ver_figura:
            nombre_fig = f"Filtro: {imagen.name} - {metodo}"
            fig = figura_filtro(
                f"{nombre_fig} ({formatear_resultados(parametros)})", region, filtrada
            )
            mostrar_figura(fig, nombre_fig)

        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="3 - Reducción de ruido",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                estimacion_ruido=sigma_ruido if "sigma_ruido" in parametros and sigma_ruido > 0 else "",
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
            )])

        lineas_medidas = [
            f"  {clave}: {formatear_valor(antes[clave])} → {formatear_valor(despues[clave])}"
            for clave in antes
        ]
        informe.value = (
            "INFORME - REDUCCIÓN DE RUIDO\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Filtro: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s\n"
            f"Capa creada: {nombre_capa}\n\n"
            "ANTES → DESPUÉS\n"
            + "\n".join(lineas_medidas)
            + (f"\n  (ROI medida: {roi_medida})" if roi_medida else "")
            + f"\n\nResiduo (original - filtrada): media = {residuo.mean():.3f}, "
            f"desvío = {residuo.std():.3f}"
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo filtrar:\n{error}"


def actualizar_parametros_filtro(event=None):
    """Muestra solo los parámetros del método elegido."""
    visibles = PARAMETROS_FILTRO[widget_filtro.metodo.value]
    for nombre in TODOS_PARAMETROS_FILTRO:
        getattr(widget_filtro, nombre).visible = nombre in visibles


widget_filtro.metodo.changed.connect(actualizar_parametros_filtro)


# ------------------------------------------------------------
# Barrido: subfiltrado / filtrado adecuado / sobrefiltrado
# ------------------------------------------------------------

# Parámetro que regula la FUERZA de cada filtro (más grande = filtra más).
# Los que son ventanas se redondean al impar más cercano.
PARAMETRO_BARRIDO_FILTRO = {
    "Mediana": "ventana_mediana",
    "Gaussiano": "sigma_espacial",
    "Bilateral": "sigma_color",
    "Difusión anisotrópica": "kappa",
    "Non-Local Means": "h",
    "Variación Total": "peso_tv",
    "Wavelet thresholding": "sigma_ruido",
    "Wiener local": "ventana",
    "BM3D": "sigma_ruido",
}
PARAMETROS_VENTANA = {"ventana_mediana", "ventana"}


def valores_barrido(valor, factor, es_ventana=False):
    """(menor, valor, mayor) = (valor/factor, valor, valor·factor)."""
    if not es_ventana:
        return valor / factor, valor, valor * factor

    def impar(x):
        n = max(3, int(round(x)))
        return n if n % 2 == 1 else n + 1

    # Ventanas: impares y distintas entre sí (salvo que la base ya sea 3,
    # que es la ventana más chica posible).
    valor = impar(valor)
    menor = impar(valor / factor)
    if menor >= valor:
        menor = max(3, valor - 2)
    mayor = impar(valor * factor)
    if mayor <= valor:
        mayor = valor + 2
    return menor, valor, mayor


def figura_barrido_filtro(nombre, original, filtradas, parametro):
    """
    Fila de arriba: original y las tres filtradas (misma escala de grises).
    Fila de abajo: perfil central y los residuos (lo que quitó cada una).
    `filtradas` es [(etiqueta, valor, imagen), ...].
    """
    fig = Figure(figsize=(18, 9), layout="constrained")
    ejes = fig.subplots(2, 4)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ejes[0, 0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ejes[0, 0].set_title("Original")

    residuos = [original - imagen for _, _, imagen in filtradas]
    lim = max(max(np.percentile(np.abs(r), 99) for r in residuos), 1e-9)
    for k, (etiqueta, valor, imagen) in enumerate(filtradas):
        ax = ejes[0, k + 1]
        ax.imshow(imagen, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{etiqueta} ({parametro} = {valor:g})")
        ax = ejes[1, k + 1]
        ax.imshow(residuos[k], cmap="gray", vmin=-lim, vmax=lim)
        ax.set_title(f"Residuo: {etiqueta.lower()}")

    for ax in list(ejes[0]) + list(ejes[1, 1:]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 0]
    fila = original.shape[0] // 2
    ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
    for etiqueta, valor, imagen in filtradas:
        ax.plot(imagen[fila], linewidth=1.0, label=f"{etiqueta} ({valor:g})")
    ax.set_xlabel("columna (px)")
    ax.set_title(f"Perfil horizontal (fila central {fila})")
    ax.legend(fontsize=7)

    fig.suptitle(nombre + "  -  residuos en la misma escala")
    return fig


boton_barrido_filtro = PushButton(text="Crear sub / adecuado / sobrefiltrado")


@boton_barrido_filtro.clicked.connect
def barrido_filtro():
    """
    Aplica el filtro tres veces: con el parámetro de fuerza del panel y
    con ese parámetro dividido y multiplicado por el factor. Crea las tres
    capas, una figura comparativa y tres filas del CSV.
    """
    try:
        valores = widget_filtro.asdict()
        metodo = valores["metodo"]
        imagen = valores["imagen"]
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de reducción de ruido.")

        parametro = PARAMETRO_BARRIDO_FILTRO[metodo]
        base = valores[parametro]
        if base <= 0:
            raise ValueError(
                f"Para barrer '{parametro}' hace falta un valor > 0 "
                "(con 0 el método lo estima solo y no hay nada que escalar)."
            )
        menor, base, mayor = valores_barrido(
            base, valores["factor_barrido"], parametro in PARAMETROS_VENTANA
        )
        barrido = [
            ("Subfiltrado", menor),
            ("Filtrado adecuado", base),
            ("Sobrefiltrado", mayor),
        ]

        roi = valores["roi"]
        region, origen_region, descripcion = region_en_gris(
            imagen, valores["modo"], roi, valores["excluir_marco"]
        )
        escala = escala_intensidad(imagen.data)
        todos = {k: valores[k] for k in TODOS_PARAMETROS_FILTRO}
        region_fila, etiqueta = region_y_etiqueta(valores["modo"], roi)

        filtradas, filas, lineas = [], [], []
        for nombre_caso, valor in barrido:
            argumentos = dict(todos)
            argumentos[parametro] = valor
            filtrada = reducir_ruido(region, metodo, escala=escala, **argumentos)
            filtradas.append((nombre_caso, valor, filtrada))

            nombre_capa = (
                f"{imagen.name}_{clave_csv(metodo)}_{clave_csv(nombre_caso)}"
            )
            agregar_o_actualizar_capa(
                nombre_capa, filtrada, colormap="gray",
                translate=tuple(origen_region),
                contrast_limits=tuple(imagen.contrast_limits),
                visible=(nombre_caso == "Filtrado adecuado"),
            )

            antes, despues, roi_medida = medidas_filtrado(
                region, filtrada, origen_region, roi
            )
            residuo = region - filtrada
            resultados = {
                "residuo_media": float(residuo.mean()),
                "residuo_desvio": float(residuo.std()),
                "capa_resultado": nombre_capa,
            }
            if roi_medida:
                resultados["roi_medida"] = roi_medida

            parametros = {k: argumentos[k] for k in PARAMETROS_FILTRO[metodo]}
            parametros["factor_barrido"] = valores["factor_barrido"]
            usa_sigma = "sigma_ruido" in PARAMETROS_FILTRO[metodo]
            filas.append(crear_registro(
                imagen=imagen.name,
                actividad="3 - Reducción de ruido",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                estimacion_ruido=(
                    argumentos["sigma_ruido"]
                    if usa_sigma and argumentos["sigma_ruido"] > 0 else ""
                ),
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
                observaciones=f"barrido: {nombre_caso.lower()}",
            ))
            lineas.append(
                f"{nombre_caso} ({parametro} = {valor:g}) → capa {nombre_capa}\n"
                + "\n".join(
                    f"    {clave}: {formatear_valor(antes[clave])} → "
                    f"{formatear_valor(despues[clave])}"
                    for clave in antes
                )
                + f"\n    residuo_desvio = {residuo.std():.4f}"
            )

        if valores["registrar"]:
            agregar_pendientes(filas)
        if valores["ver_figura"]:
            nombre_fig = f"Barrido de filtro: {imagen.name} - {metodo}"
            mostrar_figura(
                figura_barrido_filtro(nombre_fig, region, filtradas, parametro),
                nombre_fig,
            )

        informe.value = (
            "INFORME - BARRIDO DE REDUCCIÓN DE RUIDO\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Filtro: {metodo}   Parámetro barrido: {parametro} "
            f"(x1/{valores['factor_barrido']:g}, x1, x{valores['factor_barrido']:g})\n\n"
            + "\n\n".join(lineas)
            + (
                f"\n\nNota: {parametro} = {base:g} ya es la ventana mínima, así "
                "que el subfiltrado coincide con el filtrado adecuado. Para "
                "comparar hacia abajo, subí la ventana base."
                if menor == base else ""
            )
            + (f"\n\n{len(filas)} filas agregadas al CSV pendiente."
               if valores["registrar"] else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo hacer el barrido:\n{error}"


# ============================================================
# 6.5 RESTAURACIÓN Y ARTEFACTOS (Actividad 4)
#     a) Deconvolución (Wiener / Richardson-Lucy) con PSF sintética
#        o provista en una capa.
#     b) Filtro notch: detección de picos en Fourier, diseño del notch
#        y valores para controlar qué se removió.
# ============================================================

PARAMETROS_PSF = {
    "Gaussiana": ["sigma_psf"],
    "Disco (desenfoque)": ["radio_psf"],
    "Movimiento lineal": ["longitud_psf", "angulo_psf"],
    "Capa de napari": ["capa_psf"],
}
PARAMETROS_DECONVOLUCION = {
    "Wiener": ["balance"],
    "Wiener no supervisado": [],
    "Richardson-Lucy": ["iteraciones", "tolerancia"],
}
TODOS_PARAMETROS_DECONV = sorted(
    {n for v in PARAMETROS_PSF.values() for n in v}
    | {n for v in PARAMETROS_DECONVOLUCION.values() for n in v}
)


def medidas_restauracion(img):
    """Valores para comparar antes y después (nitidez, ruido, contraste)."""
    gy, gx = np.gradient(np.asarray(img, dtype=float))
    return {
        "energia_gradiente": float(np.mean(gx ** 2 + gy ** 2)),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
        "desvio": float(np.std(img)),
    }


def figura_deconvolucion(nombre, original, restaurada, psf, historial):
    """Original, restaurada, PSF y (RL) curvas de convergencia."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    ax = ejes[0, 1]
    ax.imshow(restaurada, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Restaurada (misma escala de grises)")

    ax = ejes[1, 0]
    im = ax.imshow(psf, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"PSF {psf.shape[0]}x{psf.shape[1]} (suma = {psf.sum():.3f})")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 1]
    if historial:
        iteraciones = [h["iteracion"] for h in historial]
        ax.plot(iteraciones, [h["residuo_rms"] for h in historial],
                color="tab:blue", label="residuo RMS")
        ax.set_xlabel("iteración")
        ax.set_ylabel("residuo RMS", color="tab:blue")
        eje2 = ax.twinx()
        eje2.semilogy(iteraciones, [h["cambio"] for h in historial],
                      color="tab:red", label="cambio relativo")
        eje2.set_ylabel("cambio relativo", color="tab:red")
        ax.set_title("Convergencia de Richardson-Lucy")
    else:
        fila = original.shape[0] // 2
        ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
        ax.plot(restaurada[fila], color="tab:red", linewidth=1.0, label="restaurada")
        ax.set_xlabel("columna (px)")
        ax.set_title(f"Perfil horizontal (fila central {fila})")
        ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


boton_nsr = PushButton(text="Copiar NSR (σ²/Var) al balance")


@boton_nsr.clicked.connect
def copiar_nsr():
    try:
        if not ultimo_sigma_ruido:
            raise ValueError("Todavía no se estimó el ruido (Calidad › Estimación de ruido).")
        imagen = widget_deconvolucion.imagen.value
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de deconvolución.")
        gris = a_escala_de_grises(imagen.data)
        nsr = ultimo_sigma_ruido["valor"] ** 2 / max(float(gris.var()), 1e-12)
        widget_deconvolucion.balance.value = nsr
        informe.value = (
            "INFORME\nBalance = σ²/Var(imagen) = "
            f"{ultimo_sigma_ruido['valor']:.3f}² / {gris.var():.1f} = {nsr:.6f}\n"
            f"(σ de {ultimo_sigma_ruido['metodo']} sobre {ultimo_sigma_ruido['imagen']})"
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo calcular el NSR:\n{error}"


@magicgui(
    call_button="Aplicar deconvolución",
    metodo={"choices": METODOS_DECONVOLUCION, "label": "Método"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a restaurar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    tipo_psf={"choices": TIPOS_PSF, "label": "PSF"},
    capa_psf={"label": "Capa con la PSF"},
    sigma_psf={"label": "σ de la PSF (px)", "min": 0.1, "max": 50.0, "step": 0.1},
    radio_psf={"label": "Radio del disco (px)", "min": 0.5, "max": 50.0, "step": 0.5},
    longitud_psf={"label": "Longitud (px)", "min": 2.0, "max": 100.0, "step": 1.0},
    angulo_psf={"label": "Ángulo (grados)", "min": -180.0, "max": 180.0, "step": 5.0},
    balance={
        "label": "Balance (regularización)",
        "min": 1e-6, "max": 10.0, "step": 0.001,
    },
    iteraciones={"label": "Iteraciones máx.", "min": 1, "max": 500},
    tolerancia={
        "label": "Parar si cambio <", "min": 0.0, "max": 1.0, "step": 0.0005,
    },
    ver_figura={"label": "Mostrar figura comparativa"},
    registrar={"label": "Agregar al CSV"},
)
def widget_deconvolucion(
    imagen: Image,
    metodo: str = "Wiener",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    tipo_psf: str = "Gaussiana",
    capa_psf: Image = None,
    sigma_psf: float = 2.0,
    radio_psf: float = 3.0,
    longitud_psf: float = 9.0,
    angulo_psf: float = 0.0,
    balance: float = 0.01,
    iteraciones: int = 20,
    tolerancia: float = 0.0,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        escala = escala_intensidad(imagen.data)

        psf = construir_psf(
            tipo_psf, sigma_psf, radio_psf, longitud_psf, angulo_psf,
            capa_psf.data if capa_psf is not None else None,
        )

        inicio = datetime.now()
        restaurada, info = restaurar(
            region, metodo=metodo, psf=psf, balance=balance,
            iteraciones=iteraciones, tolerancia=tolerancia, escala=escala,
        )
        segundos = (datetime.now() - inicio).total_seconds()

        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, restaurada, colormap="gray",
            translate=tuple(origen_region),
            contrast_limits=tuple(imagen.contrast_limits),
        )
        viewer.layers.selection.active = capa

        antes = medidas_restauracion(region)
        despues = medidas_restauracion(restaurada)

        parametros = {"psf": tipo_psf}
        parametros.update({
            k: v for k, v in dict(
                sigma_psf=sigma_psf, radio_psf=radio_psf,
                longitud_psf=longitud_psf, angulo_psf=angulo_psf,
            ).items() if k in PARAMETROS_PSF[tipo_psf]
        })
        if tipo_psf == "Capa de napari":
            parametros["capa_psf"] = capa_psf.name
        parametros["psf_tamano"] = f"{psf.shape[0]}x{psf.shape[1]}"
        for clave in PARAMETROS_DECONVOLUCION[metodo]:
            parametros[clave] = {"balance": balance, "iteraciones": iteraciones,
                                 "tolerancia": tolerancia}[clave]

        historial = info.get("historial") or []
        resultados = {"tiempo_s": segundos, "capa_resultado": nombre_capa}
        detalle = []
        if historial:
            ultimo = historial[-1]
            resultados.update(
                iteraciones_realizadas=ultimo["iteracion"],
                residuo_rms_final=ultimo["residuo_rms"],
                cambio_final=ultimo["cambio"],
            )
            detalle.append(
                f"Iteraciones realizadas: {ultimo['iteracion']} de {iteraciones}"
                + ("  (cortó por tolerancia)" if tolerancia > 0
                   and ultimo["cambio"] < tolerancia else "")
            )
            detalle.append(
                f"Residuo RMS: {historial[0]['residuo_rms']:.4f} → "
                f"{ultimo['residuo_rms']:.4f}   "
                f"cambio relativo final: {ultimo['cambio']:.5f}"
            )
        elif info.get("balance_usado") is not None:
            resultados["balance_usado"] = info["balance_usado"]
        else:
            resultados["balance_usado"] = "estimado (no supervisado)"

        if ver_figura:
            nombre_fig = f"Deconvolución: {imagen.name} - {metodo}"
            fig = figura_deconvolucion(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, restaurada, psf, historial,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="4 - Restauración (deconvolución)",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
            )])

        informe.value = (
            "INFORME - DECONVOLUCIÓN\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s   Capa creada: {nombre_capa}\n"
            + ("".join(f"{d}\n" for d in detalle))
            + "\nANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues[clave])}"
                for clave in antes
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo deconvolucionar:\n{error}"


def actualizar_parametros_deconv(event=None):
    """Muestra solo los parámetros de la PSF y del método elegidos."""
    visibles = set(PARAMETROS_PSF[widget_deconvolucion.tipo_psf.value])
    visibles |= set(PARAMETROS_DECONVOLUCION[widget_deconvolucion.metodo.value])
    for nombre in TODOS_PARAMETROS_DECONV:
        getattr(widget_deconvolucion, nombre).visible = nombre in visibles


widget_deconvolucion.metodo.changed.connect(actualizar_parametros_deconv)
widget_deconvolucion.tipo_psf.changed.connect(actualizar_parametros_deconv)


# ------------------------------------------------------------
# 6.5.b  Artefactos periódicos: picos de Fourier y filtro notch
# ------------------------------------------------------------

# Picos detectados en la última corrida (se eligen por número)
picos_detectados = {"picos": [], "forma": None, "imagen": "", "region": ""}


def texto_picos(picos):
    return "\n".join(
        f"  {i + 1}. período = {p['periodo_px']:.2f} px   "
        f"frecuencia = {p['frecuencia']:.4f} ciclos/px   "
        f"ángulo = {p['angulo_grados']:+.1f}°   "
        f"prominencia = {p['prominencia']:.2f}   "
        f"(fila {p['fila']}, columna {p['columna']})"
        for i, p in enumerate(picos)
    )


def figura_notch(nombre, original, filtrada, log_mag, mascara, picos, radio):
    """Espectro con los notches marcados, filtrada, residuo y máscara."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    alto, ancho = original.shape

    ax = ejes[0, 0]
    ax.imshow(log_mag, cmap="gray")
    centro_f, centro_c = alto // 2, ancho // 2
    for i, p in enumerate(picos):
        for fila, columna in (
            (p["fila"], p["columna"]),
            (2 * centro_f - p["fila"], 2 * centro_c - p["columna"]),
        ):
            ax.add_patch(
                Circle((columna, fila), radio, fill=False, color="red", linewidth=1)
            )
        ax.annotate(str(i + 1), (p["columna"], p["fila"]), color="red", fontsize=8)
    ax.set_title("log |FFT| con los notches aplicados")

    ax = ejes[0, 1]
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Máscara del filtro (0 = atenuado)")

    ax = ejes[1, 0]
    vmin, vmax = np.percentile(original, [0.5, 99.5])
    ax.imshow(filtrada, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Imagen filtrada")

    ax = ejes[1, 1]
    residuo = original - filtrada
    lim = max(np.percentile(np.abs(residuo), 99.5), 1e-9)
    im = ax.imshow(residuo, cmap="gray", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Residuo = original - filtrada (lo que se quitó)")

    for ax in ejes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(nombre)
    return fig


boton_detectar_picos = PushButton(text="Detectar picos en Fourier")


@boton_detectar_picos.clicked.connect
def detectar_picos():
    try:
        imagen = widget_notch.imagen.value
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de artefactos.")
        region, _, descripcion = region_en_gris(
            imagen, widget_notch.modo.value, widget_notch.roi.value,
            widget_notch.excluir_marco.value,
        )
        picos, log_mag = detectar_picos_fft(
            region, cantidad=widget_notch.cantidad_picos.value
        )
        picos_detectados.update(
            picos=picos, forma=region.shape, imagen=imagen.name, region=descripcion
        )

        informe.value = (
            "INFORME - PICOS DE FOURIER\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"{len(picos)} pico(s) más prominentes (cada uno con su simétrico):\n"
            + texto_picos(picos)
            + "\n\nPara filtrar, escribí sus números en 'Picos a filtrar'."
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudieron detectar picos:\n{error}"


@magicgui(
    call_button="Aplicar filtro notch",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a filtrar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    cantidad_picos={"label": "Picos a detectar", "min": 1, "max": 20},
    picos_elegidos={"label": "Picos a filtrar (ej: 1,2)"},
    radio_notch={"label": "Radio del notch (px)", "min": 0.5, "max": 100.0, "step": 0.5},
    tipo_notch={"choices": TIPOS_NOTCH, "label": "Tipo de notch"},
    orden_butterworth={"label": "Orden (Butterworth)", "min": 1, "max": 10},
    ver_figura={"label": "Mostrar figura comparativa"},
    registrar={"label": "Agregar al CSV"},
)
def widget_notch(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    cantidad_picos: int = 6,
    picos_elegidos: str = "1",
    radio_notch: float = 5.0,
    tipo_notch: str = "Gaussiano",
    orden_butterworth: int = 2,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )

        if not picos_detectados["picos"]:
            raise ValueError("Primero tocá 'Detectar picos en Fourier'.")
        if picos_detectados["forma"] != region.shape:
            raise ValueError(
                "Los picos se detectaron sobre una región de otro tamaño "
                f"({picos_detectados['forma']}): detectalos de nuevo."
            )

        try:
            numeros = [int(n) for n in picos_elegidos.replace(" ", "").split(",") if n]
        except ValueError:
            raise ValueError("Escribí los picos como números separados por comas.")
        disponibles = picos_detectados["picos"]
        if not numeros or any(n < 1 or n > len(disponibles) for n in numeros):
            raise ValueError(f"Elegí números entre 1 y {len(disponibles)}.")
        picos = [disponibles[n - 1] for n in numeros]

        mascara = mascara_notch(
            region.shape, picos, radio_notch, tipo_notch, orden_butterworth
        )
        filtrada, espectro = aplicar_notch(region, mascara)
        log_mag = np.log1p(np.abs(espectro))

        nombre_capa = f"{imagen.name}_notch"
        capa = agregar_o_actualizar_capa(
            nombre_capa, filtrada, colormap="gray",
            translate=tuple(origen_region),
            contrast_limits=tuple(imagen.contrast_limits),
        )
        viewer.layers.selection.active = capa

        control = verificar_notch(region, filtrada)
        antes = medidas_restauracion(region)
        despues = medidas_restauracion(filtrada)
        picos_despues, _ = detectar_picos_fft(filtrada, cantidad=1)
        if picos_despues:
            despues["pico_fft_prominencia"] = picos_despues[0]["prominencia"]
        antes["pico_fft_prominencia"] = max(p["prominencia"] for p in picos)

        parametros = {
            "picos": ",".join(str(n) for n in numeros),
            "radio": radio_notch,
            "tipo": tipo_notch,
        }
        if tipo_notch == "Butterworth":
            parametros["orden"] = orden_butterworth
        for i, p in zip(numeros, picos):
            parametros[f"pico{i}_periodo_px"] = p["periodo_px"]
            parametros[f"pico{i}_angulo"] = p["angulo_grados"]

        if ver_figura:
            nombre_fig = f"Notch: {imagen.name}"
            fig = figura_notch(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, filtrada, log_mag, mascara, picos, radio_notch,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        resultados = dict(control)
        resultados["capa_resultado"] = nombre_capa
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="4 - Artefacto periódico (notch)",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=f"Notch {tipo_notch}",
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
            )])

        informe.value = (
            "INFORME - FILTRO NOTCH\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Notch {tipo_notch}, radio {radio_notch:g} px, picos {parametros['picos']}\n"
            + texto_picos(picos)
            + f"\n\nCapa creada: {nombre_capa}\n"
            "\nANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues.get(clave, float('nan')))}"
                for clave in antes
            )
            + "\n\nCONTROL DE LO REMOVIDO\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}"
                for clave, valor in control.items()
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo aplicar el notch:\n{error}"


# ============================================================
# 6.6 CORRECCIÓN DE FONDO (Actividad 5)
#     Cinco métodos con sus parámetros, más un barrido que genera de
#     una vez subcorrección / corrección adecuada / sobrecorrección.
# ============================================================

PARAMETROS_FONDO = {
    "Dark-field / flat-field": ["capa_flat", "capa_dark"],
    "Kernel de gran escala": [
        "sigma_fondo", "tipo_kernel", "modo_correccion", "conservar_nivel",
    ],
    "Rolling ball": ["radio_bola", "conservar_nivel"],
    "White top-hat": ["radio_tophat", "conservar_nivel"],
    "Homomórfica": ["gamma_bajo", "gamma_alto", "corte"],
}
TODOS_PARAMETROS_FONDO = sorted(
    {n for v in PARAMETROS_FONDO.values() for n in v}
)

# Parámetro que se barre en la comparación y en qué sentido actúa:
#   "suaviza"  -> más grande = corrección más suave (subcorrige)
#   "fortalece" -> más grande = corrección más fuerte (sobrecorrige)
PARAMETRO_BARRIDO = {
    "Kernel de gran escala": ("sigma_fondo", "suaviza"),
    "Rolling ball": ("radio_bola", "suaviza"),
    "White top-hat": ("radio_tophat", "suaviza"),
    "Homomórfica": ("corte", "fortalece"),
}


def figura_fondo(nombre, original, corregida, fondo):
    """Original, fondo estimado, corregida y perfil de la fila central."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    ax = ejes[0, 1]
    im = ax.imshow(fondo, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Fondo estimado")

    ax = ejes[1, 0]
    ax.imshow(corregida, cmap="gray")
    ax.set_title("Corregida (contraste propio)")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 1]
    fila = original.shape[0] // 2
    ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
    ax.plot(fondo[fila], color="tab:green", linewidth=1.0, label="fondo")
    ax.plot(corregida[fila], color="tab:red", linewidth=1.0, label="corregida")
    ax.set_xlabel("columna (px)")
    ax.set_title(f"Perfil horizontal (fila central {fila})")
    ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


def figura_barrido(nombre, original, corregidas, parametro):
    """
    Compara subcorrección / adecuada / sobrecorrección.
    `corregidas` es [(etiqueta, valor, imagen), ...].
    """
    fig = Figure(figsize=(15, 9), layout="constrained")
    ejes = fig.subplots(2, 3)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    for ax, (etiqueta, valor, imagen) in zip(
        [ejes[0, 1], ejes[0, 2], ejes[1, 0]], corregidas
    ):
        ax.imshow(imagen, cmap="gray")
        ax.set_title(f"{etiqueta} ({parametro} = {valor:g})")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[0, 2], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    fila = original.shape[0] // 2
    ax = ejes[1, 1]
    for etiqueta, valor, imagen in corregidas:
        suave = ndi.gaussian_filter(imagen, sigma=max(3, min(imagen.shape) / 20))
        ax.plot(suave[fila], linewidth=1.0, label=f"{etiqueta} ({valor:g})")
    suave_original = ndi.gaussian_filter(
        original, sigma=max(3, min(original.shape) / 20)
    )
    ax.plot(suave_original[fila], color="0.6", linewidth=0.8, label="original")
    ax.set_xlabel("columna (px)")
    ax.set_title("Fondo residual (imagen muy suavizada, fila central)")
    ax.legend(fontsize=7)

    ax = ejes[1, 2]
    bins = 80
    ax.hist(original.ravel(), bins=bins, density=True, color="0.6",
            histtype="step", label="original")
    for etiqueta, valor, imagen in corregidas:
        ax.hist(imagen.ravel(), bins=bins, density=True, histtype="step",
                linewidth=1.0, label=etiqueta)
    ax.set_yscale("log")
    ax.set_title("Histogramas")
    ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


def rois_en_arreglo(arreglo, origen_arreglo, roi_layer):
    """
    (indice, píxeles) de las ROIs que caen enteras dentro de `arreglo`,
    cuyo píxel [0, 0] está en `origen_arreglo` (coordenadas del mundo).
    Permite medir las mismas ROIs antes y después de procesar.
    """
    if roi_layer is None or len(roi_layer.data) == 0:
        return []

    desplazamiento = np.asarray(origen_arreglo, dtype=float)
    alto, ancho = arreglo.shape
    salida = []
    for indice in range(len(roi_layer.data)):
        vertices = np.asarray(roi_layer.data[indice], dtype=float) - desplazamiento
        if (
            vertices.min() < 0
            or vertices[:, 0].max() > alto
            or vertices[:, 1].max() > ancho
        ):
            continue
        filas, columnas = polygon(vertices[:, 0], vertices[:, 1], shape=arreglo.shape)
        if filas.size > 1:
            salida.append((indice, arreglo[filas, columnas]))
    return salida


def _datos_fondo(imagen, modo, roi, excluir_marco):
    """Región a corregir, ROIs que caen adentro y descripción."""
    region, origen_region, descripcion = region_en_gris(
        imagen, modo, roi, excluir_marco
    )
    return region, origen_region, descripcion, rois_en_arreglo(
        region, origen_region, roi
    )


@magicgui(
    call_button="Corregir fondo",
    metodo={"choices": METODOS_FONDO, "label": "Método"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a corregir",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    capa_flat={"label": "Capa flat-field"},
    capa_dark={"label": "Capa dark-field (opcional)"},
    sigma_fondo={"label": "Tamaño del kernel (px)", "min": 1.0, "max": 500.0, "step": 5.0},
    tipo_kernel={"choices": ["Gaussiano", "Mediana"], "label": "Kernel"},
    modo_correccion={"choices": ["Resta", "División"], "label": "Fondo aditivo/multiplicativo"},
    radio_bola={"label": "Radio de la bola (px)", "min": 1.0, "max": 500.0, "step": 5.0},
    radio_tophat={"label": "Radio del top-hat (px)", "min": 1.0, "max": 300.0, "step": 5.0},
    gamma_bajo={"label": "γ bajas frecuencias", "min": 0.0, "max": 2.0, "step": 0.05},
    gamma_alto={"label": "γ altas frecuencias", "min": 0.1, "max": 5.0, "step": 0.05},
    corte={"label": "Corte D0 (frac. Nyquist)", "min": 0.005, "max": 1.0, "step": 0.005},
    conservar_nivel={"label": "Conservar nivel medio"},
    factor_barrido={"label": "Factor del barrido", "min": 1.1, "max": 10.0, "step": 0.1},
    ver_figura={"label": "Mostrar figura"},
    registrar={"label": "Agregar al CSV"},
)
def widget_fondo(
    imagen: Image,
    metodo: str = "Rolling ball",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    capa_flat: Image = None,
    capa_dark: Image = None,
    sigma_fondo: float = 50.0,
    tipo_kernel: str = "Gaussiano",
    modo_correccion: str = "Resta",
    radio_bola: float = 50.0,
    radio_tophat: float = 25.0,
    gamma_bajo: float = 0.5,
    gamma_alto: float = 1.5,
    corte: float = 0.05,
    conservar_nivel: bool = True,
    factor_barrido: float = 2.0,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion, lista_rois = _datos_fondo(
            imagen, modo, roi, excluir_marco
        )

        inicio = datetime.now()
        corregida, fondo = corregir_fondo(
            region, metodo=metodo,
            flat=capa_flat.data if capa_flat is not None else None,
            dark=capa_dark.data if capa_dark is not None else None,
            sigma_fondo=sigma_fondo, tipo_kernel=tipo_kernel,
            modo_correccion=modo_correccion, radio_bola=radio_bola,
            radio_tophat=radio_tophat, gamma_bajo=gamma_bajo,
            gamma_alto=gamma_alto, corte=corte, conservar_nivel=conservar_nivel,
        )
        segundos = (datetime.now() - inicio).total_seconds()

        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, corregida, colormap="gray",
            translate=tuple(origen_region),
        )
        agregar_o_actualizar_capa(
            f"{nombre_capa}_fondo", fondo, colormap="viridis",
            translate=tuple(origen_region), visible=False,
        )
        viewer.layers.selection.active = capa

        antes = medidas_fondo(region, lista_rois)
        despues = medidas_fondo(
            corregida, rois_en_arreglo(corregida, origen_region, roi)
        )

        parametros = {
            clave: valor for clave, valor in dict(
                capa_flat=capa_flat.name if capa_flat is not None else "",
                capa_dark=capa_dark.name if capa_dark is not None else "",
                sigma_fondo=sigma_fondo, tipo_kernel=tipo_kernel,
                modo_correccion=modo_correccion, radio_bola=radio_bola,
                radio_tophat=radio_tophat, gamma_bajo=gamma_bajo,
                gamma_alto=gamma_alto, corte=corte,
                conservar_nivel=conservar_nivel,
            ).items() if clave in PARAMETROS_FONDO[metodo]
        }

        if ver_figura:
            nombre_fig = f"Fondo: {imagen.name} - {metodo}"
            fig = figura_fondo(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, corregida, fondo,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="5 - Corrección de fondo",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados={"tiempo_s": segundos, "capa_resultado": nombre_capa},
            )])

        informe.value = (
            "INFORME - CORRECCIÓN DE FONDO\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s   Capas creadas: {nombre_capa} "
            f"(+ _fondo)\n\n"
            "ANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues.get(clave, float('nan')))}"
                for clave in antes
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo corregir el fondo:\n{error}"


boton_barrido_fondo = PushButton(
    text="Crear sub / adecuada / sobrecorrección"
)


@boton_barrido_fondo.clicked.connect
def barrido_fondo():
    """
    Corre el método tres veces: con el parámetro del panel y con ese
    parámetro dividido y multiplicado por el factor. Crea las tres capas,
    una figura comparativa y tres filas del CSV.
    """
    try:
        valores = widget_fondo.asdict()
        metodo = valores["metodo"]
        imagen = valores["imagen"]
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de corrección de fondo.")
        if metodo not in PARAMETRO_BARRIDO:
            raise ValueError(
                "El barrido no aplica a dark/flat-field: la corrección queda "
                "definida por los campos de referencia, no por un parámetro."
            )

        parametro, sentido = PARAMETRO_BARRIDO[metodo]
        base = valores[parametro]
        factor = valores["factor_barrido"]
        menor, mayor = base / factor, base * factor
        if sentido == "suaviza":
            barrido = [
                ("Sobrecorrección", menor),
                ("Corrección adecuada", base),
                ("Subcorrección", mayor),
            ]
        else:
            barrido = [
                ("Subcorrección", menor),
                ("Corrección adecuada", base),
                ("Sobrecorrección", mayor),
            ]

        region, origen_region, descripcion, lista_rois = _datos_fondo(
            imagen, valores["modo"], valores["roi"], valores["excluir_marco"]
        )
        antes = medidas_fondo(region, lista_rois)

        comunes = dict(
            flat=valores["capa_flat"].data if valores["capa_flat"] is not None else None,
            dark=valores["capa_dark"].data if valores["capa_dark"] is not None else None,
            sigma_fondo=valores["sigma_fondo"], tipo_kernel=valores["tipo_kernel"],
            modo_correccion=valores["modo_correccion"], radio_bola=valores["radio_bola"],
            radio_tophat=valores["radio_tophat"], gamma_bajo=valores["gamma_bajo"],
            gamma_alto=valores["gamma_alto"], corte=valores["corte"],
            conservar_nivel=valores["conservar_nivel"],
        )

        corregidas = []
        filas = []
        lineas = []
        region_fila, etiqueta = region_y_etiqueta(valores["modo"], valores["roi"])
        for nombre_caso, valor in barrido:
            argumentos = dict(comunes)
            argumentos[parametro] = valor
            corregida, _ = corregir_fondo(region, metodo=metodo, **argumentos)
            corregidas.append((nombre_caso, valor, corregida))

            nombre_capa = f"{imagen.name}_{clave_csv(metodo)}_{clave_csv(nombre_caso)}"
            agregar_o_actualizar_capa(
                nombre_capa, corregida, colormap="gray",
                translate=tuple(origen_region),
                visible=(nombre_caso == "Corrección adecuada"),
            )
            despues = medidas_fondo(
                corregida, rois_en_arreglo(corregida, origen_region, valores["roi"])
            )
            lineas.append(
                f"{nombre_caso} ({parametro} = {valor:g}) → capa {nombre_capa}\n"
                + "\n".join(
                    f"    {clave}: {formatear_valor(antes[clave])} → "
                    f"{formatear_valor(despues.get(clave, float('nan')))}"
                    for clave in antes
                )
            )
            filas.append(crear_registro(
                imagen=imagen.name,
                actividad="5 - Corrección de fondo",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros={parametro: valor, "factor_barrido": factor},
                metrica_antes=antes,
                metrica_despues=despues,
                resultados={"capa_resultado": nombre_capa},
                observaciones=f"barrido: {nombre_caso.lower()}",
            ))

        if valores["registrar"]:
            agregar_pendientes(filas)
        if valores["ver_figura"]:
            nombre_fig = f"Barrido de fondo: {imagen.name} - {metodo}"
            mostrar_figura(
                figura_barrido(nombre_fig, region, corregidas, parametro), nombre_fig
            )

        informe.value = (
            "INFORME - BARRIDO DE CORRECCIÓN DE FONDO\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}   Parámetro barrido: {parametro} "
            f"(x1/{factor:g}, x1, x{factor:g})\n\n"
            + "\n\n".join(lineas)
            + (f"\n\n{len(filas)} filas agregadas al CSV pendiente."
               if valores["registrar"] else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo hacer el barrido:\n{error}"


def actualizar_parametros_fondo(event=None):
    """Muestra solo los parámetros del método de fondo elegido."""
    visibles = PARAMETROS_FONDO[widget_fondo.metodo.value]
    for nombre in TODOS_PARAMETROS_FONDO:
        getattr(widget_fondo, nombre).visible = nombre in visibles
    widget_fondo.factor_barrido.visible = (
        widget_fondo.metodo.value in PARAMETRO_BARRIDO
    )


widget_fondo.metodo.changed.connect(actualizar_parametros_fondo)


# ============================================================
# 6.7 REALCE (Actividad 6)
#     Transformaciones de intensidad y filtros de realce, con los
#     valores que permiten ver si se amplificó ruido o se perdió
#     información (la lectura queda para el informe).
# ============================================================

PARAMETROS_REALCE = {
    "Gamma": ["gamma"],
    "Logarítmica": ["factor_log"],
    "Sigmoidal": ["centro", "ganancia"],
    "CLAHE": ["tamano_mosaico", "limite_contraste"],
    "Pasa-altos": ["sigma_altos", "conservar_nivel"],
    "Pasa-banda": ["sigma_bajo", "sigma_alto", "conservar_nivel"],
    "Unsharp masking": ["sigma_unsharp", "cantidad", "umbral"],
}
TODOS_PARAMETROS_REALCE = sorted(
    {n for v in PARAMETROS_REALCE.values() for n in v}
)


def figura_realce(nombre, original, realzada, limites):
    """Original, realzada, histogramas y curva de transformación."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=limites[0], vmax=limites[1])
    ax.set_title("Original")

    ax = ejes[0, 1]
    ax.imshow(realzada, cmap="gray")
    ax.set_title("Realzada (contraste propio)")

    for ax in (ejes[0, 0], ejes[0, 1]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 0]
    ax.hist(original.ravel(), bins=128, density=True, color="0.6",
            histtype="step", label="original")
    ax.hist(realzada.ravel(), bins=128, density=True, color="tab:red",
            histtype="step", label="realzada")
    ax.set_yscale("log")
    ax.set_title("Histogramas")
    ax.set_xlabel("intensidad")
    ax.legend(fontsize=7)

    # Relación entrada-salida: para las transformaciones puntuales es la
    # curva del método; para los filtros muestra cuánta dispersión agrega.
    ax = ejes[1, 1]
    muestra = np.random.default_rng(0).choice(original.size, size=min(20000, original.size),
                                              replace=False)
    ax.plot(original.ravel()[muestra], realzada.ravel()[muestra], ".",
            markersize=1, alpha=0.3)
    ax.set_xlabel("intensidad original")
    ax.set_ylabel("intensidad realzada")
    ax.set_title("Entrada vs. salida")

    fig.suptitle(nombre)
    return fig


@magicgui(
    call_button="Aplicar realce",
    metodo={"choices": METODOS_REALCE, "label": "Método"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a realzar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    gamma={"label": "γ", "min": 0.05, "max": 5.0, "step": 0.05},
    factor_log={"label": "Factor del log", "min": 0.1, "max": 500.0, "step": 1.0},
    centro={"label": "Centro (frac. del rango)", "min": 0.0, "max": 1.0, "step": 0.05},
    ganancia={"label": "Ganancia (pendiente)", "min": 0.5, "max": 50.0, "step": 0.5},
    tamano_mosaico={"label": "Mosaico CLAHE (px)", "min": 3, "max": 512},
    limite_contraste={"label": "Límite de contraste", "min": 0.001, "max": 1.0, "step": 0.005},
    sigma_altos={"label": "σ del pasa-altos (px)", "min": 0.5, "max": 100.0, "step": 0.5},
    sigma_bajo={"label": "σ bajo (px)", "min": 0.3, "max": 50.0, "step": 0.1},
    sigma_alto={"label": "σ alto (px)", "min": 0.5, "max": 200.0, "step": 0.5},
    sigma_unsharp={"label": "σ del unsharp (px)", "min": 0.3, "max": 50.0, "step": 0.1},
    cantidad={"label": "Cantidad", "min": 0.1, "max": 10.0, "step": 0.1},
    umbral={"label": "Umbral de detalle", "min": 0.0, "max": 100.0, "step": 0.5},
    conservar_nivel={"label": "Conservar nivel medio"},
    ver_figura={"label": "Mostrar figura"},
    registrar={"label": "Agregar al CSV"},
)
def widget_realce(
    imagen: Image,
    metodo: str = "CLAHE",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    gamma: float = 0.7,
    factor_log: float = 10.0,
    centro: float = 0.5,
    ganancia: float = 10.0,
    tamano_mosaico: int = 64,
    limite_contraste: float = 0.01,
    sigma_altos: float = 3.0,
    sigma_bajo: float = 1.0,
    sigma_alto: float = 8.0,
    sigma_unsharp: float = 2.0,
    cantidad: float = 1.0,
    umbral: float = 0.0,
    conservar_nivel: bool = True,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        rois_antes = rois_en_arreglo(region, origen_region, roi)

        todos = dict(
            gamma=gamma, factor_log=factor_log, centro=centro, ganancia=ganancia,
            tamano_mosaico=tamano_mosaico, limite_contraste=limite_contraste,
            sigma_altos=sigma_altos, sigma_bajo=sigma_bajo, sigma_alto=sigma_alto,
            sigma_unsharp=sigma_unsharp, cantidad=cantidad, umbral=umbral,
            conservar_nivel=conservar_nivel,
        )
        parametros = {k: todos[k] for k in PARAMETROS_REALCE[metodo]}

        inicio = datetime.now()
        realzada = realzar(region, metodo=metodo, **todos)
        segundos = (datetime.now() - inicio).total_seconds()

        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, realzada, colormap="gray",
            translate=tuple(origen_region),
        )
        viewer.layers.selection.active = capa

        limites = (float(region.min()), float(region.max()))
        antes = medidas_realce(region, rois_antes, limites)
        despues = medidas_realce(
            realzada, rois_en_arreglo(realzada, origen_region, roi), limites
        )
        razones = comparar_realce(region, realzada)

        if ver_figura:
            nombre_fig = f"Realce: {imagen.name} - {metodo}"
            fig = figura_realce(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, realzada, limites,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="6 - Realce",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados={**razones, "tiempo_s": segundos,
                            "capa_resultado": nombre_capa},
            )])

        informe.value = (
            "INFORME - REALCE\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s   Capa creada: {nombre_capa}\n\n"
            "ANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues.get(clave, float('nan')))}"
                for clave in antes
            )
            + "\n\nRAZONES DESPUÉS/ANTES\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}"
                for clave, valor in razones.items()
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo realzar:\n{error}"


def actualizar_parametros_realce(event=None):
    """Muestra solo los parámetros del método de realce elegido."""
    visibles = PARAMETROS_REALCE[widget_realce.metodo.value]
    for nombre in TODOS_PARAMETROS_REALCE:
        getattr(widget_realce, nombre).visible = nombre in visibles


widget_realce.metodo.changed.connect(actualizar_parametros_realce)


# ============================================================
# 6.8 EVALUACIÓN (Actividad 7)
#     Métricas sin referencia (SNR, CNR, uniformidad) sobre las ROIs y,
#     si se elige una capa de referencia, también MSE / PSNR / SSIM.
# ============================================================

def figura_metricas(nombre, imagen, referencia, mapa_ssim, sin_referencia):
    """Imagen (y referencia), diferencia, mapa SSIM y barras de SNR/CNR."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(imagen, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(imagen, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Imagen evaluada")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = ejes[0, 1]
    if referencia is not None:
        ax.imshow(referencia, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("Referencia")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(imagen.ravel(), bins=128, color="0.5")
        ax.set_yscale("log")
        ax.set_title("Histograma de la imagen evaluada")

    ax = ejes[1, 0]
    if referencia is not None:
        diferencia = imagen - referencia
        lim = max(np.percentile(np.abs(diferencia), 99), 1e-9)
        im = ax.imshow(diferencia, cmap="gray", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Diferencia (evaluada - referencia)")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")

    ax = ejes[1, 1]
    if mapa_ssim is not None:
        im = ax.imshow(mapa_ssim, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Mapa SSIM (1 = igual a la referencia)")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        claves = [
            c for c in sin_referencia
            if c.startswith(("snr_roi", "cnr_roi", "uniformidad_roi"))
        ]
        if claves:
            valores = [sin_referencia[c] for c in claves]
            ax.barh(range(len(claves)), valores, color="0.5")
            ax.set_yticks(range(len(claves)))
            ax.set_yticklabels(claves, fontsize=7)
            ax.invert_yaxis()
            ax.set_title("Métricas por ROI")
        else:
            ax.text(0.5, 0.5, "Dibujá ROIs para calcular\nSNR, CNR y uniformidad",
                    ha="center", va="center")
            ax.axis("off")

    fig.suptitle(nombre)
    return fig


@magicgui(
    call_button="Calcular métricas",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a evaluar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    referencia={"label": "Referencia (opcional)"},
    ver_figura={"label": "Mostrar figura"},
    registrar={"label": "Agregar al CSV"},
)
def widget_metricas(
    imagen: Image,
    referencia: Image = None,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        lista_rois = rois_en_arreglo(region, origen_region, roi)

        # Misma región en la referencia (se alinea por las coordenadas del mundo)
        region_referencia = None
        if referencia is not None:
            if referencia is imagen:
                raise ValueError("La referencia es la misma capa que la evaluada.")
            gris_referencia = a_escala_de_grises(referencia.data)
            f0, c0 = np.round(origen_region - origen_capa(referencia)).astype(int)
            alto, ancho = region.shape
            if (
                f0 < 0 or c0 < 0
                or f0 + alto > gris_referencia.shape[0]
                or c0 + ancho > gris_referencia.shape[1]
            ):
                raise ValueError(
                    f"La región no entra en la referencia '{referencia.name}' "
                    f"({gris_referencia.shape})."
                )
            region_referencia = gris_referencia[f0:f0 + alto, c0:c0 + ancho]

        # El rango dinámico sale de la REFERENCIA: así el PSNR y el SSIM
        # de distintas versiones procesadas son comparables entre sí.
        sin_referencia, con_referencia, mapa = evaluar(
            region,
            referencia=region_referencia,
            lista_rois=lista_rois,
            rango_datos=escala_intensidad(
                referencia.data if referencia is not None else imagen.data
            ),
        )

        if ver_figura:
            nombre_fig = f"Métricas: {imagen.name}"
            fig = figura_metricas(
                nombre_fig + (f" vs {referencia.name}" if referencia else ""),
                region, region_referencia, mapa, sin_referencia,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            metricas_referencia = {}
            if region_referencia is not None:
                metricas_referencia = metricas_sin_referencia(
                    region_referencia,
                    rois_en_arreglo(region_referencia, origen_region, roi),
                )
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="7 - Evaluación",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=(
                    f"Con y sin referencia (vs {referencia.name})"
                    if referencia is not None else "Sin referencia"
                ),
                parametros={"sin_marco": excluir_marco,
                            "rois": len(lista_rois)},
                metrica_antes=metricas_referencia,
                metrica_despues=sin_referencia,
                resultados=con_referencia,
            )])

        informe.value = (
            "INFORME - EVALUACIÓN\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"ROIs usadas: {len(lista_rois)}\n\n"
            "SIN REFERENCIA\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}"
                for clave, valor in sin_referencia.items()
            )
            + (
                f"\n\nCON REFERENCIA (vs {referencia.name})\n"
                + "\n".join(
                    f"  {clave} = {formatear_valor(valor)}"
                    for clave, valor in con_referencia.items()
                )
                if con_referencia else
                "\n\n(Sin capa de referencia: solo métricas sin referencia.)"
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudieron calcular las métricas:\n{error}"


# ============================================================
# 7. REGISTRO DEL PIPELINE (Actividad 7 - documentación)
#    Carga un paso (diagnóstico/método/parámetros/métricas) al mismo CSV
#    que las mediciones. Los campos los completa el usuario (la hipótesis
#    sale del análisis propio); el CSV se guarda con el botón.
# ============================================================

@magicgui(
    call_button="Cargar paso al pipeline",
    diagnostico={
        "choices": [
            "Ruido",
            "Fondo / inhomogeneidad",
            "Artefacto estructurado",
            "Desenfoque / pérdida de resolución",
            "Sin degradación (referencia)",
        ],
        "label": "Diagnóstico (hipótesis)",
    },
)
def widget_pipeline(
    imagen: Image,
    diagnostico: str = "Ruido",
    metodo: str = "",
    parametros: str = "",
    estimacion_ruido: str = "",
    metrica_antes: str = "",
    metrica_despues: str = "",
    observaciones: str = "",
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    registro = crear_registro(
        imagen=imagen.name,
        actividad="Pipeline",
        diagnostico=diagnostico,
        metodo=metodo,
        parametros=parametros,
        estimacion_ruido=estimacion_ruido,
        metrica_antes=metrica_antes,
        metrica_despues=metrica_despues,
        observaciones=observaciones,
    )
    agregar_pendientes([registro])

    informe.value = (
        "INFORME\n"
        f"Paso preparado para: {imagen.name}\n"
        f"Diagnóstico: {diagnostico}\n"
        f"Método: {metodo or '(vacío)'}"
    )


# ============================================================
# 8. CONTROLES DEL CSV
# ============================================================

ruta_csv = LineEdit(
    value="resultados/tp1/resultados_tp1.csv",
    label="Archivo CSV",
)

boton_guardar_csv = PushButton(text="Guardar en CSV")
boton_descartar = PushButton(text="Descartar filas pendientes")


@boton_guardar_csv.clicked.connect
def guardar_resultados_actuales():
    try:
        cantidad = len(registros_pendientes)
        ruta = guardar_registros_csv(ruta_csv.value, registros_pendientes)
        registros_pendientes.clear()
        estado_csv.value = (
            "CSV\n"
            f"{cantidad} fila(s) agregada(s).\n"
            f"Archivo: {ruta}"
        )
    except Exception as error:
        estado_csv.value = f"CSV\nNo se pudo guardar:\n{error}"


@boton_descartar.clicked.connect
def descartar_pendientes():
    cantidad = len(registros_pendientes)
    registros_pendientes.clear()
    estado_csv.value = f"CSV\n{cantidad} fila(s) descartada(s)."


# ============================================================
# 9. PANEL LATERAL
# ============================================================

def titulo(texto):
    """Encabezado de sección del panel."""
    etiqueta = Label(value=texto)
    etiqueta.native.setStyleSheet("font-weight: bold; margin-top: 8px;")
    return etiqueta


# Bloques funcionales. La ventana del TP3 los reparte en solapas según la
# tarea (calidad, preprocesamiento, resultados...), no según el TP de origen.
# labels=False: sin la columna con los nombres internos de cada bloque
# (widget_gris, widget_pipeline...), que ensanchaba mucho el panel.
def _bloque(*widgets):
    return Container(labels=False, widgets=list(widgets))


secciones = {
    "rois": _bloque(boton_crear_roi, etiqueta_nueva, boton_etiquetar),
    "gris": _bloque(widget_gris),
    "medicion": _bloque(widget_medicion),
    "diagnostico": _bloque(boton_crear_perfil, widget_diagnostico),
    "ruido": _bloque(widget_ruido),
    "filtro": _bloque(boton_usar_sigma, widget_filtro, boton_barrido_filtro),
    "deconvolucion": _bloque(boton_nsr, widget_deconvolucion),
    "notch": _bloque(boton_detectar_picos, widget_notch),
    "fondo": _bloque(widget_fondo, boton_barrido_fondo),
    "realce": _bloque(widget_realce),
    "metricas": _bloque(widget_metricas),
    "registro": _bloque(
        widget_pipeline,
        titulo("Archivo CSV"),
        ruta_csv,
        boton_guardar_csv,
        boton_descartar,
        estado_csv,
    ),
}

# El lanzador del TP3 monta cada `secciones[clave].native` en su solapa y
# acopla `informe.native` abajo del visor.


def refrescar_capas(event=None):
    """Fuera del dock propio, magicgui no detecta solo el visor: se refrescan
    a mano las listas de capas (Image / Shapes) de todos los bloques."""
    for bloque in secciones.values():
        bloque.reset_choices()


viewer.layers.events.inserted.connect(refrescar_capas)
viewer.layers.events.removed.connect(refrescar_capas)
viewer.layers.events.moved.connect(refrescar_capas)
refrescar_capas()
actualizar_parametros_filtro()
actualizar_parametros_deconv()
actualizar_parametros_fondo()
actualizar_parametros_realce()
