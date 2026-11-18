"""Funciones puras del TP1 (preprocesamiento y evaluación de calidad).

Reciben y devuelven arreglos y no dependen de napari, de modo que el TP3
las usa tanto en scripts como en el widget (preprocesamiento de Act. 1,
correcciones de Act. 8).
"""

import csv
from itertools import combinations
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pywt
from scipy import ndimage as ndi
from scipy.signal import fftconvolve, wiener

from skimage.draw import polygon
from skimage.exposure import equalize_adapthist
from skimage.measure import profile_line
from skimage.metrics import structural_similarity
from skimage.morphology import disk, white_tophat
from skimage.restoration import (
    denoise_nl_means,
    rolling_ball,
    denoise_tv_chambolle,
    denoise_wavelet,
    unsupervised_wiener,
)
from skimage.restoration import wiener as wiener_deconv


# ============================================================
# 1. HELPERS BASE 
# ============================================================

def calcular_estadisticas(datos):
    """
    Calcula estadísticas básicas sobre un conjunto de píxeles.

    Devuelve (n, media, varianza_muestral, desviacion).
    """
    valores = np.asarray(datos, dtype=float).ravel()
    valores = valores[np.isfinite(valores)]

    if valores.size < 2:
        raise ValueError(
            "Se necesitan al menos dos píxeles para calcular la varianza."
        )

    n = valores.size
    media = np.mean(valores)
    varianza = np.var(valores, ddof=1)   # muestral (N-1)
    desviacion = np.std(valores, ddof=1)

    return n, media, varianza, desviacion


def obtener_datos_roi(imagen_np, roi_layer, indice, origen=(0, 0)):
    """
    Devuelve los píxeles de la imagen contenidos dentro de una ROI
    (una figura de la capa Shapes). `origen` es el desplazamiento de
    la capa Image (útil si la imagen es un recorte trasladado).
    """
    imagen_np = np.asarray(imagen_np)

    if imagen_np.ndim != 2:
        raise ValueError(
            "Para esta práctica se utilizarán imágenes 2D en escala de grises."
        )

    if roi_layer is None:
        raise ValueError("Primero creá una capa de ROIs.")

    if len(roi_layer.data) == 0:
        raise ValueError("La capa ROI todavía no contiene regiones.")

    vertices = np.asarray(roi_layer.data[indice])

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("La ROI debe estar dibujada sobre una imagen 2D.")

    vertices = vertices - np.asarray(origen, dtype=float)

    filas, columnas = polygon(
        vertices[:, 0],
        vertices[:, 1],
        shape=imagen_np.shape,
    )
    return imagen_np[filas, columnas]


def obtener_indice_roi_seleccionada(roi_layer):
    """
    Índice de la ROI actualmente seleccionada con la herramienta Select.
    """
    if roi_layer is None:
        raise ValueError("Primero creá una capa de ROIs.")

    if len(roi_layer.data) == 0:
        raise ValueError("Dibujá al menos una ROI.")

    seleccionadas = list(roi_layer.selected_data)

    if len(seleccionadas) == 0:
        raise ValueError("Seleccioná una ROI con la herramienta Select.")

    if len(seleccionadas) > 1:
        raise ValueError("Seleccioná solamente una ROI para este cálculo.")

    return seleccionadas[0]


# ============================================================
# 2. CSV ACUMULATIVO (Actividad 7 - documentación)
#    Un solo CSV por trabajo: mediciones (Act. 1 y 2) y pasos del
#    pipeline conviven, distinguidos por "actividad", "region" y
#    "etiqueta_roi". Incluye las columnas mínimas pedidas por el TP.
# ============================================================

COLUMNAS_CSV = [
    "fecha_hora",
    "imagen",
    "actividad",
    "region",          # "Imagen completa", "ROI 2", ...
    "etiqueta_roi",    # segmento de la ROI: fondo, vaso, disco...
    "diagnostico",
    "metodo",
    "parametros",
    "n_px",
    "media",
    "desvio",
    "mediana",
    "sigma_mad",
    "estimacion_ruido",
    "metrica_antes",
    "metrica_despues",
    "resultados",      # resto de los valores calculados, "clave=valor; ..."
    "observaciones",
]


def formatear_valor(valor):
    """Números con 4 decimales; el resto como texto."""
    if isinstance(valor, (float, np.floating)):
        return f"{valor:.4f}"
    return str(valor)


def formatear_resultados(valores):
    """{'a': 1.5, 'b': 2} -> 'a=1.5000; b=2'."""
    return "; ".join(f"{k}={formatear_valor(v)}" for k, v in valores.items())


def crear_registro(**campos):
    """
    Arma una fila del CSV. Las columnas que no se pasan quedan vacías;
    `resultados` y `parametros` pueden ser diccionarios.
    """
    desconocidas = set(campos) - set(COLUMNAS_CSV)
    if desconocidas:
        raise ValueError(f"Columnas desconocidas: {sorted(desconocidas)}")

    registro = {columna: "" for columna in COLUMNAS_CSV}
    registro["fecha_hora"] = datetime.now().isoformat(timespec="seconds")
    for columna, valor in campos.items():
        if isinstance(valor, dict):
            valor = formatear_resultados(valor)
        registro[columna] = formatear_valor(valor)
    return registro


def guardar_registros_csv(ruta_csv, registros):
    """
    Agrega una o varias filas a un CSV acumulativo.
    Si el archivo no existe lo crea con cabecera; si existe, appendea.
    """
    if not registros:
        raise ValueError("No hay resultados pendientes para guardar.")

    ruta = Path(ruta_csv)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    existe = ruta.exists() and ruta.stat().st_size > 0
    columnas = list(registros[0].keys())

    if existe:
        with ruta.open("r", newline="", encoding="utf-8") as archivo:
            cabecera = next(csv.reader(archivo), None)
        if cabecera != columnas:
            raise ValueError(
                "El CSV existente tiene columnas distintas a las del widget "
                "(¿versión anterior?). Usá otro nombre de archivo."
            )

    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        if not existe:
            escritor.writeheader()
        escritor.writerows(registros)

    return ruta


# ============================================================
# 3. FUNCIONES DEL PIPELINE 
#    Cada una recibe una imagen 2D (np.ndarray) y devuelve otra 2D.
# ============================================================

# ------------------------------------------------------------
# Actividad 1 - Diagnóstico avanzado
# (histograma, perfiles, MAD, mapas de variabilidad local,
#  diferencias entre píxeles, FFT, autocorrelación, wavelet)
# ------------------------------------------------------------


def a_escala_de_grises(imagen_np):
    """
    Convierte una imagen 2D, RGB o RGBA en una imagen 2D float
    en escala de grises, conservando la escala original (0-255 en uint8).

    Si los canales R, G y B son iguales (gris guardado como color)
    se toma uno de ellos; si no, se usa la luminancia
    Y = 0.299 R + 0.587 G + 0.114 B. El canal alfa se descarta.
    """
    img = np.asarray(imagen_np)

    if img.ndim == 2:
        return img.astype(float)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(float)
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(
            rgb[..., 0], rgb[..., 2]
        ):
            return rgb[..., 0]
        return rgb @ np.array([0.299, 0.587, 0.114])

    raise ValueError(
        f"Formato no soportado {img.shape}: se esperaba 2D, RGB o RGBA."
    )


def quitar_marco(imagen_np):
    """
    Elimina filas/columnas de los bordes que tienen un valor constante
    (por ejemplo el marco blanco de las imágenes Retina).
    Devuelve (imagen_recortada, (fila0, columna0)).
    """
    img = np.asarray(imagen_np)
    f0, f1, c0, c1 = 0, img.shape[0], 0, img.shape[1]

    cambio = True
    while cambio and (f1 - f0) > 2 and (c1 - c0) > 2:
        cambio = False
        if np.ptp(img[f0, c0:c1]) == 0:
            f0 += 1
            cambio = True
        if np.ptp(img[f1 - 1, c0:c1]) == 0:
            f1 -= 1
            cambio = True
        if np.ptp(img[f0:f1, c0]) == 0:
            c0 += 1
            cambio = True
        if np.ptp(img[f0:f1, c1 - 1]) == 0:
            c1 -= 1
            cambio = True

    return img[f0:f1, c0:c1], (f0, c0)


def recortar_roi(imagen_np, vertices):
    """
    Recorta el rectángulo que contiene a la ROI (bounding box).
    Los análisis 2D (FFT, autocorrelación, wavelet) necesitan
    una región rectangular. Devuelve (recorte, (fila0, columna0)).
    """
    vertices = np.asarray(vertices)
    alto, ancho = imagen_np.shape[:2]

    f0 = int(np.clip(np.floor(vertices[:, 0].min()), 0, alto - 1))
    f1 = int(np.clip(np.ceil(vertices[:, 0].max()), f0 + 1, alto))
    c0 = int(np.clip(np.floor(vertices[:, 1].min()), 0, ancho - 1))
    c1 = int(np.clip(np.ceil(vertices[:, 1].max()), c0 + 1, ancho))

    return imagen_np[f0:f1, c0:c1], (f0, c0)


def estadisticas_robustas(datos):
    """
    Media, varianza, desvío, mediana y MAD.
    sigma_MAD = 1.4826 * MAD estima el desvío si el ruido es gaussiano,
    y es poco sensible a bordes y valores atípicos.
    """
    n, media, varianza, desviacion = calcular_estadisticas(datos)
    valores = np.asarray(datos, dtype=float).ravel()
    valores = valores[np.isfinite(valores)]

    mediana = np.median(valores)
    mad = np.median(np.abs(valores - mediana))

    return {
        "n": n,
        "media": media,
        "varianza": varianza,
        "desviacion": desviacion,
        "mediana": mediana,
        "mad": mad,
        "sigma_mad": 1.4826 * mad,
    }


def mapa_variabilidad_local(imagen, ventana=7):
    """Desvío estándar local en una ventana ventana x ventana."""
    img = np.asarray(imagen, dtype=float)
    media = ndi.uniform_filter(img, ventana)
    media_cuad = ndi.uniform_filter(img ** 2, ventana)
    return np.sqrt(np.clip(media_cuad - media ** 2, 0, None))


def diferencias_pixeles(imagen):
    """
    Diferencias entre píxeles vecinos (horizontal y vertical).
    Si el ruido es blanco con desvío sigma, Var(dx) = 2 sigma^2,
    así que sigma ~ std(dx)/sqrt(2); la versión MAD ignora los bordes.
    """
    img = np.asarray(imagen, dtype=float)
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    todas = np.concatenate([dx.ravel(), dy.ravel()])

    mad = np.median(np.abs(todas - np.median(todas)))
    curtosis = np.mean((todas - todas.mean()) ** 4) / todas.var() ** 2

    return {
        "dx": dx,
        "dy": dy,
        "sigma_std": todas.std() / np.sqrt(2),
        "sigma_mad": 1.4826 * mad / np.sqrt(2),
        "curtosis": curtosis,
    }


def _distancia_radial(forma):
    alto, ancho = forma
    filas, columnas = np.indices(forma)
    return np.hypot(filas - alto // 2, columnas - ancho // 2)


def espectro_fourier(imagen):
    """
    Espectro de magnitud (log) centrado, con ventana de Hann para
    evitar la cruz que generan los bordes. Busca el pico más prominente
    fuera del centro respecto de la mediana de su anillo: un pico alto
    indica un patrón periódico (artefacto estructurado).
    """
    img = np.asarray(imagen, dtype=float)
    alto, ancho = img.shape
    ventana = np.outer(np.hanning(alto), np.hanning(ancho))

    magnitud = np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * ventana)))
    log_mag = np.log1p(magnitud)

    radio = _distancia_radial(img.shape)
    anillos = radio.astype(int)
    mediana_anillo = np.asarray(
        ndi.median(log_mag, anillos, np.arange(anillos.max() + 1))
    )
    prominencia = log_mag - mediana_anillo[anillos]

    radio_min = max(4, int(0.015 * min(alto, ancho)))
    candidatos = np.where(radio > radio_min, prominencia, -np.inf)
    fila, columna = np.unravel_index(np.argmax(candidatos), candidatos.shape)

    fy = (fila - alto // 2) / alto        # ciclos / píxel
    fx = (columna - ancho // 2) / ancho
    frecuencia = np.hypot(fx, fy)

    # Densidad espectral radial (sin ventana) para el análisis por bandas
    potencia = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    radio_norm = radio / (min(alto, ancho) / 2)   # 1 = Nyquist

    def energia(lo, hi):
        return potencia[(radio_norm > lo) & (radio_norm <= hi)].sum()

    relacion_altas = energia(0.25, 0.5) / max(energia(0.10, 0.25), 1e-12)

    n_bins = min(alto, ancho) // 2
    psd_radial = np.asarray(
        ndi.mean(potencia, anillos, np.arange(n_bins))
    )

    return {
        "log_magnitud": log_mag,
        "pico_prominencia": float(prominencia[fila, columna]),
        "pico_posicion": (int(fila), int(columna)),
        "pico_frecuencia": (float(fy), float(fx)),
        "pico_periodo": 1.0 / frecuencia if frecuencia > 0 else np.inf,
        "relacion_altas": float(relacion_altas),
        "psd_radial": psd_radial,
        "frecuencias_radiales": np.arange(n_bins) / min(alto, ancho),
    }


def autocorrelacion(imagen):
    """
    Autocorrelación normalizada (Wiener-Khinchin: IFFT de |FFT|^2),
    centrada en el origen. Ruido blanco -> cae casi a cero en lag 1;
    desenfoque -> pico ancho; patrón periódico -> oscilaciones.
    """
    img = np.asarray(imagen, dtype=float)
    centrada = img - img.mean()
    potencia = np.abs(np.fft.fft2(centrada)) ** 2
    ac = np.real(np.fft.ifft2(potencia))
    ac /= ac[0, 0]
    ac = np.fft.fftshift(ac)

    alto, ancho = img.shape
    fila_c, col_c = alto // 2, ancho // 2
    perfil_x = ac[fila_c, col_c:]
    perfil_y = ac[fila_c:, col_c]

    def ancho_medio(perfil):
        # primer lag donde la autocorrelación cae por debajo de 0.5
        debajo = np.where(perfil < 0.5)[0]
        return int(debajo[0]) if debajo.size else len(perfil)

    return {
        "mapa": ac,
        "lag1_x": float(perfil_x[1]),
        "lag1_y": float(perfil_y[1]),
        "perfil_x": perfil_x,
        "perfil_y": perfil_y,
        "ancho_medio_x": ancho_medio(perfil_x),
        "ancho_medio_y": ancho_medio(perfil_y),
    }


def descomposicion_wavelet(imagen, wavelet="db2", niveles=3):
    """
    Descomposición wavelet 2D. El ruido estimado con el detalle
    diagonal del nivel 1: sigma = mediana(|HH1|) / 0.6745 (Donoho).
    """
    img = np.asarray(imagen, dtype=float)
    max_niveles = pywt.dwtn_max_level(img.shape, wavelet)
    niveles = int(max(1, min(niveles, max_niveles)))

    coeficientes = pywt.wavedec2(img, wavelet, level=niveles)
    horizontal_1, vertical_1, diagonal_1 = coeficientes[-1]

    # energía de detalle por nivel (1 = más fino)
    energias = []
    for detalles in coeficientes[:0:-1]:
        energias.append(sum(float(np.mean(d ** 2)) for d in detalles))

    return {
        "niveles": niveles,
        "aproximacion": coeficientes[0],
        "horizontal_1": horizontal_1,
        "vertical_1": vertical_1,
        "diagonal_1": diagonal_1,
        "sigma_donoho": float(np.median(np.abs(diagonal_1)) / 0.6745),
        "energias_detalle": energias,
    }


def indicador_fondo(imagen):
    """
    Compara el nivel medio de las cuatro esquinas contra la zona
    central (suavizado leve). Si difiere mucho, hay inhomogeneidad
    (viñeteado, iluminación no uniforme). El mapa de fondo que se
    devuelve es un gaussiano de gran escala, para inspección visual.
    """
    img = np.asarray(imagen, dtype=float)
    fondo = ndi.gaussian_filter(img, sigma=max(3, min(img.shape) / 10))
    suave = ndi.gaussian_filter(img, sigma=3)

    alto, ancho = img.shape
    lado = max(2, int(0.12 * min(alto, ancho)))
    esquinas = np.mean([
        suave[:lado, :lado].mean(),
        suave[:lado, -lado:].mean(),
        suave[-lado:, :lado].mean(),
        suave[-lado:, -lado:].mean(),
    ])

    fc, cc = alto // 2, ancho // 2
    medio = max(1, int(0.2 * min(alto, ancho)))
    centro = suave[fc - medio:fc + medio, cc - medio:cc + medio]

    return {
        "fondo": fondo,
        "relacion_borde_centro": float(esquinas / max(centro.mean(), 1e-12)),
    }


def fraccion_impulsos(imagen, umbral=60):
    """
    Fracción de píxeles que se apartan mucho de la mediana 3x3:
    detecta ruido impulsivo (sal y pimienta).
    """
    img = np.asarray(imagen, dtype=float)
    return float(np.mean(np.abs(img - ndi.median_filter(img, 3)) > umbral))


def diagnosticar(imagen, ventana=7, wavelet="db2", niveles=3, **kwargs):
    """
    Corre todos los análisis de la Actividad 1 sobre una imagen 2D
    (ya en escala de grises) y devuelve un diccionario con los
    indicadores y los mapas. No interpreta: la hipótesis de la
    degradación dominante se formula en el informe.
    """
    img = a_escala_de_grises(imagen)
    if min(img.shape) < 16:
        raise ValueError("La región es muy chica: usá al menos 16x16 píxeles.")

    estadisticas = estadisticas_robustas(img)
    variabilidad = mapa_variabilidad_local(img, ventana)
    diferencias = diferencias_pixeles(img)
    fourier = espectro_fourier(img)
    auto = autocorrelacion(img)
    ondita = descomposicion_wavelet(img, wavelet, niveles)
    fondo = indicador_fondo(img)

    rango = np.percentile(img, 99) - np.percentile(img, 1)

    indicadores = {
        "rango_p1_p99": rango,
        "sigma_wavelet": ondita["sigma_donoho"],
        "sigma_wavelet_sobre_rango": ondita["sigma_donoho"] / max(rango, 1e-12),
        "sigma_diferencias_mad": diferencias["sigma_mad"],
        "curtosis_diferencias": diferencias["curtosis"],
        "fraccion_impulsos": fraccion_impulsos(img),
        "pico_fft_prominencia": fourier["pico_prominencia"],
        "pico_fft_periodo_px": fourier["pico_periodo"],
        "relacion_altas_medias": fourier["relacion_altas"],
        "autocorr_lag1_x": auto["lag1_x"],
        "autocorr_lag1_y": auto["lag1_y"],
        "autocorr_ancho_medio_x": auto["ancho_medio_x"],
        "autocorr_ancho_medio_y": auto["ancho_medio_y"],
        "fondo_borde_centro": fondo["relacion_borde_centro"],
        "variabilidad_local_media": float(variabilidad.mean()),
    }

    return {
        "imagen": img,
        "rango": rango,
        "estadisticas": estadisticas,
        "variabilidad": variabilidad,
        "diferencias": diferencias,
        "fourier": fourier,
        "autocorrelacion": auto,
        "wavelet": ondita,
        "fondo": fondo,
        "indicadores": indicadores,
    }


# ------------------------------------------------------------
# Actividad 2 - Estimación de ruido
# (ROI homogénea, múltiples ROIs, MAD, diferencias, wavelet)
# ------------------------------------------------------------
METODOS_RUIDO = [
    "ROI homogénea",
    "Múltiples ROIs",
    "MAD",
    "Diferencias entre adquisiciones",
    "Diferencias locales",
    "Coeficientes wavelet",
]

def _valores(datos):
    valores = np.asarray(datos, dtype=float).ravel()
    return valores[np.isfinite(valores)]


def ruido_roi_homogenea(datos, media_fija=None):
    """
    σ = desvío muestral de una ROI homogénea.
    Con media fija μ0 (por ejemplo 0 en una ROI de fondo) se calcula además
    la dispersión alrededor de μ0: σ_μ0² = mean((x - μ0)²), que al conocer la
    media se divide por n. Vale σ_μ0² ≈ σ² + Δ², con Δ = media_obs - μ0:
    separa la parte aleatoria (σ) del sesgo sistemático (Δ).
    """
    est = estadisticas_robustas(datos)
    valores = _valores(datos)
    sigma = est["desviacion"]

    resultado = {
        "sigma": sigma,
        "media": est["media"],
        "n": est["n"],
        "snr": est["media"] / sigma if sigma > 0 else np.inf,
        "muestra": valores,
        "centro": est["media"],
    }
    if media_fija is not None:
        resultado["media_fija"] = float(media_fija)
        resultado["delta"] = est["media"] - media_fija
        resultado["sigma_media_fija"] = float(
            np.sqrt(np.mean((valores - media_fija) ** 2))
        )
    return resultado


def ruido_mad(datos, media_fija=None):
    """
    σ robusto = 1.4826 · mediana(|x - mediana|).
    Con media fija μ0 se informa además 1.4826 · mediana(|x - μ0|), que
    mezcla el ruido con el sesgo Δ = mediana - μ0 (igual que en ROI homogénea).
    """
    valores = _valores(datos)
    if valores.size < 2:
        raise ValueError("La región tiene menos de dos píxeles.")
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))
    resultado = {
        "sigma": 1.4826 * mad,
        "mad": mad,
        "mediana": mediana,
        "n": valores.size,
        "muestra": valores,
        "centro": mediana,
    }
    if media_fija is not None:
        resultado["media_fija"] = float(media_fija)
        resultado["delta"] = mediana - media_fija
        resultado["sigma_media_fija"] = 1.4826 * float(
            np.median(np.abs(valores - media_fija))
        )
    return resultado


def ruido_multiples_rois(lista_rois):
    """
    lista_rois: [(indice, pixeles), ...]. Devuelve el σ combinado
    sqrt(Σ(n_i - 1) s_i² / Σ(n_i - 1)) y el ajuste lineal var = a + b·media,
    que indica si el ruido depende de la intensidad.
    """
    if len(lista_rois) < 2:
        raise ValueError("Dibujá al menos 2 ROIs homogéneas.")

    filas = []
    residuos = []
    for indice, datos in lista_rois:
        n, media, varianza, desvio = calcular_estadisticas(datos)
        filas.append((indice, n, media, varianza, desvio))
        residuos.append(_valores(datos) - media)

    n = np.array([f[1] for f in filas], dtype=float)
    medias = np.array([f[2] for f in filas])
    varianzas = np.array([f[3] for f in filas])
    desvios = np.array([f[4] for f in filas])

    sigma = float(np.sqrt(np.sum((n - 1) * varianzas) / np.sum(n - 1)))
    cv_sigma = float(desvios.std() / max(desvios.mean(), 1e-12))

    ajuste = None
    if len(filas) >= 3 and np.ptp(medias) > 0:
        b, a = np.polyfit(medias, varianzas, 1)
        prediccion = a + b * medias
        ss_tot = np.sum((varianzas - varianzas.mean()) ** 2)
        r2 = 1 - np.sum((varianzas - prediccion) ** 2) / max(ss_tot, 1e-12)
        ajuste = {"a": float(a), "b": float(b), "r2": float(r2)}

    return {
        "sigma": sigma,
        "rois": filas,
        "cv_sigma": cv_sigma,
        "ajuste": ajuste,
        "muestra": np.concatenate(residuos),
        "centro": 0.0,
    }


def ruido_adquisiciones(imagen_a, imagen_b, referencia_limpia=False):
    """
    σ a partir de la diferencia entre dos adquisiciones de la misma escena.
    Independientes con igual σ: σ = std(A-B)/√2. Si B es una referencia
    sin ruido: σ = std(A-B). Se informa también la versión MAD (robusta a
    bordes mal alineados) y el offset medio entre ambas.
    """
    a = np.asarray(imagen_a, dtype=float)
    b = np.asarray(imagen_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"Las adquisiciones tienen tamaños distintos: {a.shape} vs {b.shape}."
        )

    diferencia = a - b
    factor = 1.0 if referencia_limpia else np.sqrt(2)
    mediana = float(np.median(diferencia))
    mad = float(np.median(np.abs(diferencia - mediana)))

    if mad == 0 and diferencia.std() == 0:
        raise ValueError("Las dos imágenes son idénticas: elegí otra adquisición.")

    return {
        "sigma": 1.4826 * mad / factor,
        "sigma_std": float(diferencia.std(ddof=1) / factor),
        "offset": float(diferencia.mean()),
        "factor": float(factor),
        "correlacion": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "mapa": diferencia,
        "muestra": (diferencia.ravel() - mediana) / factor,
        "centro": 0.0,
    }


def ruido_diferencias_locales(imagen):
    """σ con diferencias entre vecinos: 1.4826·MAD(dx, dy)/√2."""
    dif = diferencias_pixeles(imagen)
    todas = np.concatenate([dif["dx"].ravel(), dif["dy"].ravel()])
    return {
        "sigma": dif["sigma_mad"],
        "sigma_std": dif["sigma_std"],
        "curtosis": dif["curtosis"],
        "mapa": dif["dx"],
        "muestra": (todas - np.median(todas)) / np.sqrt(2),
        "centro": 0.0,
    }


def ruido_wavelet(imagen, wavelet="db2"):
    """σ de Donoho sobre HH1 (wavelet ortonormal: HH1 ~ N(0, σ) si es ruido)."""
    ondita = descomposicion_wavelet(imagen, wavelet, niveles=1)
    return {
        "sigma": ondita["sigma_donoho"],
        "mapa": ondita["diagonal_1"],
        "muestra": ondita["diagonal_1"].ravel(),
        "centro": 0.0,
    }


def estimar_ruido(
    imagen,
    metodo="ROI homogénea",
    datos_roi=None,
    recorte_roi=None,
    lista_rois=None,
    segunda=None,
    referencia_limpia=False,
    media_fija=None,
    wavelet="db2",
    motivos=None,
):
    """
    Estima σ del ruido con la estrategia elegida (`metodo`) y, para comparar,
    con todas las demás que se puedan aplicar con los datos disponibles.

    imagen:      región rectangular 2D (diferencias locales / wavelet).
    datos_roi:   píxeles de la ROI seleccionada (ROI homogénea / MAD).
    recorte_roi: rectángulo de esa ROI (para chequear que sea homogénea).
    lista_rois:  [(indice, pixeles)] de todas las ROIs (Múltiples ROIs).
    segunda:     misma región en otra adquisición (Diferencias entre adq.).
    motivos:     {metodo: texto} con la razón por la que un dato falta.
    """
    img = a_escala_de_grises(imagen)
    motivos = motivos or {}

    def requerir(dato, nombre, mensaje):
        if dato is None:
            raise ValueError(motivos.get(nombre, mensaje))
        return dato

    calculadores = {
        "ROI homogénea": lambda: ruido_roi_homogenea(
            requerir(datos_roi, "ROI", "Seleccioná una ROI homogénea."),
            media_fija,
        ),
        "Múltiples ROIs": lambda: ruido_multiples_rois(lista_rois or []),
        "MAD": lambda: ruido_mad(
            requerir(datos_roi, "ROI", "Seleccioná una ROI homogénea."),
            media_fija,
        ),
        "Diferencias entre adquisiciones": lambda: ruido_adquisiciones(
            img,
            requerir(segunda, "segunda", "Elegí otra adquisición de la escena."),
            referencia_limpia,
        ),
        "Diferencias locales": lambda: ruido_diferencias_locales(img),
        "Coeficientes wavelet": lambda: ruido_wavelet(img, wavelet),
    }

    resultados = {}
    for nombre, calcular in calculadores.items():
        try:
            resultados[nombre] = calcular()
        except Exception as error:
            resultados[nombre] = {"error": str(error)}

    principal = resultados[metodo]
    if "error" in principal:
        raise ValueError(f"{metodo}: {principal['error']}")
    sigma = principal["sigma"]

    # --- Valores complementarios (sin interpretar)
    complementarios = {"fraccion_impulsos": fraccion_impulsos(img)}
    if recorte_roi is not None and min(recorte_roi.shape) >= 3:
        # σ por diferencias locales dentro de la misma ROI: comparándolo con
        # el σ de la ROI se ve cuánto aporta la estructura de la región.
        complementarios["sigma_dif_locales_en_roi"] = (
            diferencias_pixeles(recorte_roi)["sigma_mad"]
        )

    return {
        "metodo": metodo,
        "sigma": sigma,
        "principal": principal,
        "resultados": resultados,
        "complementarios": complementarios,
    }


# ------------------------------------------------------------
# Actividad 3 - Reducción de ruido
# (mediana, gaussiano, bilateral, difusión anisotrópica, NLM, TV,
#  wavelet thresholding, Wiener local, BM3D)
# ------------------------------------------------------------
#
# Convención de unidades: todo parámetro de INTENSIDAD (σ del ruido,
# sigma_color, h, kappa) se da en la escala original de la imagen
# (0-255 en uint8), la misma del σ estimado en la Act. 2. Internamente
# se normaliza a [0, 1] para las funciones que lo necesitan.

METODOS_FILTRO = [
    "Mediana",
    "Gaussiano",
    "Bilateral",
    "Difusión anisotrópica",
    "Non-Local Means",
    "Variación Total",
    "Wavelet thresholding",
    "Wiener local",
    "BM3D",
]


def filtro_mediana(img, ventana=3, forma="Cuadrada", umbral_impulso=0.0):
    """
    Mediana en una ventana de `ventana` x `ventana` px (impar). Es el filtro
    indicado para ruido impulsivo (sal y pimienta): un píxel aislado a 0 o
    255 no mueve la mediana del vecindario, en cambio arrastraría un promedio.
      forma "Disco": vecindario circular inscripto en la ventana (menos
        distorsión de esquinas y bordes diagonales).
      umbral_impulso > 0: mediana CONMUTADA; solo se reemplazan los píxeles
        que se apartan de la mediana local más que ese valor (intensidad).
        El resto queda intacto, así se conserva el detalle fino.
    """
    ventana = int(ventana)
    if ventana < 3 or ventana % 2 == 0:
        raise ValueError("La ventana de la mediana debe ser impar y >= 3.")

    img = np.asarray(img, dtype=float)
    if forma == "Disco":
        mediana = ndi.median_filter(img, footprint=disk(ventana // 2))
    else:
        mediana = ndi.median_filter(img, size=ventana)

    if umbral_impulso > 0:
        return np.where(np.abs(img - mediana) > umbral_impulso, mediana, img)
    return mediana


def filtro_gaussiano(img, sigma_espacial=1.0):
    return ndi.gaussian_filter(img, sigma=sigma_espacial)


def filtro_bilateral(img, sigma_color=20.0, sigma_espacial=2.0):
    """
    Promedia vecinos cercanos en el espacio (sigma_espacial, px) Y en
    intensidad (sigma_color): no mezcla píxeles a ambos lados de un borde.
    """
    return cv2.bilateralFilter(
        img.astype(np.float32), d=-1,
        sigmaColor=float(sigma_color), sigmaSpace=float(sigma_espacial),
    ).astype(float)


def difusion_anisotropica(img, iteraciones=15, kappa=20.0, gamma=0.2,
                          funcion="exponencial"):
    """
    Perona-Malik. En cada iteración difunde hacia los 4 vecinos con un
    coeficiente g(|∇I|) que se apaga en los bordes (gradiente >> kappa):
      exponencial: g = exp(-(∇I/kappa)²)   (favorece bordes de alto contraste)
      cuadrática:  g = 1 / (1 + (∇I/kappa)²) (favorece regiones amplias)
    gamma <= 0.25 para que el esquema explícito sea estable.
    """
    if not 0 < gamma <= 0.25:
        raise ValueError("gamma debe estar en (0, 0.25] para que sea estable.")
    if kappa <= 0:
        raise ValueError("kappa debe ser positivo.")

    u = np.asarray(img, dtype=float).copy()
    for _ in range(int(iteraciones)):
        p = np.pad(u, 1, mode="edge")
        gradientes = (
            p[:-2, 1:-1] - u,   # norte
            p[2:, 1:-1] - u,    # sur
            p[1:-1, 2:] - u,    # este
            p[1:-1, :-2] - u,   # oeste
        )
        flujo = np.zeros_like(u)
        for d in gradientes:
            if funcion == "exponencial":
                g = np.exp(-((d / kappa) ** 2))
            else:
                g = 1.0 / (1.0 + (d / kappa) ** 2)
            flujo += g * d
        u += gamma * flujo
    return u


def filtro_nlm(img, h=10.0, sigma_ruido=0.0, tamano_parche=5,
               distancia_busqueda=6, escala=255.0):
    """
    Non-Local Means: cada píxel es el promedio de píxeles cuyo PARCHE
    (vecindario) se parece al suyo, buscando en una ventana amplia.
    h controla cuánto se parecen tienen que ser (unidades de intensidad).
    sigma_ruido > 0: se descuenta la varianza del ruido de la distancia
    entre parches (recomendado por skimage); 0 = no se usa.
    """
    salida = denoise_nl_means(
        img / escala,
        h=h / escala,
        sigma=sigma_ruido / escala,
        patch_size=int(tamano_parche),
        patch_distance=int(distancia_busqueda),
        fast_mode=True,
    )
    return salida * escala


def filtro_tv(img, peso=0.05, escala=255.0):
    """
    Variación Total (Chambolle): minimiza ||u - f||² + peso·TV(u).
    Produce regiones planas con bordes nítidos; mucho peso -> "cartoon".
    El peso es adimensional (imagen normalizada a [0, 1]).
    """
    return denoise_tv_chambolle(img / escala, weight=peso) * escala


def filtro_wavelet(img, sigma_ruido=0.0, wavelet="db2", metodo="BayesShrink",
                   modo="soft", niveles=0, escala=255.0):
    """
    Umbraliza los coeficientes de detalle wavelet (el ruido se reparte en
    muchos coeficientes chicos; la estructura, en pocos grandes).
      VisuShrink: umbral universal σ·√(2 ln N) (conservador, suaviza más).
      BayesShrink: un umbral por subbanda, adaptado a su varianza.
    sigma_ruido = 0: lo estima skimage (Donoho sobre HH1). niveles = 0: auto.
    """
    salida = denoise_wavelet(
        img / escala,
        sigma=(sigma_ruido / escala) if sigma_ruido > 0 else None,
        wavelet=wavelet,
        mode=modo,
        method=metodo,
        wavelet_levels=int(niveles) if niveles > 0 else None,
        rescale_sigma=True,
    )
    return salida * escala


def filtro_wiener_local(img, ventana=5, sigma_ruido=0.0):
    """
    Wiener adaptativo (Lee): en cada ventana
      u = μ + max(σ²_local - σ²_n, 0) / σ²_local · (f - μ)
    Donde la varianza local ≈ σ²_n (zona plana) promedia; donde es mucho
    mayor (bordes) deja la imagen casi igual.
    sigma_ruido = 0: σ²_n = promedio de las varianzas locales (scipy).
    """
    ruido = sigma_ruido ** 2 if sigma_ruido > 0 else None
    return wiener(img, mysize=int(ventana), noise=ruido)


def filtro_bm3d(img, sigma_ruido, etapas="Completo (2 etapas)", escala=255.0):
    """
    BM3D: agrupa parches parecidos en pilas 3D, filtra cada pila en un
    dominio transformado (umbral duro y luego Wiener) y reagrega.
    Necesita el σ del ruido (unidades de intensidad).
    """
    try:
        import bm3d
    except ImportError as error:
        raise ImportError(
            "Falta el paquete bm3d: correr 'pixi install' para actualizar el entorno."
        ) from error
    if sigma_ruido <= 0:
        raise ValueError("BM3D necesita σ del ruido > 0 (estimalo en la Act. 2).")

    etapa = (
        bm3d.BM3DStages.ALL_STAGES
        if etapas.startswith("Completo")
        else bm3d.BM3DStages.HARD_THRESHOLDING
    )
    salida = bm3d.bm3d(img / escala, sigma_psd=sigma_ruido / escala, stage_arg=etapa)
    return np.asarray(salida, dtype=float) * escala


def reducir_ruido(imagen, metodo="Gaussiano", escala=255.0, **p):
    """
    Aplica el filtro `metodo` a una imagen 2D y devuelve la imagen filtrada
    (float, misma escala). `p` son los parámetros del widget; cada método
    toma solo los suyos.
    """
    img = a_escala_de_grises(imagen)

    if metodo == "Mediana":
        return filtro_mediana(
            img, p["ventana_mediana"], p["forma_mediana"], p["umbral_impulso"]
        )
    if metodo == "Gaussiano":
        return filtro_gaussiano(img, p["sigma_espacial"])
    if metodo == "Bilateral":
        return filtro_bilateral(img, p["sigma_color"], p["sigma_espacial"])
    if metodo == "Difusión anisotrópica":
        return difusion_anisotropica(
            img, p["iteraciones"], p["kappa"], p["gamma"], p["funcion_conduccion"]
        )
    if metodo == "Non-Local Means":
        return filtro_nlm(
            img, p["h"], p["sigma_ruido"], p["tamano_parche"],
            p["distancia_busqueda"], escala,
        )
    if metodo == "Variación Total":
        return filtro_tv(img, p["peso_tv"], escala)
    if metodo == "Wavelet thresholding":
        return filtro_wavelet(
            img, p["sigma_ruido"], p["wavelet"], p["umbral"],
            p["modo_umbral"], p["niveles_wavelet"], escala,
        )
    if metodo == "Wiener local":
        return filtro_wiener_local(img, p["ventana"], p["sigma_ruido"])
    if metodo == "BM3D":
        return filtro_bm3d(img, p["sigma_ruido"], p["etapas_bm3d"], escala)

    raise ValueError(f"Método desconocido: {metodo}")


# ------------------------------------------------------------
# Actividad 4 - Restauración y artefactos
# (deconvolución Wiener / Richardson-Lucy; filtro notch en Fourier)
# ------------------------------------------------------------
#
# 4.a  PSF y deconvolución
#      La PSF puede sintetizarse (gaussiana, disco, movimiento) o venir
#      de una capa de napari (PSF medida o provista por la cátedra).

TIPOS_PSF = ["Gaussiana", "Disco (desenfoque)", "Movimiento lineal", "Capa de napari"]
METODOS_DECONVOLUCION = ["Wiener", "Wiener no supervisado", "Richardson-Lucy"]


def psf_gaussiana(sigma=2.0):
    """PSF gaussiana isótropa de desvío sigma (px)."""
    if sigma <= 0:
        raise ValueError("El sigma de la PSF debe ser positivo.")
    radio = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-radio:radio + 1, -radio:radio + 1]
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def psf_disco(radio=3.0):
    """PSF de disco (desenfoque por apertura circular)."""
    if radio <= 0:
        raise ValueError("El radio de la PSF debe ser positivo.")
    r = int(np.ceil(radio))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    psf = (np.hypot(x, y) <= radio).astype(float)
    return psf / psf.sum()


def psf_movimiento(longitud=9.0, angulo=0.0):
    """PSF de movimiento lineal: un segmento de `longitud` px a `angulo` grados."""
    if longitud < 2:
        raise ValueError("La longitud de la PSF debe ser de al menos 2 px.")
    radio = int(np.ceil(longitud / 2))
    psf = np.zeros((2 * radio + 1, 2 * radio + 1))
    theta = np.deg2rad(angulo)
    pasos = np.linspace(-longitud / 2, longitud / 2, int(4 * longitud))
    filas = np.round(radio - pasos * np.sin(theta)).astype(int)
    columnas = np.round(radio + pasos * np.cos(theta)).astype(int)
    np.add.at(psf, (filas, columnas), 1.0)
    return psf / psf.sum()


def psf_desde_datos(datos):
    """PSF a partir de una capa: pasa a gris, recorta negativos y normaliza."""
    psf = a_escala_de_grises(datos)
    psf = np.clip(psf - psf.min(), 0, None)
    total = psf.sum()
    if total <= 0:
        raise ValueError("La capa elegida como PSF está vacía.")
    return psf / total


def construir_psf(tipo, sigma_psf=2.0, radio_psf=3.0, longitud_psf=9.0,
                  angulo_psf=0.0, datos_psf=None):
    if tipo == "Gaussiana":
        return psf_gaussiana(sigma_psf)
    if tipo == "Disco (desenfoque)":
        return psf_disco(radio_psf)
    if tipo == "Movimiento lineal":
        return psf_movimiento(longitud_psf, angulo_psf)
    if tipo == "Capa de napari":
        if datos_psf is None:
            raise ValueError("Elegí la capa que contiene la PSF.")
        return psf_desde_datos(datos_psf)
    raise ValueError(f"Tipo de PSF desconocido: {tipo}")


def _con_borde(img, ancho):
    """Extiende los bordes (evita el anillado de la convolución circular)."""
    return np.pad(img, ancho, mode="edge")


def _sin_borde(img, ancho):
    return img[ancho:-ancho, ancho:-ancho] if ancho > 0 else img


def deconvolucion_wiener(img, psf, balance=0.01, escala=255.0, no_supervisado=False):
    """
    Deconvolución de Wiener: U = (H* / (|H|² + balance·|R|²)) · F.
    `balance` es la regularización: es el peso del término que penaliza las
    altas frecuencias. Un valor del orden de NSR = σ²/Var(imagen) equilibra
    resolución y ruido; más chico afina pero amplifica el ruido.
    "No supervisado" estima esa regularización con un muestreo bayesiano
    (skimage.restoration.unsupervised_wiener) en vez de fijarla a mano.
    """
    ancho = max(psf.shape) // 2 + 1
    extendida = _con_borde(np.asarray(img, dtype=float) / escala, ancho)

    if no_supervisado:
        salida, _ = unsupervised_wiener(extendida, psf, clip=False)
        usado = None
    else:
        if balance <= 0:
            raise ValueError("El balance (regularización) debe ser positivo.")
        salida = wiener_deconv(extendida, psf, balance=balance, clip=False)
        usado = balance

    return _sin_borde(np.asarray(salida, dtype=float), ancho) * escala, usado


def richardson_lucy(img, psf, iteraciones=20, tolerancia=0.0, escala=255.0):
    """
    Richardson-Lucy: iteración multiplicativa que supone ruido de Poisson.
      u_{k+1} = u_k · [ (f / (u_k * h)) * h⁻ ]
    Devuelve (imagen, historial). En cada iteración se registra:
      - residuo_rms: RMS de f - (u_k * h), el ajuste a los datos observados;
      - cambio: ||u_k - u_{k-1}|| / ||u_{k-1}||, cuánto se mueve la solución.
    RL no converge a algo estable: amplifica el ruido a partir de cierta
    iteración, así que hace falta un criterio de parada. Con tolerancia > 0
    se corta cuando el cambio relativo baja de ese valor.
    """
    psf = np.asarray(psf, dtype=float)
    psf_espejo = psf[::-1, ::-1]
    ancho = max(psf.shape) // 2 + 1
    f = _con_borde(np.asarray(img, dtype=float) / escala, ancho)

    u = np.full(f.shape, f.mean())
    historial = []
    eps = 1e-12

    for k in range(1, int(iteraciones) + 1):
        convolucion = fftconvolve(u, psf, mode="same")
        u_previo = u
        u = u * fftconvolve(f / (convolucion + eps), psf_espejo, mode="same")
        u = np.clip(u, 0, None)

        cambio = float(
            np.linalg.norm(u - u_previo) / max(np.linalg.norm(u_previo), eps)
        )
        historial.append({
            "iteracion": k,
            "residuo_rms": float(np.sqrt(np.mean((f - convolucion) ** 2)) * escala),
            "cambio": cambio,
        })
        if tolerancia > 0 and cambio < tolerancia:
            break

    return _sin_borde(u, ancho) * escala, historial


def restaurar(imagen, metodo="Wiener", psf=None, balance=0.01, iteraciones=20,
              tolerancia=0.0, escala=255.0):
    """
    Deconvolución de una imagen 2D con una PSF dada.
    Devuelve (imagen_restaurada, informacion) donde `informacion` trae el
    balance usado (Wiener) o el historial de iteraciones (Richardson-Lucy).
    """
    img = a_escala_de_grises(imagen)
    if psf is None:
        raise ValueError("Hace falta una PSF.")

    if metodo in ("Wiener", "Wiener no supervisado"):
        salida, usado = deconvolucion_wiener(
            img, psf, balance, escala, no_supervisado=(metodo != "Wiener")
        )
        return salida, {"balance_usado": usado}
    if metodo == "Richardson-Lucy":
        salida, historial = richardson_lucy(img, psf, iteraciones, tolerancia, escala)
        return salida, {"historial": historial}

    raise ValueError(f"Método de deconvolución desconocido: {metodo}")


# 4.b  Artefactos periódicos: detección en Fourier y filtro notch

TIPOS_NOTCH = ["Gaussiano", "Ideal", "Butterworth"]


def detectar_picos_fft(imagen, cantidad=6, radio_min_rel=0.015, separacion=5):
    """
    Picos del espectro que sobresalen de su anillo (patrones periódicos).
    Para cada uno devuelve su posición en el espectro centrado, la
    frecuencia (ciclos/px), el período (px), la orientación y la
    prominencia sobre la mediana del anillo. Los pares simétricos
    (f y -f) se informan una sola vez.
    """
    img = a_escala_de_grises(imagen)
    alto, ancho = img.shape
    ventana = np.outer(np.hanning(alto), np.hanning(ancho))
    log_mag = np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * ventana)))
    )

    radio = _distancia_radial(img.shape)
    anillos = radio.astype(int)
    mediana_anillo = np.asarray(
        ndi.median(log_mag, anillos, np.arange(anillos.max() + 1))
    )
    prominencia = log_mag - mediana_anillo[anillos]

    maximos = ndi.maximum_filter(prominencia, size=separacion)
    radio_min = max(4, int(radio_min_rel * min(alto, ancho)))
    candidatos = (prominencia >= maximos) & (radio > radio_min)

    filas, columnas = np.where(candidatos)
    orden = np.argsort(prominencia[filas, columnas])[::-1]

    centro_f, centro_c = alto // 2, ancho // 2
    picos = []
    vistos = set()
    for indice in orden:
        fila, columna = int(filas[indice]), int(columnas[indice])
        simetrico = (2 * centro_f - fila, 2 * centro_c - columna)
        if (fila, columna) in vistos or simetrico in vistos:
            continue
        vistos.add((fila, columna))

        fy = (fila - centro_f) / alto
        fx = (columna - centro_c) / ancho
        frecuencia = np.hypot(fx, fy)
        picos.append({
            "fila": fila,
            "columna": columna,
            "fy": float(fy),
            "fx": float(fx),
            "frecuencia": float(frecuencia),
            "periodo_px": float(1 / frecuencia) if frecuencia > 0 else np.inf,
            "angulo_grados": float(np.degrees(np.arctan2(fy, fx))),
            "prominencia": float(prominencia[fila, columna]),
        })
        if len(picos) >= cantidad:
            break

    return picos, log_mag


def mascara_notch(forma, picos, radio=5.0, tipo="Gaussiano", orden=2):
    """
    Máscara multiplicativa (espectro centrado) que atenúa cada pico y su
    simétrico. `radio` es la extensión del notch en píxeles del espectro:
    tiene que cubrir el ancho del pico, pero no más.
      Ideal: corta de golpe (puede dejar ondulaciones tipo Gibbs).
      Gaussiano: transición suave.
      Butterworth: transición controlada por el orden.
    """
    if radio <= 0:
        raise ValueError("El radio del notch debe ser positivo.")

    alto, ancho = forma
    centro_f, centro_c = alto // 2, ancho // 2
    filas, columnas = np.indices(forma)
    mascara = np.ones(forma, dtype=float)

    for pico in picos:
        posiciones = [
            (pico["fila"], pico["columna"]),
            (2 * centro_f - pico["fila"], 2 * centro_c - pico["columna"]),
        ]
        for fila, columna in posiciones:
            d = np.hypot(filas - fila, columnas - columna)
            if tipo == "Ideal":
                mascara[d <= radio] = 0.0
            elif tipo == "Butterworth":
                with np.errstate(divide="ignore"):
                    mascara *= 1.0 / (1.0 + (radio / np.maximum(d, 1e-9)) ** (2 * orden))
            else:
                mascara *= 1.0 - np.exp(-(d ** 2) / (2 * radio ** 2))

    return mascara


def aplicar_notch(imagen, mascara):
    """Filtra la imagen con la máscara en el espectro centrado."""
    img = a_escala_de_grises(imagen)
    espectro = np.fft.fftshift(np.fft.fft2(img))
    filtrada = np.real(np.fft.ifft2(np.fft.ifftshift(espectro * mascara)))
    return filtrada, espectro


def verificar_notch(original, filtrada):
    """
    Valores para controlar que el notch se haya llevado el patrón y no la
    anatomía (la lectura es de quien analiza):
      - fraccion_energia_removida: ||o - f||² / ||o - media||².
      - correlacion_residuo_filtrada: correlación entre lo removido y lo
        que quedó (la anatomía estimada). El patrón periódico es ajeno a
        la anatomía: cerca de 0 significa que lo removido es independiente
        de lo que se conserva.
      - correlacion_residuo_original: lo mismo contra la imagen de entrada,
        que todavía contiene el patrón.
      - razon_energia_gradiente: energía de gradiente después / antes
        (cuánto se ablandaron los bordes).
      - razon_varianza: varianza después / antes.
    """
    original = np.asarray(original, dtype=float)
    filtrada = np.asarray(filtrada, dtype=float)
    residuo = original - filtrada

    def energia_gradiente(img):
        gy, gx = np.gradient(img)
        return float(np.mean(gx ** 2 + gy ** 2))

    energia_total = float(np.sum((original - original.mean()) ** 2))

    def correlacion(otra):
        if residuo.std() == 0 or otra.std() == 0:
            return 0.0
        return float(np.corrcoef(residuo.ravel(), otra.ravel())[0, 1])

    return {
        "fraccion_energia_removida": float(np.sum(residuo ** 2)) / max(energia_total, 1e-12),
        "correlacion_residuo_filtrada": correlacion(filtrada),
        "correlacion_residuo_original": correlacion(original),
        "razon_energia_gradiente": energia_gradiente(filtrada)
        / max(energia_gradiente(original), 1e-12),
        "razon_varianza": float(filtrada.var()) / max(float(original.var()), 1e-12),
        "residuo_amplitud_pico_a_pico": float(np.ptp(residuo)),
    }


# ------------------------------------------------------------
# Actividad 5 - Corrección de fondo
# (dark/flat-field, kernel de gran escala, rolling ball,
#  white top-hat, homomórfica)
# ------------------------------------------------------------
#
# Cada función devuelve (corregida, fondo_estimado): el fondo se puede
# mirar como capa para ver qué se está quitando.

METODOS_FONDO = [
    "Dark-field / flat-field",
    "Kernel de gran escala",
    "Rolling ball",
    "White top-hat",
    "Homomórfica",
]


def _nivel(img, fondo, conservar_nivel):
    """Offset que se suma tras restar el fondo (para no dejar todo en ~0)."""
    return float(np.mean(fondo)) if conservar_nivel else 0.0


def fondo_flat_field(img, flat, dark=None):
    """
    Corrección por campos de referencia:
        corregida = (I - D) / (F - D) · media(F - D)
    D (dark-field) es la señal con el obturador cerrado: corriente de
    oscuridad y offset del sensor. F (flat-field) es una toma de campo
    uniforme: recoge la respuesta del sistema (viñeteo, polvo, ganancia
    de cada píxel). El factor media(F - D) devuelve la imagen a su
    nivel original en vez de dejarla alrededor de 1.
    """
    flat = a_escala_de_grises(flat)
    oscuro = np.zeros_like(img) if dark is None else a_escala_de_grises(dark)
    if flat.shape != img.shape or oscuro.shape != img.shape:
        raise ValueError(
            "El flat-field y el dark-field deben tener el mismo tamaño que la región."
        )

    ganancia = flat - oscuro
    media = float(np.mean(ganancia))
    if media <= 0:
        raise ValueError("El flat-field menos el dark-field da un campo vacío.")

    corregida = (img - oscuro) / np.where(np.abs(ganancia) < 1e-9, 1e-9, ganancia) * media
    return corregida, ganancia


def fondo_kernel_grande(img, sigma=50.0, tipo="Gaussiano", modo="Resta",
                        conservar_nivel=True):
    """
    Estima el fondo suavizando la imagen con un kernel mucho más grande
    que las estructuras de interés (gaussiano o mediana) y lo quita.
      Resta:    corregida = I - B + media(B)   (fondo aditivo)
      División: corregida = I / B · media(B)   (fondo multiplicativo)
    """
    if sigma <= 0:
        raise ValueError("El tamaño del kernel debe ser positivo.")

    if tipo == "Mediana":
        fondo = ndi.median_filter(img, size=int(max(3, 2 * round(sigma) + 1)))
    else:
        fondo = ndi.gaussian_filter(img, sigma=sigma)

    if modo == "División":
        seguro = np.where(np.abs(fondo) < 1e-9, 1e-9, fondo)
        return img / seguro * float(np.mean(fondo)), fondo
    return img - fondo + _nivel(img, fondo, conservar_nivel), fondo


def fondo_rolling_ball(img, radio=50.0, conservar_nivel=True):
    """
    Rolling ball: hace rodar una esfera de radio `radio` por debajo de la
    superficie de intensidades; lo que toca es el fondo. El radio tiene
    que ser mayor que las estructuras que se quieren conservar.
    """
    if radio <= 0:
        raise ValueError("El radio de la bola debe ser positivo.")
    fondo = rolling_ball(img, radius=float(radio))
    return img - fondo + _nivel(img, fondo, conservar_nivel), fondo


def fondo_tophat(img, radio=25.0, conservar_nivel=True):
    """
    White top-hat: I - apertura(I). La apertura con un disco de radio
    `radio` borra todo lo más chico que el disco (queda el fondo), así
    que la resta deja las estructuras claras y chicas.
    """
    if radio <= 0:
        raise ValueError("El radio del top-hat debe ser positivo.")
    elemento = disk(int(round(radio)))
    corregida = white_tophat(img, elemento)
    fondo = img - corregida
    return corregida + _nivel(img, fondo, conservar_nivel), fondo


def fondo_homomorfico(img, gamma_bajo=0.5, gamma_alto=1.5, corte=0.05):
    """
    Corrección homomórfica: si I = iluminación · reflectancia, en log se
    vuelve una suma. La iluminación es de baja frecuencia y la
    reflectancia de alta, así que se filtra en log con
        H = gamma_bajo + (gamma_alto - gamma_bajo)·(1 - exp(-D²/D0²))
    y se vuelve con exp. gamma_bajo < 1 atenúa la iluminación despareja;
    gamma_alto > 1 realza el detalle. `corte` es D0 en fracción de
    Nyquist: más grande = se considera "iluminación" un rango más amplio.
    """
    if not 0 < corte <= 1:
        raise ValueError("El corte debe estar entre 0 y 1 (fracción de Nyquist).")

    desplazado = img - img.min() + 1.0
    logaritmo = np.log(desplazado)

    alto, ancho = img.shape
    radio = _distancia_radial(img.shape) / (min(alto, ancho) / 2)
    filtro = gamma_bajo + (gamma_alto - gamma_bajo) * (
        1 - np.exp(-(radio ** 2) / (corte ** 2))
    )

    espectro = np.fft.fftshift(np.fft.fft2(logaritmo))
    corregida = np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(espectro * filtro))))
    corregida = corregida + img.min() - 1.0

    # Atenuar las bajas frecuencias comprime el nivel general: se reescala
    # con una ganancia constante para volver a la media original (la
    # ganancia es un factor único, no cambia el contraste relativo).
    media_actual = float(np.mean(corregida))
    if abs(media_actual) > 1e-9:
        corregida = corregida * (float(np.mean(img)) / media_actual)

    # Fondo implícito: lo que se le quitó a la imagen
    return corregida, img - corregida + float(np.mean(img))


def corregir_fondo(imagen, metodo="Rolling ball", flat=None, dark=None,
                   sigma_fondo=50.0, tipo_kernel="Gaussiano",
                   modo_correccion="Resta", radio_bola=50.0, radio_tophat=25.0,
                   gamma_bajo=0.5, gamma_alto=1.5, corte=0.05,
                   conservar_nivel=True, **kwargs):
    """Aplica el método de corrección de fondo elegido: (corregida, fondo)."""
    img = a_escala_de_grises(imagen)

    if metodo == "Dark-field / flat-field":
        if flat is None:
            raise ValueError("Elegí la capa de flat-field.")
        return fondo_flat_field(img, flat, dark)
    if metodo == "Kernel de gran escala":
        return fondo_kernel_grande(
            img, sigma_fondo, tipo_kernel, modo_correccion, conservar_nivel
        )
    if metodo == "Rolling ball":
        return fondo_rolling_ball(img, radio_bola, conservar_nivel)
    if metodo == "White top-hat":
        return fondo_tophat(img, radio_tophat, conservar_nivel)
    if metodo == "Homomórfica":
        return fondo_homomorfico(img, gamma_bajo, gamma_alto, corte)

    raise ValueError(f"Método de corrección de fondo desconocido: {metodo}")


def medidas_fondo(img, lista_rois=None):
    """
    Valores para comparar antes y después (uniformidad del fondo).
      fondo_borde_centro: nivel medio de las esquinas / nivel del centro.
      rango_fondo_suave: cuánto varía la imagen muy suavizada (p95 - p5),
        es decir la amplitud de la inhomogeneidad que queda.
      cv_medias_rois: dispersión de las medias entre ROIs (si hay 2 o más);
        con el fondo corregido, ROIs del mismo tejido deberían acercarse.
    """
    img = np.asarray(img, dtype=float)
    suave = ndi.gaussian_filter(img, sigma=max(3, min(img.shape) / 20))

    medidas = {
        "media": float(img.mean()),
        "desvio": float(img.std()),
        "fondo_borde_centro": indicador_fondo(img)["relacion_borde_centro"],
        "rango_fondo_suave": float(np.percentile(suave, 95) - np.percentile(suave, 5)),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
    }
    if lista_rois and len(lista_rois) >= 2:
        medias = np.array([float(np.mean(datos)) for _, datos in lista_rois])
        medidas["cv_medias_rois"] = float(medias.std() / max(abs(medias.mean()), 1e-12))
    return medidas


# ------------------------------------------------------------
# Actividad 6 - Realce
# (gamma, log, sigmoidal, CLAHE, pasa-altos, pasa-banda, unsharp)
# ------------------------------------------------------------
#
# Las transformaciones de intensidad (gamma, log, sigmoidal, CLAHE)
# trabajan sobre la imagen normalizada a [0, 1] y vuelven al rango
# original, así el antes y el después son comparables.

METODOS_REALCE = [
    "Gamma",
    "Logarítmica",
    "Sigmoidal",
    "CLAHE",
    "Pasa-altos",
    "Pasa-banda",
    "Unsharp masking",
]


def _normalizar(img):
    """Lleva la imagen a [0, 1] y devuelve (normalizada, minimo, rango)."""
    img = np.asarray(img, dtype=float)
    minimo = float(img.min())
    rango = float(img.max()) - minimo
    if rango <= 0:
        raise ValueError("La región es constante: no hay nada que realzar.")
    return (img - minimo) / rango, minimo, rango


def realce_gamma(img, gamma=0.7):
    """
    s = r^gamma sobre [0, 1].
    gamma < 1 aclara y expande el contraste de las zonas oscuras;
    gamma > 1 oscurece y expande el de las zonas claras.
    """
    if gamma <= 0:
        raise ValueError("gamma debe ser positivo.")
    normalizada, minimo, rango = _normalizar(img)
    return normalizada ** gamma * rango + minimo


def realce_logaritmico(img, factor=10.0):
    """
    s = log(1 + factor·r) / log(1 + factor).
    Expande los niveles oscuros y comprime los claros; `factor` regula
    cuánto (más grande = más expansión de las sombras).
    """
    if factor <= 0:
        raise ValueError("El factor debe ser positivo.")
    normalizada, minimo, rango = _normalizar(img)
    salida = np.log1p(factor * normalizada) / np.log1p(factor)
    return salida * rango + minimo


def realce_sigmoidal(img, centro=0.5, ganancia=10.0):
    """
    s = 1 / (1 + exp(ganancia·(centro - r))), reescalada a [0, 1].
    Estira el contraste alrededor de `centro` (en fracción del rango) y
    comprime los extremos. `ganancia` es la pendiente.
    """
    normalizada, minimo, rango = _normalizar(img)
    salida = 1.0 / (1.0 + np.exp(ganancia * (centro - normalizada)))
    bajo = 1.0 / (1.0 + np.exp(ganancia * centro))
    alto = 1.0 / (1.0 + np.exp(ganancia * (centro - 1.0)))
    salida = (salida - bajo) / max(alto - bajo, 1e-12)
    return salida * rango + minimo


def realce_clahe(img, tamano_mosaico=64, limite_contraste=0.01, niveles=256):
    """
    CLAHE: ecualiza el histograma por mosaicos e interpola entre ellos.
    `limite_contraste` recorta el histograma de cada mosaico antes de
    ecualizar: es el freno a la amplificación del ruido en zonas planas.
    """
    normalizada, minimo, rango = _normalizar(img)
    salida = equalize_adapthist(
        normalizada,
        kernel_size=int(max(3, tamano_mosaico)),
        clip_limit=float(limite_contraste),
        nbins=int(niveles),
    )
    return salida * rango + minimo


def realce_pasa_altos(img, sigma=3.0, conservar_nivel=True):
    """
    Pasa-altos espacial: I - gaussiana(I, sigma). Deja los detalles finos
    y bordes; quita las variaciones lentas (incluido el fondo).
    """
    img = np.asarray(img, dtype=float)
    detalle = img - ndi.gaussian_filter(img, sigma=sigma)
    return detalle + (float(img.mean()) if conservar_nivel else 0.0)


def realce_pasa_banda(img, sigma_bajo=1.0, sigma_alto=8.0, conservar_nivel=True):
    """
    Pasa-banda por diferencia de gaussianas: gauss(σ_bajo) - gauss(σ_alto).
    Conserva las estructuras de tamaño intermedio: σ_bajo frena el ruido
    de 1 px y σ_alto quita el fondo.
    """
    if sigma_bajo >= sigma_alto:
        raise ValueError("σ bajo debe ser menor que σ alto.")
    img = np.asarray(img, dtype=float)
    banda = ndi.gaussian_filter(img, sigma_bajo) - ndi.gaussian_filter(img, sigma_alto)
    return banda + (float(img.mean()) if conservar_nivel else 0.0)


def realce_unsharp(img, sigma=2.0, cantidad=1.0, umbral=0.0):
    """
    Unsharp masking: I + cantidad·(I - gaussiana(I, sigma)).
    Con `umbral` > 0 solo se realzan los detalles cuya amplitud supera
    ese valor (en intensidad): evita amplificar el ruido de las zonas planas.
    """
    img = np.asarray(img, dtype=float)
    detalle = img - ndi.gaussian_filter(img, sigma=sigma)
    if umbral > 0:
        detalle = np.where(np.abs(detalle) > umbral, detalle, 0.0)
    return img + cantidad * detalle


def realzar(imagen, metodo="Gamma", gamma=0.7, factor_log=10.0, centro=0.5,
            ganancia=10.0, tamano_mosaico=64, limite_contraste=0.01,
            sigma_altos=3.0, sigma_bajo=1.0, sigma_alto=8.0, sigma_unsharp=2.0,
            cantidad=1.0, umbral=0.0, conservar_nivel=True, **kwargs):
    """Aplica el método de realce elegido a una imagen 2D."""
    img = a_escala_de_grises(imagen)

    if metodo == "Gamma":
        return realce_gamma(img, gamma)
    if metodo == "Logarítmica":
        return realce_logaritmico(img, factor_log)
    if metodo == "Sigmoidal":
        return realce_sigmoidal(img, centro, ganancia)
    if metodo == "CLAHE":
        return realce_clahe(img, tamano_mosaico, limite_contraste)
    if metodo == "Pasa-altos":
        return realce_pasa_altos(img, sigma_altos, conservar_nivel)
    if metodo == "Pasa-banda":
        return realce_pasa_banda(img, sigma_bajo, sigma_alto, conservar_nivel)
    if metodo == "Unsharp masking":
        return realce_unsharp(img, sigma_unsharp, cantidad, umbral)

    raise ValueError(f"Método de realce desconocido: {metodo}")


def entropia(img, niveles=256):
    """
    Entropía de Shannon del histograma (bits). Baja cuando la
    transformación junta niveles distintos en uno solo, es decir cuando
    se pierde información de intensidad.
    """
    img = np.asarray(img, dtype=float)
    conteos, _ = np.histogram(img, bins=niveles)
    p = conteos[conteos > 0] / conteos.sum()
    return float(-np.sum(p * np.log2(p)))


def medidas_realce(img, lista_rois=None, limites=None):
    """
    Valores para evaluar un realce.
      contraste_rms / rango_p1_p99: cuánto contraste hay.
      entropia: información de intensidad que sobrevive.
      sigma_wavelet: nivel de ruido (si sube, el realce lo amplificó).
      fraccion_saturados: píxeles pegados a los extremos del rango
        `limites` (el de la imagen original): ahí la información se pierde
        porque distintos niveles quedan en el mismo valor.
      snr_roi_i = media/σ de cada ROI; cnr_i_j = |μi - μj| / √((σi²+σj²)/2).
    """
    img = np.asarray(img, dtype=float)
    medidas = {
        "media": float(img.mean()),
        "contraste_rms": float(img.std()),
        "rango_p1_p99": float(np.percentile(img, 99) - np.percentile(img, 1)),
        "entropia": entropia(img),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
    }

    if limites is not None:
        bajo, alto = limites
        margen = 0.01 * (alto - bajo)
        medidas["fraccion_saturados"] = float(
            np.mean((img <= bajo + margen) | (img >= alto - margen))
        )

    if lista_rois:
        estadisticas = []
        for indice, datos in lista_rois:
            datos = np.asarray(datos, dtype=float)
            if datos.size < 2:
                continue
            media, desvio = float(datos.mean()), float(datos.std(ddof=1))
            estadisticas.append((indice, media, desvio))
            if desvio > 0:
                medidas[f"snr_roi{indice + 1}"] = media / desvio
        for (i, media_i, desvio_i), (j, media_j, desvio_j) in combinations(
            estadisticas, 2
        ):
            ruido = np.sqrt((desvio_i ** 2 + desvio_j ** 2) / 2)
            if ruido > 0:
                medidas[f"cnr_roi{i + 1}_roi{j + 1}"] = abs(media_i - media_j) / ruido

    return medidas


def comparar_realce(original, realzada):
    """
    Razones después/antes y correlación con la imagen de entrada.
    razon_sigma_wavelet > 1 significa que el ruido se amplificó;
    la correlación baja cuando el realce cambió mucho la estructura.
    """
    original = np.asarray(original, dtype=float)
    realzada = np.asarray(realzada, dtype=float)

    sigma_antes = descomposicion_wavelet(original, "db2", 1)["sigma_donoho"]
    sigma_despues = descomposicion_wavelet(realzada, "db2", 1)["sigma_donoho"]
    correlacion = 0.0
    if original.std() > 0 and realzada.std() > 0:
        correlacion = float(np.corrcoef(original.ravel(), realzada.ravel())[0, 1])

    return {
        "razon_sigma_wavelet": sigma_despues / max(sigma_antes, 1e-12),
        "razon_contraste_rms": float(realzada.std()) / max(float(original.std()), 1e-12),
        "razon_entropia": entropia(realzada) / max(entropia(original), 1e-12),
        "correlacion_con_original": correlacion,
    }


# ------------------------------------------------------------
# Actividad 7 - Evaluación
# (SIN referencia: SNR, CNR, uniformidad / CON referencia: MSE, PSNR, SSIM)
# ------------------------------------------------------------
#
# Las métricas sin referencia se calculan sobre las ROIs: SNR y
# uniformidad necesitan una región de un solo tejido, y CNR necesita
# dos regiones distintas. Las métricas con referencia comparan píxel a
# píxel contra una imagen de referencia del mismo tamaño.


def uniformidad(datos):
    """
    Uniformidad integral: 1 - (p95 - p5) / (p95 + p5).
    Vale 1 si la región es perfectamente plana y baja cuando hay
    gradiente de fondo. Se usan percentiles para que un píxel aislado
    no defina el resultado.
    """
    valores = _valores(datos)
    if valores.size < 2:
        raise ValueError("La región tiene menos de dos píxeles.")
    alto = float(np.percentile(valores, 95))
    bajo = float(np.percentile(valores, 5))
    if abs(alto + bajo) < 1e-12:
        return 0.0
    return 1.0 - (alto - bajo) / (alto + bajo)


def metricas_sin_referencia(img, lista_rois=None):
    """
    SNR, CNR y uniformidad (más contraste y ruido global).
      snr_roiN      = media / desvío de esa ROI.
      uniformidad_roiN = uniformidad integral de esa ROI.
      cnr_roiI_roiJ = |μI - μJ| / √((σI² + σJ²) / 2).
      uniformidad_entre_rois = 1 - (max - min) / (max + min) de las medias.
    """
    img = np.asarray(img, dtype=float)
    metricas = {
        "media": float(img.mean()),
        "contraste_rms": float(img.std()),
        "rango_p1_p99": float(np.percentile(img, 99) - np.percentile(img, 1)),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
        "uniformidad_global": uniformidad(img),
        "fondo_borde_centro": indicador_fondo(img)["relacion_borde_centro"],
    }

    estadisticas = []
    for indice, datos in lista_rois or []:
        datos = _valores(datos)
        if datos.size < 2:
            continue
        media, desvio = float(datos.mean()), float(datos.std(ddof=1))
        estadisticas.append((indice, media, desvio))
        if desvio > 0:
            metricas[f"snr_roi{indice + 1}"] = media / desvio
        metricas[f"uniformidad_roi{indice + 1}"] = uniformidad(datos)

    for (i, media_i, desvio_i), (j, media_j, desvio_j) in combinations(estadisticas, 2):
        ruido = np.sqrt((desvio_i ** 2 + desvio_j ** 2) / 2)
        if ruido > 0:
            metricas[f"cnr_roi{i + 1}_roi{j + 1}"] = abs(media_i - media_j) / ruido

    if len(estadisticas) >= 2:
        medias = np.array([m for _, m, _ in estadisticas])
        suma = medias.max() + medias.min()
        if abs(suma) > 1e-12:
            metricas["uniformidad_entre_rois"] = 1.0 - (
                medias.max() - medias.min()
            ) / suma

    return metricas


def metricas_con_referencia(img, referencia, rango_datos=None):
    """
    MSE, RMSE, MAE, PSNR y SSIM contra una imagen de referencia.
      MSE/RMSE/MAE: error medio píxel a píxel (0 = idénticas).
      PSNR = 10·log10(rango² / MSE) en dB: crece al achicarse el error.
      SSIM: compara luminancia, contraste y estructura en ventanas
        locales; 1 = idénticas. Devuelve también el mapa por píxel.
    `rango_datos` es el rango dinámico usado por PSNR y SSIM.
    """
    img = np.asarray(img, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    if img.shape != referencia.shape:
        raise ValueError(
            f"La imagen {img.shape} y la referencia {referencia.shape} "
            "tienen tamaños distintos."
        )

    if rango_datos is None:
        rango_datos = float(
            max(img.max(), referencia.max()) - min(img.min(), referencia.min())
        )
    rango_datos = max(float(rango_datos), 1e-12)

    error = img - referencia
    mse = float(np.mean(error ** 2))
    psnr = 10 * np.log10(rango_datos ** 2 / mse) if mse > 0 else np.inf
    ssim, mapa = structural_similarity(
        referencia, img, data_range=rango_datos, full=True
    )

    correlacion = 0.0
    if img.std() > 0 and referencia.std() > 0:
        correlacion = float(np.corrcoef(img.ravel(), referencia.ravel())[0, 1])

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "psnr_db": float(psnr),
        "ssim": float(ssim),
        "correlacion": correlacion,
        "rango_datos": rango_datos,
    }, mapa


def evaluar(imagen, referencia=None, lista_rois=None, rango_datos=None):
    """
    Métricas de la Actividad 7: siempre las que no necesitan referencia
    y, si se pasa una, también las de comparación.
    Devuelve (metricas_sin_referencia, metricas_con_referencia, mapa_ssim).
    """
    img = a_escala_de_grises(imagen)
    sin_referencia = metricas_sin_referencia(img, lista_rois)

    if referencia is None:
        return sin_referencia, {}, None

    con_referencia, mapa = metricas_con_referencia(
        img, a_escala_de_grises(referencia), rango_datos
    )
    return sin_referencia, con_referencia, mapa


