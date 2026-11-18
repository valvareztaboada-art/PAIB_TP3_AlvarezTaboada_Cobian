"""Carga de la base HRF (subconjunto de segmentación) a resolución nativa.

Identificador de imagen: el nombre sin extensión del archivo de HRF
("01_h", "07_g", "12_dr"). Cada imagen corresponde a un sujeto distinto,
por lo que imagen = sujeto = unidad de split.
"""

import hashlib

import numpy as np
from PIL import Image

from . import rutas

# Orden fijo de clases: lo usan ML y DL para las columnas de probabilidad.
CLASES = ("sano", "glaucoma", "rd")
SUFIJO_A_CLASE = {"h": "sano", "g": "glaucoma", "dr": "rd"}
CLASE_A_INDICE = {c: i for i, c in enumerate(CLASES)}


def grupo_de(id_imagen):
    """'03_dr' -> 'rd'."""
    return SUFIJO_A_CLASE[id_imagen.split("_", 1)[1]]


def listar_ids(carpeta=None):
    """Los 45 identificadores ordenados por grupo y número."""
    carpeta = carpeta or rutas.HRF_IMAGENES
    ids = [p.stem for p in carpeta.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    orden = {s: i for i, s in enumerate(SUFIJO_A_CLASE)}
    return sorted(ids, key=lambda i: (orden[i.split("_", 1)[1]], int(i.split("_")[0])))


def ruta_imagen(id_imagen):
    # Las imágenes de RD tienen extensión .JPG en mayúscula.
    for ext in (".jpg", ".JPG", ".jpeg"):
        p = rutas.HRF_IMAGENES / f"{id_imagen}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No existe la imagen {id_imagen}")


def ruta_manual(id_imagen):
    return rutas.HRF_MANUAL / f"{id_imagen}.tif"


def ruta_fov(id_imagen):
    return rutas.HRF_FOV / f"{id_imagen}_mask.tif"


def _leer(p):
    # PIL lee los TIFF con compresión LZW sin depender de imagecodecs.
    with Image.open(p) as im:
        return np.asarray(im)


def _binaria(a):
    """Máscaras de HRF: 0/255, a veces en RGB. Devuelve bool 2D."""
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    return a > 0


def cargar_rgb(id_imagen):
    """RGB uint8 (2336, 3504, 3)."""
    return _leer(ruta_imagen(id_imagen))


def cargar_gt(id_imagen):
    """Gold standard de vasos como bool (2336, 3504)."""
    return _binaria(_leer(ruta_manual(id_imagen)))


def cargar_fov(id_imagen):
    """Máscara del campo de visión como bool (2336, 3504)."""
    return _binaria(_leer(ruta_fov(id_imagen)))


def cargar(id_imagen):
    """(rgb, gt, fov) a resolución nativa."""
    return cargar_rgb(id_imagen), cargar_gt(id_imagen), cargar_fov(id_imagen)


def sha256(path, bloque=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bloque), b""):
            h.update(b)
    return h.hexdigest()


def leer_centros_disco(tolerancia_rel=0.25):
    """Centro y diámetro del disco óptico según los dos expertos de HRF.

    El .xls tiene una hoja por experto ("Expert A", "Expert B") con columnas
    image, Pap. Center x/y, vessel orig. x/y, disk diameter, en px nativos
    (x = columna, y = fila). Los ids difieren entre hojas ("01_dr" / "01dr").

    Devuelve {id: dict} con los valores de cada experto, el promedio y una
    marca `disco_discrepancia` cuando los centros se separan más de
    `tolerancia_rel` × diámetro (hay errores de tipeo en el archivo, p. ej.
    experto B en 02_g con y = 110). En ese caso NO se promedia: se deja el
    centro del experto A y se revisa visualmente en napari.
    """
    import xlrd

    libro = xlrd.open_workbook(str(rutas.HRF_DISCO_XLS))
    por_experto = {}
    for hoja, clave in zip(libro.sheets(), ("A", "B")):
        for r in range(1, hoja.nrows):
            fila = hoja.row_values(r)
            if not fila[0]:
                continue
            nombre = str(fila[0]).strip().replace("_", "")
            num, suf = nombre[:2], nombre[2:]
            por_experto.setdefault(f"{num}_{suf}", {})[clave] = dict(
                x=float(fila[1]), y=float(fila[2]), d=float(fila[5])
            )
    salida = {}
    for i, e in por_experto.items():
        a, b = e.get("A"), e.get("B")
        reg = {}
        for k, v in (("A", a), ("B", b)):
            if v:
                reg.update({f"disco_{k}_fila": v["y"], f"disco_{k}_col": v["x"], f"disco_{k}_diam": v["d"]})
        if a and b:
            sep = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))
            diam = (a["d"] + b["d"]) / 2
            discrepa = sep > tolerancia_rel * diam
            reg.update(
                disco_separacion_expertos_px=sep,
                disco_discrepancia=bool(discrepa),
                disco_fila=a["y"] if discrepa else (a["y"] + b["y"]) / 2,
                disco_col=a["x"] if discrepa else (a["x"] + b["x"]) / 2,
                disco_diam=a["d"] if discrepa else diam,
            )
        else:
            v = a or b
            reg.update(disco_fila=v["y"], disco_col=v["x"], disco_diam=v["d"], disco_discrepancia=False)
        salida[i] = reg
    return salida
