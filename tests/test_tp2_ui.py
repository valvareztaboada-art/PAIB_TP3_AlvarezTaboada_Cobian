"""Herramienta de descriptores: parámetros por defecto válidos, ROIs y pool de retinas."""

import json

import numpy as np
import pytest

from tp2 import engine
from tp2.ui import ACTIVIDADES, mascara_de_roi, parametros_por_defecto, resumen
from tp3 import rutas


@pytest.mark.parametrize("actividad", [a for a in ACTIVIDADES if a != "06_puntos"])
def test_parametros_por_defecto_ejecutan(actividad):
    """Los parámetros que muestra la interfaz son JSON válido y el motor los acepta."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (96, 96))
    roi = np.zeros(img.shape, bool)
    roi[16:80, 16:80] = True
    p = json.loads(json.dumps(parametros_por_defecto(actividad)))
    r = engine.run(actividad, img, roi, p, dict(imagen_id="x", variante="", roi_id="r"))
    assert r["features"]
    assert "DESCRIPTORES" in resumen(r)


def test_parametros_por_defecto_puntos():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (128, 128)).astype(float)
    p = parametros_por_defecto("06_puntos")
    r = engine.run("06_puntos", img, np.ones(img.shape, bool), p, {}, image_b=img)
    assert r["features"]["n_correspondencias"] > 0


class _Labels:  # el tipo se reconoce por nombre de clase, como en napari
    def __init__(self, data, name="L"):
        self.data, self.name = data, name


_Labels.__name__ = "Labels"


def test_mascara_de_roi_labels_y_completa():
    datos = np.zeros((10, 10), np.uint8)
    datos[2:4, 2:4] = 1
    datos[6:9, 6:9] = 2
    capa = _Labels(datos)
    assert mascara_de_roi((10, 10)).all()
    assert mascara_de_roi((10, 10), capa, -1).sum() == 4 + 9
    assert mascara_de_roi((10, 10), capa, 2).sum() == 9
    with pytest.raises(ValueError):
        mascara_de_roi((5, 5), capa, -1)


def test_pool_completo_y_sin_modificar():
    """El manifiesto del pool apunta a archivos presentes con el hash registrado."""
    from tp2.data import sha256_file

    m = json.loads((rutas.POOL / "manifest.json").read_text(encoding="utf-8"))
    assert len(m["images"]) == 13
    for it in m["images"]:
        assert sha256_file(rutas.POOL / it["path"]) == it["sha256"], it["path"]
