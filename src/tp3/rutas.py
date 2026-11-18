"""Rutas del proyecto. Todo módulo las toma de acá para no repetir cadenas."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

DATA = RAIZ / "data"
HRF_RAW = DATA / "HRF" / "raw"
HRF_IMAGENES = HRF_RAW / "images"
HRF_MANUAL = HRF_RAW / "manual1"
HRF_FOV = HRF_RAW / "mask"
HRF_DISCO_XLS = HRF_RAW / "optic_disk_centers.xls"
PROCESADO = DATA / "procesado"
# Retinas de los TPs anteriores: conjunto externo de evaluación (no se entrena con ellas).
POOL = DATA / "pool"

CONFIG = RAIZ / "config"
MANIFEST = DATA / "manifest.csv"
SPLITS = CONFIG / "splits.json"
PREPROCESO = CONFIG / "preproceso.json"
LANDMARKS = CONFIG / "landmarks.csv"

MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
CSV_RESULTADOS = RESULTADOS / "resultados_tp3.csv"
# Un CSV por familia de descriptores y un registro reproducible por medición.
RESULTADOS_DESCRIPTORES = RESULTADOS / "descriptores"
