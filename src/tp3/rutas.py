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
# Espacio de trabajo de los descriptores por lotes; su data/ es el pool de retinas de los TPs anteriores.
ESPACIO_TP2 = RAIZ / "espacio_tp2"
POOL_TP1 = ESPACIO_TP2 / "data"

CONFIG = RAIZ / "config"
MANIFEST = DATA / "manifest.csv"
SPLITS = CONFIG / "splits.json"
PREPROCESO = CONFIG / "preproceso.json"
LANDMARKS = CONFIG / "landmarks.csv"

MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
CSV_RESULTADOS = RESULTADOS / "resultados_tp3.csv"
