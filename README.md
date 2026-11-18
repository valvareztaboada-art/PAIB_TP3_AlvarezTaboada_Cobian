# TP3 · Clasificación y segmentación de bioimágenes: del Machine Learning clásico a la familia U-Net

**Autores:** Victoria Sol Álvarez Taboada y Tomás Cobián
**Asignatura:** 16.85 — Procesamiento Avanzado de Imágenes en Biología y Biomedicina, 2.º cuatrimestre de 2026
**Profesor:** Roberto Sebastián Tomás

## Objetivo

Segmentar vasos y clasificar retinas (sana / glaucoma / retinopatía diabética) en la base HRF con
ML clásico y con Deep Learning, usando el mismo split, el mismo preprocesamiento y las mismas
métricas para ambos enfoques.

## Datos

Base HRF (High-Resolution Fundus Image Database), subconjunto de segmentación: 45 retinografías
de 3504 × 2336 px (15 sanas, 15 con glaucoma y 15 con retinopatía diabética), con segmentación
manual de vasos y máscara de FOV. Se usa además el archivo de centros del disco óptico marcados
por dos expertos que publica la misma base.

> Budai A., Bock R., Maier A., Hornegger J., Michelson G. *Robust Vessel Segmentation in Fundus
> Images*. International Journal of Biomedical Imaging, 2013. Licencia CC-BY 4.0.

Los datos no se incluyen en el repositorio. Para descargarlos:

```powershell
curl -L -o all.zip https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
mkdir data/HRF/raw
tar -xf all.zip -C data/HRF/raw            # crea images/, manual1/ y mask/
curl -L -o data/HRF/raw/optic_disk_centers.xls https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/optic_disk_centers.xls
```

`data/manifest.csv` registra el SHA-256 de cada archivo, de modo que se puede comprobar que se
trabaja con la misma base.

Las retinas de los TPs anteriores (`espacio_tp2/data/`) no se usan para entrenar: son un conjunto
externo para evaluar robustez frente a degradaciones.

## Reproducción

```powershell
pixi install
pixi run check-tp3     # verifica el entorno y los módulos del proyecto
pixi run manifest      # verifica las 45 imágenes y escribe data/manifest.csv
pixi run splits        # escribe config/splits.json (semilla 42)
pixi run test
pixi run widget        # abre napari con el widget del TP3
```

## Widget

Un único visor de napari. El panel lateral agrupa las herramientas por la tarea que resuelven, en
solapas con secciones plegables:

| Solapa | Secciones |
|---|---|
| Datos | Base HRF · ROIs de medición · Regiones persistentes |
| Calidad | Medición en ROIs · Diagnóstico de ruido y artefactos · Estimación de ruido · Métricas de calidad antes/después |
| Preproceso | Escala de grises y recorte · Reducción de ruido · Deconvolución · Artefactos periódicos (notch) · Corrección de fondo · Realce · Degradaciones controladas |
| Descriptores | Cálculo por lotes (intensidad, forma, estructura vascular, textura, multiescala, puntos de interés, radiomics) · Máscaras vasculares · Feature maps por píxel |
| Segmentar | ML clásico (k-NN, SVM, Random Forest) · Deep Learning (U-Net) |
| Clasificar | ML clásico (k-NN, SVM, Random Forest) · Deep Learning (CNN) |
| Resultados | Registro del pipeline y CSV · Resultados de descriptores · Comparación ML vs DL · Robustez frente a degradaciones |

Debajo del visor quedan fijos el informe de texto de cada operación y el progreso de los cálculos
por lotes. El visor abre vacío: el pool de retinas y sus regiones se cargan desde
Datos › Regiones persistentes, o automáticamente al usar una herramienta que los necesita. Las herramientas de calidad y preprocesamiento guardan en `resultados/tp1/`; las de
descriptores trabajan sobre `espacio_tp2/`.

## División de los datos

La división se hace por imagen completa (cada imagen es un sujeto distinto): ningún píxel ni
parche de una imagen de validación o test se usa para entrenar. `config/splits.json` es el único
archivo de división y lo usan tanto los modelos clásicos como los de Deep Learning.

| Conjunto | Imágenes | Uso |
|---|---|---|
| Test | 9 (3 por grupo) | Evaluación final, una sola vez |
| Desarrollo | 36, en 5 folds estratificados por grupo | Validación cruzada en clasificación |
| ↳ Entrenamiento de segmentación | 28 (folds 1–4) | Entrenamiento |
| ↳ Validación de segmentación | 8 (fold 0) | Hiperparámetros, umbral, postproceso, época de parada |

El archivo también fija subconjuntos anidados de 3, 6 y 10 imágenes de entrenamiento para el
análisis de eficiencia en datos.

## Estructura

| Ruta | Contenido |
|---|---|
| `widget_tp3.py` | Lanzador del widget |
| `src/tp1/core.py` | Funciones de calidad y preprocesamiento del TP1 (diagnóstico, filtrado, restauración, fondo, realce, métricas) |
| `src/tp1/ui.py` | Bloques de interfaz de calidad y preprocesamiento |
| `src/tp2/` | Descriptores del TP2 (medidas, estructura vascular, textura, multiescala, puntos, radiomics) y su interfaz por lotes (`batch_ui.py`) |
| `espacio_tp2/` | Imágenes, regiones persistentes, configuración y resultados de los descriptores por lotes |
| `src/tp3/datos.py` | Carga de HRF, grupos diagnósticos y centros del disco óptico |
| `src/tp3/splits.py` | División por imagen y su validación |
| `src/tp3/ml/` | ML clásico: feature maps, muestreo, segmentación y clasificación |
| `src/tp3/dl/` | Deep Learning: dataset de parches, U-Net, CNN, entrenamiento e inferencia |
| `src/tp3/ui/` | Ventana con solapas por función (`ventana.py`) y secciones plegables (`secciones.py`) |
| `scripts/` | Pasos del pipeline, numerados |
| `config/` | División de los datos y parámetros de preprocesamiento |
| `modelos/` | Pesos entrenados y su configuración |
| `resultados/` | CSV acumulativo de experimentos y figuras |
| `tests/` | Pruebas automatizadas |

## Entorno

Python 3.10 con napari 0.5, PySide6, scikit-learn y PyTorch, fijado en `pixi.toml` / `pixi.lock`.

- `pyradiomics` sólo publica binarios hasta Python 3.9 y no puede instalarse en este entorno.
  La GLCM se calcula con una implementación propia con las mismas definiciones (discretización
  fija, matriz simétrica, desplazamientos con distancia de Chebyshev d, promedio por dirección),
  que reproduce los valores de PyRadiomics 3.1.0 con error relativo < 1e-12. Las familias GLRLM,
  GLSZM, GLDM y NGTDM quedan sin calcular (NaN) en el cálculo de descriptores por lotes.
- Los modelos de Deep Learning se entrenan con GPU fuera del widget; el widget carga los pesos
  para la inferencia.
