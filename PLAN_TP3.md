# TP3 — Plan de trabajo y decisiones previas al código

Alvarez Taboada (Act. 1–4) · Cobián (Act. 5–8)

Este documento fija **qué** vamos a hacer antes de escribir código. Las decisiones marcadas
**[verificar]** son provisorias y se confirman con una medición en la Actividad 1; todo lo demás
queda fijo salvo acuerdo entre los dos.

---

## 0. Relevamiento del material

| Material | Estado | Qué aporta al TP3 |
|---|---|---|
| Clases I–VIII | Leídas (slides son mayormente imagen; se revisaron las de la clase VII sobre leakage, desbalance, anotaciones y postproceso) | Clase VII = guía directa de Act. 1–4; clase VIII = Act. 5–6 |
| Informe TP1 | Leído | Parámetros de degradaciones y correcciones (ver §4.3) |
| Informe TP2 (`PAIBB_TPs.pdf`) | Leído | Configuraciones de LBP, Gabor, Wavelet, LoG/DoG, Frangi, esqueleto |
| Consigna TP3 | Leída | — |
| `pixi.toml` | Leído | Es el entorno de la cátedra: Python 3.10, napari 0.5, PySide6, PyTorch, sklearn |
| `widget_tp1.py` | Leído (5086 líneas) | Funciones puras de diagnóstico, filtrado, fondo, realce y métricas |
| `widget_tp2.py` | **Es sólo el lanzador** | El código real está en `src/tp2/` del repo `tcobian831/PAIB_TP2_AlvarezTaboada_Cobian` (se clonó para revisarlo) |

### Problemas detectados que hay que resolver antes de empezar

1. **El código del TP2 no está en la carpeta.** Hay que copiar `src/tp2/` al proyecto del TP3.
   Las funciones que vamos a reutilizar: `texture.multiscale`, `texture.gabor_bank`,
   `texture.lbp_features`, `texture.wavelet_features`, `structure.segment_vessels` (Frangi),
   `structure.vascular` (esqueleto, bifurcaciones, extremos, calibre, tortuosidad), `measure.intensity`.
2. **Entornos incompatibles.** El TP2 usaba Python 3.9 + PyQt5 + `pyradiomics==3.1.0`; el `pixi.toml`
   de la cátedra usa Python 3.10 + PySide6 y no trae pyradiomics.
   - El lanzador del TP2 fuerza `QT_API=pyqt5` → hay que sacarlo.
   - `tp2/radiomics.py` importa `radiomics` → probar si `pyradiomics` instala en py3.10/Windows;
     si no, GLCM con `skimage.feature.graycomatrix` (ya está en el entorno) con la misma
     discretización del TP2 (32 niveles, distancia 1).
3. **`widget_tp1.py` crea el `napari.Viewer()` al importarse (línea 1971).** No se puede importar
   para reutilizar funciones. Hay que separar las líneas 1–~1966 (funciones puras) en `tp1/core.py`
   y dejar la interfaz aparte.
4. **El TP1 no tiene funciones que generen degradaciones** (las imágenes venían degradadas).
   Para Act. 5 (augmentation) y Act. 8 hay que escribirlas (`degradaciones.py`), calibradas con lo
   que diagnosticamos en el TP1.
5. **El pool del TP1 es otra escala y otro dominio**: 539×539 px, gris, recorte macular sin borde
   de FOV. Estimación a ojo (Retina1: disco–fóvea ≈ 265 px) → equivale a HRF submuestreado ≈ ×4.
   **[verificar]** midiendo la distancia disco–fóvea en HRF.
6. **HRF no está descargado.** Ver §6.
7. **La carpeta no es un repositorio git.** Siendo dos personas trabajando en paralelo sobre
   módulos compartidos, conviene crear el repo antes de escribir código.

**Conclusión:** el TP es realizable con el entorno de la cátedra. Lo único que puede costar es
pyradiomics (hay alternativa) y el tiempo de inferencia de k-NN/SVM sobre imágenes completas (§3).

---

## 1. Decisiones generales (valen para ML y DL)

La consigna exige que ML y DL resuelvan **las mismas tareas, con los mismos conjuntos y las mismas
métricas**. Por eso todo lo de esta sección se implementa una vez, en módulos compartidos, y ambos
enfoques lo llaman.

### 1.1 Las dos tareas con la secuencia que pide la consigna

| | Segmentación de vasos | Clasificación diagnóstica |
|---|---|---|
| ¿Qué quiero decidir? | ¿Este píxel es vaso o fondo? | ¿Esta retina es sana, glaucoma o RD? |
| Unidad de muestra | Píxel (ML) / parche (DL) | Imagen = sujeto |
| Unidad de split | **Imagen** (siempre) | **Imagen** |
| X | Stack de feature maps por píxel (ML) / parche de imagen (DL) | Vector de descriptores por imagen (ML) / imagen o parches (DL) |
| y | Gold standard HRF dentro del FOV | Sufijo del archivo: `_h`, `_g`, `_dr` |
| Desbalance | Fuerte (~10 % vaso, **[verificar]**) | Ninguno (15/15/15) |
| Métricas | Dice, IoU, Se, Sp por imagen dentro del FOV; media ± desvío; por grupo | Matriz de confusión, Se, Sp, precisión, F1 por clase |

### 1.2 Representación común de la imagen (Act. 1, la usan todos)

Orden fijo y justificado:

1. **Canal verde** (mayor contraste vaso/fondo). Se justifica con números: CNR vaso vs. fondo en R, G, B
   y gris usando el gold standard. Relación con el pool del TP1: el pool es gris; la normalización
   del paso 4 reduce esa diferencia, y en Act. 8 se discute como cambio de dominio.
2. **Submuestreo ×f** con anti-aliasing (promedio por área). **Provisorio f = 2** → 1752×1168 px.
   **[verificar]** con el análisis de calibre del §2.1.
   - Gold standard: fracción de área ≥ 0,5 tras submuestrear.
   - FOV: vecino más cercano + **erosión de unos px** (el borde del FOV produce respuestas fuertes en
     todos los filtros y no es vaso).
3. **Corrección de fondo** (opcional, se decide en validación): kernel gaussiano grande por resta,
   como en Retina6 del TP1 (σ = 60 a escala pool ≈ σ = 120 a f = 2).
4. **Normalización por imagen dentro del FOV**: percentiles 1–99 → [0, 1] (o z-score robusto). Corrige
   diferencias de ganancia/iluminación entre retinas.
5. (Opcional) CLAHE, con los parámetros del TP1 escalados. Se prueba en validación, no por defecto.
6. Relleno de la zona fuera del FOV (reflejo / valor medio del borde) antes de filtrar, para que
   los filtros no "vean" el borde negro.

Los arrays resultantes se **cachean en disco** (`data/procesado/f2/<id>_{img,gt,fov}.tif`) y tanto ML
como DL leen de ahí. Esto garantiza que comparamos modelos, no preprocesamientos distintos.

**La evaluación se hace a la resolución de trabajo** (f = 2) para ambos enfoques.

### 1.3 Split (Act. 2) — un único archivo `config/splits.json`, semilla 42

```
45 imágenes (15 h / 15 g / 15 dr)
├── TEST: 9 imágenes (3 por grupo) — se usa UNA vez, al final, para ML y DL
└── DESARROLLO: 36 imágenes (12 por grupo) → 5 folds estratificados por grupo (7–8 imágenes c/u)
      ├── Segmentación:  fold 0 = validación, folds 1–4 = entrenamiento (≈ 28 imágenes)
      └── Clasificación: validación cruzada 5-fold sobre las 36
```

Además, en el mismo archivo se fijan los **subconjuntos anidados para eficiencia en datos** (Act. 7):
3 (1/grupo) ⊂ 6 (2/grupo) ⊂ 10 ⊂ todas las de entrenamiento. Así ML y DL usan exactamente las mismas.

Reglas:
- Escalado, selección de features y PCA dentro de un `sklearn.Pipeline` → se ajustan sólo con train.
- Validación para hiperparámetros, umbral, postproceso y época de parada.
- Ningún píxel ni parche de una imagen de test/val se usa para entrenar.

**Decisión pendiente (A):** test de 9 imágenes (3/grupo) vs. 15 (5/grupo). Con 9, cada error en
clasificación vale 11 puntos; con 15, el desarrollo queda en 30. Recomiendo **9**, y reportar el test
de clasificación con esa advertencia explícita.

### 1.4 Contrato entre ML y DL (lo que hace que 5–8 se pueda construir sobre 1–4)

Todo modelo de segmentación, sea RF o U-Net, entrega:

```python
prob: np.ndarray  # float32, HxW, [0,1], a resolución f, 0 fuera del FOV
```

y a partir de ahí **todo es compartido**:

| Función compartida | Módulo | La usa |
|---|---|---|
| `cargar_preprocesada(id)` | `datos.py` | Act. 1–8 |
| `leer_split()` | `splits.py` | Act. 2–8 |
| `elegir_umbral(probs_val, gts_val, fovs_val)` → máx. Dice medio en validación | `postproceso.py` | Act. 3, 5 |
| `postprocesar(prob, umbral, r_apertura, r_cierre, min_obj)` | `postproceso.py` | Act. 3, 5, 7, 8 |
| `metricas_segmentacion(mask, gt, fov)` → Dice, IoU, Se, Sp, accuracy | `metricas.py` | Act. 3, 5, 7, 8 |
| `mapa_tp_fp_fn(mask, gt, fov)` → Labels 1/2/3 | `metricas.py` | Act. 7 |
| `entropia(prob)` | `metricas.py` | Act. 7 |
| `descriptores_vasculares(mask, fov)` (envuelve `tp2.structure.vascular`) | `descriptores.py` | Act. 4, 7 |
| `metricas_clasificacion(y, y_pred)` | `metricas.py` | Act. 4, 6, 7 |
| `degradar(img, tipo, **params)` | `degradaciones.py` | Act. 5, 8 |
| `corregir_tp1(img, tipo)` | `tp1/core.py` | Act. 1, 8 |
| `registrar_experimento(dict)` → CSV acumulativo | `registro.py` | Todas |

Los modelos de clasificación entregan `probas` de forma `(n_imagenes, 3)` con el orden de clases
fijo `["sano", "glaucoma", "rd"]`.

Cada modelo entrenado se guarda con un JSON al lado (`modelo.joblib` / `modelo.pt` + `modelo.json`)
que registra: hash del split, preprocesamiento, features, hiperparámetros, umbral y postproceso.
El widget carga ambos y no necesita saber cómo se entrenó.

### 1.5 CSV de resultados — `resultados/resultados_tp3.csv`

Una fila por **experimento × imagen evaluada**, más una fila `imagen_id = "AGREGADO"` con media y
desvío (mismo criterio que la fila "pipeline" del TP1). Columnas: las que lista la consigna,
con los campos compuestos (split, hiperparámetros, features) serializados en JSON y un
`experimento_id`. La función `registrar_experimento` la escribimos al principio y la usan los dos.

### 1.6 Estructura del proyecto

```
PAIB_TP3/
├── pixi.toml
├── _referencia/              # PDFs y widgets viejos: sólo consulta local, fuera de git
├── widget_tp3.py             # lanzador único: napari con solapas por función (ver decisión G)
├── config/  splits.json  preproceso.json  landmarks.csv
├── data/    HRF/ (crudo, fuera de git)  procesado/
├── espacio_tp2/              # config, ROIs, resultados y data/ (= pool de retinas) de los descriptores por lotes
├── src/
│   ├── tp1/core.py           # funciones puras del TP1
│   ├── tp1/ui.py             # solapa TP1 (widget del TP1; toma napari.current_viewer())
│   ├── tp2/                  # paquete completo del TP2; batch_ui.Panel se reparte en solapas
│   └── tp3/
│       ├── datos.py splits.py degradaciones.py postproceso.py metricas.py registro.py descriptores.py
│       ├── ml/  features.py  muestreo.py  segmentacion.py  clasificacion.py   ← Victoria
│       ├── dl/  dataset.py  unet.py  cnn.py  entrenar.py  inferencia.py       ← Tomás
│       └── ui/  ventana.py  secciones.py  (+ un módulo por herramienta nueva)
├── scripts/                  # entrenamientos por línea de comando / notebooks Colab
├── modelos/                  # pesos + json
└── resultados/               # CSV, figuras, labels
```

Criterio heredado del TP1: **funciones puras separadas de la interfaz**. Lo pesado (búsqueda de
hiperparámetros, entrenar U-Net) corre en scripts; el widget entrena lo liviano (p. ej. RF con trazos)
y carga modelos para inferir y visualizar.

---

## 2. Actividades 1–4 en detalle

### 2.1 Actividad 1 — Definición del problema y preparación de datos

| Paso | Qué se hace | Qué se reporta / justifica |
|---|---|---|
| Inventario | Leer 45 imágenes + gold standard + FOV; verificar tamaños, tipos y que las máscaras sean binarias; asignar grupo por sufijo; hash SHA-256 (como en el TP2) | `data/manifest.csv` |
| Canal | CNR vaso/fondo con el gold standard en R, G, B, gris | Tabla → se elige verde |
| Factor de submuestreo | Sobre el gold standard a resolución nativa: (a) distribución de calibres = 2·EDT sobre el esqueleto; (b) % de longitud de esqueleto que sobrevive al submuestrear ×1, ×2, ×3, ×4 | Elegir el f más grande que conserve los vasos finos con ≥ ~2 px de calibre. Se relaciona con la escala del pool (≈ ×4) |
| Normalización / preproceso TP1 | Percentiles dentro del FOV; corrección de fondo y CLAHE como opciones a decidir en validación (Act. 3) | Se documenta el orden |
| FOV | Erosión de k px; no se entrena ni se evalúa fuera | Cuántos px se pierden |
| Desbalance | % de vaso dentro del FOV por imagen y por grupo | Tabla + boxplot por grupo |
| Napari | Imagen + `Labels` gold standard + `Labels` FOV, con selector de imagen | Capturas |

### 2.2 Actividad 2 — Validación sin data leakage

- Generar `splits.json` (semilla 42, estratificado por grupo) con test, folds y subconjuntos anidados.
- Registrar la lista de imágenes por conjunto en el CSV.
- **Experimento de leakage**: con el mismo RF y las mismas features, comparar
  (a) split por imagen (val = fold 0) vs. (b) píxeles de las 36 imágenes mezclados al azar 80/20.
  Se espera que (b) dé un Dice inflado: píxeles vecinos comparten casi las mismas features (los
  filtros de σ = 4–8 px solapan) y el modelo ve la misma retina en train y en test. Se reporta la
  diferencia. Es barato con RF; Tomás puede repetirlo con parches de la U-Net si hay tiempo.

### 2.3 Actividad 3 — ML clásico para segmentación de píxeles

**Stack de features (≈ 35–45 mapas), reutilizando TP2:**

| Familia | Parámetros (a f = 2) | Función |
|---|---|---|
| Intensidad normalizada | — | — |
| Gauss | σ = 1, 2, 4, 8 | `tp2.texture.multiscale` |
| LoG normalizado / DoG | σ = 1, 2, 4, 8; k = 2^(1/3) | `tp2.texture.multiscale` |
| Magnitud del gradiente | σ = 1, 2, 4 | nueva (gaussian_gradient_magnitude) |
| Gabor | λ = 4, 8, 16 px × 4 orientaciones (+ máximo sobre orientaciones por λ) | `tp2.texture.gabor_bank` (devuelve los mapas) |
| LBP | (8,1), (16,2) uniform, promediado localmente | `tp2.texture.lbp_features` (devuelve el mapa de códigos) |
| Media y desvío locales | ventanas 5 y 11 | nueva |
| Frangi / autovalores de Hessiana | σ = 1, 2, 3 | `tp2.structure.segment_vessels` (devuelve la respuesta) |

Advertencias de reutilización: las funciones del TP2 piden una máscara (pasar el FOV) y `gabor_bank`
divide por 255 (pasarle la imagen en esa escala). El código LBP es categórico: para k-NN/SVM no tiene
sentido como distancia, por eso se usa suavizado localmente.

**Construcción de (X, y):**
- Por imagen de entrenamiento: muestreo **balanceado 1:1** dentro del FOV erosionado (p. ej. 2000 vaso
  + 2000 fondo; la mitad del fondo cerca de los vasos, a 1–5 px, para incluir los casos difíciles).
- Consecuencia que hay que explicar: con muestreo balanceado la probabilidad queda calibrada a un prior
  50/50 → **el umbral 0,5 no es el correcto**, se elige en validación (máximo Dice medio).
- Costo: RF con ~110k muestras es rápido. **k-NN y SVM-RBF no escalan a predecir ~1,4 M píxeles
  por imagen**: se entrenan con un submuestreo (~20k) y la búsqueda de hiperparámetros se evalúa
  sobre una muestra de píxeles de validación; la imagen completa sólo para el mejor de cada familia.
  Se registra el tiempo de inferencia (la consigna lo pide y es parte de la comparación).

**a) Máscara completa vs. b) trazos en Napari:**
- (b): capa `Labels` donde se pintan 1 = vaso, 2 = fondo sobre 2–3 imágenes de entrenamiento.
- Además una **curva de anotación**: Dice en validación vs. número de píxeles anotados (100, 500, 2k,
  10k… muestreados del gold standard) para responder "¿cuánta anotación necesita?". Sesgo esperable
  de los trazos: se anotan vasos gruesos y centrales → peor sensibilidad en vasos finos.

**Modelos** (`Pipeline(StandardScaler → clf)`), elección en validación con PR-AUC (no depende del umbral):

| Modelo | Grilla |
|---|---|
| k-NN | k ∈ {5, 15, 31, 61}; pesos uniformes / por distancia |
| SVM | kernel ∈ {lineal, RBF}; C ∈ {0,1; 1; 10}; γ = "scale" |
| Random Forest | árboles ∈ {100, 300}; profundidad ∈ {10, 20, None}; min_samples_leaf ∈ {1, 5} |

Por qué escalar: k-NN y SVM usan distancias/productos internos y una feature con rango grande domina;
los árboles cortan cada feature por separado, son invariantes a transformaciones monótonas.

**De probabilidad a máscara:** mapa de probabilidad (capa `Image`) → umbral elegido en validación →
apertura/cierre (radio 0–1) → eliminar objetos pequeños (tamaño mínimo elegido en validación) → capa
`Labels`. Orden justificado como en la clase VII.

**Selección y reducción:** varianza, correlación (|ρ| > 0,95), importancia RF y por permutación
(calculada en validación), PCA (95 % de varianza). Comparar completo / subconjunto / PCA, siempre
ajustado dentro del Pipeline en train.

**No supervisado de referencia:** k-means (k = 2) y GMM sobre la intensidad sola y sobre el espacio
multicaracterística escalado; el cluster "vaso" se asigna por intensidad media más baja.

### 2.4 Actividad 4 — ML clásico para clasificación de imágenes

**Vector por imagen** (todos normalizados por el área del FOV cuando dependen del tamaño):
- Vasculares (`tp2.structure.vascular`): densidad vascular, longitud del esqueleto, bifurcaciones y
  extremos por unidad de área, calibre medio/mediano (2·EDT sobre el esqueleto), tortuosidad
  media/mediana de ramas con longitud mínima (las ramas muy cortas dan tortuosidades sin sentido).
- Textura sobre el canal verde: LBP (entropía, uniformidad), energías Gabor, energías Wavelet por
  subbanda, GLCM (32 niveles, distancia 1, como en el TP2).
- Por ROI: **peripapilar**, **macular** y **periferia**.

**Decisión pendiente (B): cómo ubicar disco óptico y fóvea** (HRF no los trae). Recomiendo marcar
disco y fóvea a mano con una capa `Points` en Napari (90 clics, guardados en `config/landmarks.csv`,
reproducible) con una propuesta automática como punto de partida (disco = zona más brillante
suavizada). ROIs: anillo peripapilar entre 1 y 2 radios de disco, disco macular de ~1 diámetro de
disco centrado en la fóvea, periferia = resto del FOV.

**Dos versiones de los descriptores vasculares:**
1. Desde el gold standard.
2. Desde la segmentación predicha por el mejor modelo de la Act. 3.
   **Ojo con el leakage:** si el modelo de Act. 3 se entrenó con esas imágenes, sus máscaras son
   demasiado buenas y los descriptores no se parecen a los que tendremos en test. Solución: máscaras
   **fuera de fold** — para cada uno de los 5 folds se reentrena el segmentador (RF, barato) con los
   otros 4 y se predicen las imágenes del fold. Para test se usa el modelo final.
   Se compara cada descriptor GT vs. predicho (correlación / Bland-Altman) y el desempeño de la
   clasificación con cada versión.

**Entrenamiento y evaluación:**
- `Pipeline(StandardScaler → SelectKBest/importancia → clf)` con k-NN, SVM, RF y un `DummyClassifier`
  como línea base (33 %).
- Con p ≫ n (≈ 60–100 features para 36 imágenes), la selección **va dentro de cada fold**.
- Hiperparámetros: grilla chica con CV interna de 3 folds dentro de cada fold externo (CV anidada).
- Reportar por fold y media ± desvío; repetir el 5-fold con varias semillas para ver cuánto varía.
- "¿Memoriza?": comparar desempeño en train vs. validación por fold (un RF llega a ~100 % en train).
- Importancia por permutación + discusión clínica (glaucoma → región peripapilar; RD → lesiones,
  calibre y tortuosidad, textura macular).
- Entrenar con las 36 y evaluar **una vez** en test.

---

## 3. Riesgos de cómputo

| Riesgo | Mitigación |
|---|---|
| Stack de 40 features × 2 M px ≈ 300 MB por imagen | Calcular por imagen, guardar sólo las muestras; float32 |
| Inferencia k-NN / SVM-RBF en imagen completa | Submuestreo para buscar hiperparámetros; imagen completa sólo para el mejor; reportar tiempos |
| `tp2.structure.skeleton_graph` es Python puro | Probar tiempo en 1 imagen; `calculate_thickness=False` si hace falta |
| U-Net en CPU | Colab con GPU; el widget sólo carga pesos (lo permite la consigna) |

---

## 4. Lo que Act. 5–8 necesita de Act. 1–4 (y viceversa)

### 4.1 Entregables de Victoria que Tomás usa

- `splits.json` con test, folds y subconjuntos anidados.
- Imágenes preprocesadas cacheadas y `preproceso.json`.
- `metricas.py`, `postproceso.py`, `registro.py`, `descriptores.py`.
- `degradaciones.py` (lo arma quien llegue primero; lo usan Act. 5 y 8).
- Mejor modelo ML de segmentación y de clasificación guardados con su JSON (Act. 7–8).
- Funciones de entrenamiento ML que acepten una lista de IDs (para la curva de eficiencia de Act. 7).

### 4.2 Lo que Tomás tiene que respetar

- Mismo split, mismo preprocesamiento, misma resolución de evaluación.
- La U-Net devuelve `prob` en el mismo formato y pasa por el mismo `elegir_umbral` + `postprocesar`.
- Mismas filas/columnas en el CSV.

### 4.3 Degradaciones calibradas con el TP1

Los parámetros están medidos a la escala del pool (≈ HRF ×4); a f = 2 los parámetros espaciales se
multiplican por ~2 **[verificar escala]**.

| Degradación | Diagnóstico TP1 | Corrección TP1 |
|---|---|---|
| Ruido gaussiano | σ ≈ 7,2 (wavelet), aditivo y estacionario (Retina3) | Variación total, peso 0,04 |
| Ruido impulsivo | 0,8 % de píxeles (Retina7) | Mediana 3×3 |
| Artefacto periódico | rayas verticales, período ≈ 23 px (Retina5) | Notch gaussiano radio 5 |
| Desenfoque | PSF gaussiana σ ≈ 1,5–2 (Retina4) | Wiener, balance 0,1 |
| Iluminación no uniforme | borde/centro 0,72, contraste RMS 15,2 vs 25,7 (Retina6) | Fondo por kernel gaussiano σ = 60 (resta) → CLAHE (0,01; mosaico 64) |

---

## 5. Orden de trabajo propuesto (Victoria)

1. Repo git + estructura + entorno (`pixi install`, pyradiomics sí/no).
2. Extraer `tp1/core.py`; copiar `tp2/`; comprobar que importan sin abrir napari.
3. Descargar HRF → `manifest.csv`.
4. Act. 1: análisis de canal y de calibre → **fijar f** → cachear preprocesadas.
5. Act. 2: `splits.json` → **avisar a Tomás** (a partir de acá puede empezar la U-Net).
6. Módulos compartidos: `metricas`, `postproceso`, `registro`.
7. Act. 3: features → RF primero (rápido, sirve para el experimento de leakage) → k-NN, SVM →
   selección/PCA → no supervisado → trazos.
8. Act. 4: landmarks → descriptores (GT y fuera de fold) → CV anidada → test.
9. Solapa ML del widget.

---

## 6. Decisiones

| | Decisión | Estado |
|---|---|---|
| A | Tamaño del test | **9 imágenes (3 por grupo)** — fijado en `config/splits.json` |
| B | Disco óptico y fóvea | **Disco: archivo oficial de HRF** (`optic_disk_centers.xls`, dos expertos, centro + diámetro). 02_g y 02_dr tienen discrepancia entre expertos (error de tipeo) → revisar en napari. **Fóvea: marcado manual** en napari (no viene en HRF) |
| C | Fuente de HRF | **FAU oficial** (`all.zip`, 73 MB, CC-BY 4.0) |
| D | Factor de submuestreo | f = 2 provisorio, se confirma con el análisis de calibre |
| E | Repo compartido | Repo git local inicializado; falta crear el remoto en GitHub |
| F | Radiomics | **pyradiomics descartado** (binarios sólo hasta Python 3.9). GLCM propia en `tp2/radiomics.py`, verificada contra los 157 valores del TP2 (error < 1e-12). GLRLM/GLSZM/GLDM/NGTDM = NaN |
| G | Solapas | **Por función, no por TP** (pedido del profesor): Datos · Calidad · Preproceso · Descriptores · Segmentar · Clasificar · Resultados, con secciones plegables. Las herramientas del TP1 y TP2 se reparten entre ellas; ML y DL de una misma tarea comparten solapa. Cada herramienta nueva reemplaza un `pendiente(...)` en `src/tp3/ui/ventana.py` |

## 7. Estado de la puesta en marcha

- Entorno pixi instalado y verificado (`pixi run check-tp3`). Sin GPU local.
- `data/manifest.csv`: 45 imágenes verificadas, con hashes. Fracción de vaso dentro del FOV:
  sanos 11,0 % ± 1,1; glaucoma 8,2 % ± 0,7; RD 8,2 % ± 1,2 (resolución nativa).
  El desbalance es ~1:10 y la densidad vascular ya separa sanos del resto: es un buen candidato
  para la Act. 4.
- Diámetro del disco óptico ≈ 370 px nativos. Con disco–fóvea ≈ 2,5 diámetros ≈ 925 px, el pool
  del TP1 (≈ 265 px disco–fóvea) queda a escala ≈ HRF / 3,5 **[verificar con la fóvea marcada]**.
- `config/splits.json` (semilla 42, hash `2bff0994171d`):
  - test: 01_g, 03_dr, 07_dr, 07_h, 08_h, 09_g, 10_dr, 10_h, 14_g
  - validación de segmentación (fold 0): 02_dr, 04_dr, 04_g, 04_h, 06_h, 07_g, 14_h, 15_dr
  - 28 de entrenamiento; folds 1–4 y subconjuntos 3 ⊂ 6 ⊂ 10 en el archivo.
