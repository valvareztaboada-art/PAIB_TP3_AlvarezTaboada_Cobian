"""Comprueba que el entorno y los módulos del proyecto importen sin abrir napari."""
import _path  # noqa: F401
import importlib

modulos = [
    "numpy", "scipy", "skimage", "sklearn", "pandas", "pywt", "cv2", "torch", "torchvision",
    "napari", "magicgui", "xlrd",
    "tp1.core", "tp2.texture", "tp2.structure", "tp2.measure", "tp2.radiomics", "tp2.batch_ui",
    "tp3.datos", "tp3.splits", "tp3.ui.ventana",
]
fallas = []
for m in modulos:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        fallas.append(m)
        print(f"FALLA {m}: {e}")
from tp2.radiomics import PYRADIOMICS
print("PyRadiomics:", "sí" if PYRADIOMICS else "no (GLCM propia equivalente; demás familias = NaN)")
import torch
print("CUDA disponible:", torch.cuda.is_available())
raise SystemExit(1 if fallas else 0)
