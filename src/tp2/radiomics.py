"""Radiómica 2D con discretización común explícita, sin reescalar ROIs.

Con PyRadiomics instalado se calculan las cinco familias (GLCM, GLRLM, GLSZM,
GLDM, NGTDM). PyRadiomics sólo publica binarios hasta Python 3.9; en el
entorno del TP3 (Python 3.10) no está disponible y se usa una GLCM propia
con la misma discretización y las mismas definiciones que PyRadiomics
(simétrica; desplazamientos con distancia de Chebyshev d; promedio de cada
característica). Reproduce los 157 valores GLCM del TP2 calculados con
PyRadiomics 3.1.0 (error relativo < 1e-12). Las otras familias se informan
como NaN y lo indica `settings["engine"]`.
"""

import logging
import numpy as np
from .data import validate_mask
from .measure import intensity, geometry

FEATURES = {
    "glcm": ["Contrast", "JointEntropy", "JointEnergy", "Idm", "Correlation"],
    "glrlm": [
        "ShortRunEmphasis",
        "LongRunEmphasis",
        "GrayLevelNonUniformityNormalized",
        "RunPercentage",
    ],
    "glszm": ["SmallAreaEmphasis", "LargeAreaEmphasis", "ZonePercentage", "ZoneEntropy"],
    "gldm": ["SmallDependenceEmphasis", "LargeDependenceEmphasis", "DependenceEntropy"],
    "ngtdm": ["Coarseness", "Contrast", "Busyness", "Complexity", "Strength"],
}

try:
    import SimpleITK as sitk
    from radiomics import glcm, glrlm, glszm, gldm, ngtdm

    PYRADIOMICS = True
    _CLASSES = dict(
        glcm=glcm.RadiomicsGLCM,
        glrlm=glrlm.RadiomicsGLRLM,
        glszm=glszm.RadiomicsGLSZM,
        gldm=gldm.RadiomicsGLDM,
        ngtdm=ngtdm.RadiomicsNGTDM,
    )
    FAMILIES = {k: (_CLASSES[k], v) for k, v in FEATURES.items()}
except ImportError:
    PYRADIOMICS = False
    FAMILIES = {}


def _cooccurrence(q, mask, dr, dc, n):
    """Matriz n×n de pares (q[r,c], q[r+dr,c+dc]) con ambos píxeles en la ROI, simetrizada."""
    h, w = q.shape
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    a, b = q[r0:r1, c0:c1], q[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    ok = mask[r0:r1, c0:c1] & mask[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    m = np.bincount((a[ok] - 1) * n + (b[ok] - 1), minlength=n * n).reshape(n, n).astype(float)
    return m + m.T


def glcm_propia(q, mask, distances=(1,)):
    """GLCM 2D simétrica sobre niveles 1..N dentro de la ROI, con las definiciones de PyRadiomics.

    Como PyRadiomics (imageoperations.generateAngles), para cada distancia d
    usa todos los desplazamientos con distancia de Chebyshev d, uno por cada
    par de sentidos opuestos: 4 direcciones para d = 1, 8 para d = 2. No es la
    distancia euclídea redondeada de skimage.feature.graycomatrix. Cada
    característica se calcula por desplazamiento y luego se promedia.
    """
    q = q.astype(np.int64)
    n = int(q[mask].max())
    offsets = sorted(
        {
            (dr, dc)
            for d in map(int, distances)
            for dr in range(0, d + 1)
            for dc in range(-d, d + 1)
            if max(abs(dr), abs(dc)) == d and (dr > 0 or dc > 0)
        }
    )
    i, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    valores = {k: [] for k in FEATURES["glcm"]}
    for dr, dc in offsets:
        p = _cooccurrence(q, mask, dr, dc, n)
        total = p.sum()
        if total == 0:
            continue
        p = p / total
        mux, muy = (p * i).sum(), (p * j).sum()
        sx = np.sqrt((p * (i - mux) ** 2).sum())
        sy = np.sqrt((p * (j - muy) ** 2).sum())
        nz = p[p > 0]
        valores["Contrast"].append((p * (i - j) ** 2).sum())
        valores["JointEntropy"].append(-(nz * np.log2(nz)).sum())
        valores["JointEnergy"].append((p**2).sum())
        valores["Idm"].append((p / (1 + (i - j) ** 2)).sum())
        # PyRadiomics define la correlación como 1 cuando la varianza es nula.
        corr = ((p * i * j).sum() - mux * muy) / (sx * sy) if sx * sy > 0 else 1.0
        valores["Correlation"].append(corr)
    return {f"glcm_{k}": float(np.mean(v)) if v else np.nan for k, v in valores.items()}


def discretize(image, mask, mode="count", bins=32, width=8, low=0, high=255):
    """Cuantiza con extremos fijos comunes; cuenta los valores recortados al intervalo."""
    validate_mask(image, mask, 4)
    if high <= low:
        raise ValueError("El límite superior debe superar al inferior.")
    if mode == "count":
        if not 2 <= int(bins) <= 128:
            raise ValueError("Usar entre 2 y 128 bins.")
        step = (high - low) / int(bins)
        n = int(bins)
    elif mode == "width":
        if width <= 0:
            raise ValueError("Ancho de bin positivo requerido.")
        step = float(width)
        n = int(np.ceil((high - low) / step))
        if n > 128:
            raise ValueError("El ancho solicitado produce más de 128 bins; aumentarlo.")
    else:
        raise ValueError("Discretización desconocida.")
    q = np.clip(np.floor((image - low) / step), 0, n - 1).astype(np.int32) + 1
    clipped = int(np.count_nonzero(mask & ((image < low) | (image > high))))
    return q, dict(
        bin_mode=mode,
        bin_count_efectivo=n,
        bin_width_efectivo=step,
        bin_low=low,
        bin_high=high,
        n_intensidades_recortadas=clipped,
    )


class FixedBins:
    """Adapta el hook de binning 3.1.0 para conservar etiquetas 1..N ya cuantizadas.

    Se usa el motor de matrices de PyRadiomics sin su recentrado del mínimo de
    cada ROI; las etiquetas absolutas conservan la misma interpretación entre imágenes.
    """

    def _applyBinning(self, matrix):
        """Informa niveles presentes y Ng al motor, sin una segunda discretización."""
        q = np.rint(matrix).astype(int)
        levels = np.unique(q[self.maskArray])
        self.coefficients["grayLevels"] = levels
        self.coefficients["Ng"] = int(levels.max())
        return q


def extract_radiomics(
    image, mask, mode="count", bins=32, width=8, low=0, high=255, distances=(1,), alpha=0
):
    """Extrae familias representativas en 2D, GLCM simétrica y promedio por dirección."""
    mask = validate_mask(image, mask, 4)
    q, settings = discretize(image, mask, mode, bins, width, low, high)
    if not distances or any(int(d) < 1 for d in distances):
        raise ValueError("Las distancias deben ser enteros positivos.")
    results = {f"primer_orden_{k}": v for k, v in intensity(image, mask).items()}
    geom, _ = geometry(mask)
    results.update({f"forma_{k}": v for k, v in geom.items()})
    if not PYRADIOMICS:
        results.update(glcm_propia(q, mask, distances))
        for family, names in FEATURES.items():
            if family != "glcm":
                results.update({f"{family}_{name}": np.nan for name in names})
        results["n_intensidades_recortadas"] = settings.pop("n_intensidades_recortadas")
        settings.update(
            distances=list(distances),
            gldm_alpha=alpha,
            dimensions=2,
            directions="GLCM: desplazamientos con distancia de Chebyshev d, promedio de características; simétrica",
            engine="GLCM propia equivalente a PyRadiomics (PyRadiomics no disponible: GLRLM, GLSZM, GLDM y NGTDM = NaN)",
        )
        return results, q, settings
    img = sitk.GetImageFromArray(q.astype(float))
    roi = sitk.GetImageFromArray(mask.astype(np.uint8))
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    for family, (cls, names) in FAMILIES.items():
        engine_class = type(f"Fixed_{family}", (FixedBins, cls), {})
        engine = engine_class(
            img,
            roi,
            distances=list(map(int, distances)),
            gldm_a=int(alpha),
            symmetricalGLCM=True,
            label=1,
        )
        for feature in names:
            engine.enableFeatureByName(feature)
        values = engine.execute()
        results.update(
            {f"{family}_{key}": float(np.asarray(value).squeeze()) for key, value in values.items()}
        )
    results["n_intensidades_recortadas"] = settings.pop("n_intensidades_recortadas")
    settings.update(
        distances=list(distances),
        gldm_alpha=alpha,
        glszm_connectivity=8,
        dimensions=2,
        directions="GLCM/GLRLM: cuatro direcciones 2D, promedio de características; GLCM simétrica",
        engine="PyRadiomics 3.1.0 + FixedBins",
    )
    return results, q, settings
