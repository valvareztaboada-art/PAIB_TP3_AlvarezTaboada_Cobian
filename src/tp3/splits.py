"""División de HRF por imagen (= sujeto), única para ML y DL.

Esquema (ver PLAN_TP3.md §1.3):

    45 imágenes (15 por grupo)
    ├── test: n_test_por_grupo por grupo; se usa una sola vez, al final
    └── desarrollo: el resto, en n_folds folds estratificados por grupo
          ├── segmentación: fold `fold_val_seg` = validación; el resto = entrenamiento
          └── clasificación: validación cruzada con los mismos folds

También fija subconjuntos anidados del entrenamiento de segmentación para la
curva de eficiencia en datos (Act. 7), estratificados por grupo, de modo que
ML y DL usen exactamente las mismas imágenes.

La asignación depende sólo de la lista de ids y de la semilla.
"""

import hashlib
import json
from collections import OrderedDict

import numpy as np

from . import rutas
from .datos import CLASES, grupo_de

SEMILLA = 42


def _por_grupo(ids):
    g = OrderedDict((c, []) for c in CLASES)
    for i in ids:
        g[grupo_de(i)].append(i)
    return g


def _reparto(n, k):
    """Reparte n elementos en k grupos lo más parejo posible: n=10,k=3 -> [4,3,3]."""
    base, resto = divmod(n, k)
    return [base + (j < resto) for j in range(k)]


def generar_splits(
    ids,
    semilla=SEMILLA,
    n_test_por_grupo=3,
    n_folds=5,
    fold_val_seg=0,
    tamanos_eficiencia=(3, 6, 10),
):
    ids = sorted(ids)
    rng = np.random.default_rng(semilla)
    grupos = _por_grupo(ids)
    barajados = {c: list(rng.permutation(v)) for c, v in grupos.items()}

    test = {c: sorted(v[:n_test_por_grupo]) for c, v in barajados.items()}
    desarrollo = {c: v[n_test_por_grupo:] for c, v in barajados.items()}

    # Folds: se recorren los grupos concatenados repartiendo en ronda; el índice
    # sigue corriendo entre grupos, así cada fold queda con 2-3 imágenes por grupo
    # y tamaños que difieren a lo sumo en uno.
    folds = [[] for _ in range(n_folds)]
    k = 0
    for c in CLASES:
        for i in desarrollo[c]:
            folds[k % n_folds].append(str(i))
            k += 1
    folds = [sorted(f) for f in folds]

    val_seg = folds[fold_val_seg]
    train_seg = sorted(i for j, f in enumerate(folds) if j != fold_val_seg for i in f)

    # Subconjuntos anidados: dentro de cada grupo se respeta el orden barajado,
    # así el subconjunto de 3 está contenido en el de 6, y este en el de 10.
    orden_train = {c: [i for i in desarrollo[c] if i in set(train_seg)] for c in CLASES}
    eficiencia = OrderedDict()
    for n in tamanos_eficiencia:
        cuotas = _reparto(n, len(CLASES))
        eficiencia[str(n)] = sorted(
            str(i) for c, q in zip(CLASES, cuotas) for i in orden_train[c][:q]
        )
    eficiencia["todas"] = train_seg

    asignacion = OrderedDict(
        test=sorted(str(i) for v in test.values() for i in v),
        folds=folds,
        segmentacion=OrderedDict(
            train=train_seg, val=val_seg, fold_val=fold_val_seg
        ),
        clasificacion=OrderedDict(
            esquema=f"{n_folds}-fold estratificado por grupo sobre desarrollo",
            desarrollo=sorted(i for f in folds for i in f),
        ),
        eficiencia_datos=eficiencia,
    )
    return OrderedDict(
        descripcion="Split por imagen de HRF (imagen = sujeto). Común a ML y DL.",
        semilla=semilla,
        parametros=OrderedDict(
            n_test_por_grupo=n_test_por_grupo,
            n_folds=n_folds,
            fold_val_seg=fold_val_seg,
            tamanos_eficiencia=list(tamanos_eficiencia),
        ),
        clases=list(CLASES),
        hash=hash_asignacion(asignacion),
        **asignacion,
    )


def hash_asignacion(asignacion):
    """Huella corta de la asignación: va al CSV y al JSON de cada modelo."""
    claves = ("test", "folds", "segmentacion", "eficiencia_datos")
    texto = json.dumps({k: asignacion[k] for k in claves}, sort_keys=True)
    return hashlib.sha256(texto.encode()).hexdigest()[:12]


def validar(s):
    """Comprueba que no haya imágenes compartidas entre conjuntos."""
    test = set(s["test"])
    folds = [set(f) for f in s["folds"]]
    todas = set().union(*folds)
    assert not test & todas, "imágenes de test en desarrollo"
    for a in range(len(folds)):
        for b in range(a + 1, len(folds)):
            assert not folds[a] & folds[b], f"folds {a} y {b} comparten imágenes"
    tr, va = set(s["segmentacion"]["train"]), set(s["segmentacion"]["val"])
    assert not tr & va and not (tr | va) & test, "fuga entre train/val/test de segmentación"
    anterior = set()
    for n, sub in s["eficiencia_datos"].items():
        sub = set(sub)
        assert sub <= tr, f"subconjunto {n} fuera de train"
        assert anterior <= sub, f"subconjunto {n} no contiene al anterior"
        anterior = sub
    assert s["hash"] == hash_asignacion(s), "el hash no coincide con la asignación"
    return True


def guardar(s, ruta=None):
    ruta = ruta or rutas.SPLITS
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def leer_split(ruta=None):
    s = json.loads((ruta or rutas.SPLITS).read_text(encoding="utf-8"))
    validar(s)
    return s


def conjunto_de(id_imagen, s=None):
    """'test', 'train' o 'val' para segmentación; útil para el CSV."""
    s = s or leer_split()
    if id_imagen in s["test"]:
        return "test"
    return "val" if id_imagen in s["segmentacion"]["val"] else "train"


def fold_de(id_imagen, s=None):
    s = s or leer_split()
    for k, f in enumerate(s["folds"]):
        if id_imagen in f:
            return k
    return None
