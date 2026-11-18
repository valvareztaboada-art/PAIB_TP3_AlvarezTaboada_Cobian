"""Act. 2: genera config/splits.json (semilla 42) y lo valida.

Uso: pixi run splits
"""
import _path  # noqa: F401

from tp3 import datos, splits

ids = datos.listar_ids()
assert len(ids) == 45, f"Se esperaban 45 imágenes y hay {len(ids)}"
s = splits.generar_splits(ids)
splits.validar(s)
ruta = splits.guardar(s)

print(f"Split guardado en {ruta}  (semilla {s['semilla']}, hash {s['hash']})")
print(f"test ({len(s['test'])}): {s['test']}")
for k, f in enumerate(s["folds"]):
    cuenta = {c: sum(datos.grupo_de(i) == c for i in f) for c in datos.CLASES}
    marca = "  <- validación segmentación" if k == s["segmentacion"]["fold_val"] else ""
    print(f"fold {k} ({len(f)}) {cuenta}: {f}{marca}")
print(f"train segmentación: {len(s['segmentacion']['train'])} imágenes")
for n, sub in s["eficiencia_datos"].items():
    print(f"eficiencia {n:>5}: {sub}")
