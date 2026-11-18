"""Act. 1 (inventario): verifica HRF y escribe data/manifest.csv.

Una fila por imagen con grupo, rutas, hashes SHA-256, tamaño, chequeos de
las máscaras, desbalance vaso/fondo dentro del FOV (resolución nativa) y
centro/diámetro del disco óptico de los expertos de HRF.

Uso: pixi run manifest
"""
import _path  # noqa: F401

import pandas as pd

from tp3 import datos, rutas

ids = datos.listar_ids()
assert len(ids) == 45, f"Se esperaban 45 imágenes y hay {len(ids)}"
discos = datos.leer_centros_disco()

filas = []
for i in ids:
    rgb, gt, fov = datos.cargar(i)
    assert rgb.shape[:2] == gt.shape == fov.shape, f"{i}: tamaños distintos"
    n_fov = int(fov.sum())
    n_vaso = int((gt & fov).sum())
    filas.append(
        dict(
            imagen_id=i,
            grupo=datos.grupo_de(i),
            archivo_imagen=datos.ruta_imagen(i).relative_to(rutas.RAIZ).as_posix(),
            archivo_gt=datos.ruta_manual(i).relative_to(rutas.RAIZ).as_posix(),
            archivo_fov=datos.ruta_fov(i).relative_to(rutas.RAIZ).as_posix(),
            sha256_imagen=datos.sha256(datos.ruta_imagen(i)),
            sha256_gt=datos.sha256(datos.ruta_manual(i)),
            sha256_fov=datos.sha256(datos.ruta_fov(i)),
            alto=rgb.shape[0],
            ancho=rgb.shape[1],
            fov_px=n_fov,
            fov_fraccion_imagen=n_fov / fov.size,
            vaso_px_fov=n_vaso,
            vaso_px_fuera_fov=int((gt & ~fov).sum()),
            fraccion_vaso_fov=n_vaso / n_fov,
            **discos.get(i, {}),
        )
    )
    print(f"{i:>6}  vaso/FOV = {n_vaso / n_fov:6.2%}")

df = pd.DataFrame(filas)
df.to_csv(rutas.MANIFEST, index=False)
print(f"\nManifest: {rutas.MANIFEST}  ({len(df)} imágenes)")
print("\nFracción de vaso dentro del FOV por grupo (resolución nativa):")
print(df.groupby("grupo")["fraccion_vaso_fov"].agg(["mean", "std", "min", "max"]).round(4))
print(f"\nPíxeles de vaso fuera del FOV (total): {df['vaso_px_fuera_fov'].sum()}")
if "disco_discrepancia" in df:
    print("Discrepancias entre expertos en el disco óptico:",
          df.loc[df["disco_discrepancia"] == True, "imagen_id"].tolist())  # noqa: E712
