"""El split debe ser reproducible, estratificado y sin fugas entre conjuntos."""
import pytest

from tp3 import datos, splits

IDS = [f"{n:02d}_{s}" for s in ("h", "g", "dr") for n in range(1, 16)]


def test_reproducible_con_misma_semilla():
    assert splits.generar_splits(IDS) == splits.generar_splits(IDS)


def test_otra_semilla_cambia_la_asignacion():
    assert splits.generar_splits(IDS)["hash"] != splits.generar_splits(IDS, semilla=7)["hash"]


def test_sin_fugas_y_estratificado():
    s = splits.generar_splits(IDS)
    splits.validar(s)
    assert len(s["test"]) == 9
    for c in datos.CLASES:
        assert sum(datos.grupo_de(i) == c for i in s["test"]) == 3
    assert sorted(len(f) for f in s["folds"]) == [7, 7, 7, 7, 8]
    for f in s["folds"]:
        for c in datos.CLASES:
            assert 2 <= sum(datos.grupo_de(i) == c for i in f) <= 3
    assert set(s["test"]) | {i for f in s["folds"] for i in f} == set(IDS)


def test_subconjuntos_anidados_y_estratificados():
    s = splits.generar_splits(IDS)
    e = s["eficiencia_datos"]
    assert [len(e[k]) for k in ("3", "6", "10")] == [3, 6, 10]
    assert {datos.grupo_de(i) for i in e["3"]} == set(datos.CLASES)


def test_validar_detecta_fuga():
    s = splits.generar_splits(IDS)
    s["segmentacion"]["train"] = s["segmentacion"]["train"] + [s["test"][0]]
    with pytest.raises(AssertionError):
        splits.validar(s)


@pytest.mark.skipif(not splits.rutas.SPLITS.exists(), reason="sin config/splits.json")
def test_archivo_guardado_valido():
    splits.leer_split()
