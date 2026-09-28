"""Los DAGs se cargan en Airflow real y respetan las reglas del proyecto.

Corre en su propio job de CI, con Airflow instalado y SIN las dependencias del modelo:
el mismo aislamiento que en producción. Un error de import en un DAG no rompe el
entrenamiento ni la API, pero sí deja a Airflow sin el DAG —y eso sólo se ve en la UI,
tarde. Aquí se ve en el PR.
"""

import ast
from pathlib import Path

import pytest
from airflow.models.dagbag import DagBag

DAGS = Path(__file__).resolve().parent.parent / "dags"
# El DAG orquesta, no calcula (ver reentrenar.py). Importar esto en un DAG es la señal de
# que el cómputo se está metiendo en Airflow.
PROHIBIDOS = {"pandas", "numpy", "sklearn", "mlflow", "src"}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS))


def test_todos_los_dags_cargan_sin_errores(dagbag):
    assert dagbag.import_errors == {}, dagbag.import_errors
    assert {"hola", "reentrenar"} <= set(dagbag.dag_ids)


def test_reentrenar_tiene_el_orden_correcto(dagbag):
    dag = dagbag.dags["reentrenar"]  # el dict, no get_dag(): ése consulta la base
    assert dag.get_task("construir_dataset").downstream_task_ids == {"validar_datos"}
    assert dag.get_task("validar_datos").downstream_task_ids == {"entrenar"}
    assert dag.get_task("entrenar").downstream_task_ids == {"comparar_con_champion"}
    assert dag.max_active_runs == 1 and dag.schedule is None


def test_solo_construir_escribe_y_todas_llevan_edicion_y_linaje(dagbag):
    dag = dagbag.dags["reentrenar"]  # el dict, no get_dag(): ése consulta la base
    assert dag.params["edicion"] == 2024
    for t in dag.tasks:
        escribe = [m["Target"] for m in t.mounts if not m["ReadOnly"]]
        # Sólo construir_dataset escribe, y sólo en los procesados; el crudo, nunca.
        esperado = ["/app/data/processed"] if t.task_id == "construir_dataset" else []
        assert escribe == esperado, f"{t.task_id} escribe en {escribe}"
        assert t.environment["EDICION"] == "{{ params.edicion }}"
        assert t.mount_tmp_dir is False
    for tid in ("entrenar", "comparar_con_champion"):
        assert dag.get_task(tid).environment["ORQUESTADOR_RUN_ID"] == "{{ run_id }}"


@pytest.mark.parametrize("archivo", sorted(DAGS.glob("*.py")), ids=lambda p: p.name)
def test_los_dags_no_importan_computo(archivo):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    importados = {n.name.split(".")[0] for nodo in ast.walk(arbol)
                  if isinstance(nodo, ast.Import) for n in nodo.names}
    importados |= {nodo.module.split(".")[0] for nodo in ast.walk(arbol)
                   if isinstance(nodo, ast.ImportFrom) and nodo.module}
    assert not importados & PROHIBIDOS, f"{archivo.name} importa {importados & PROHIBIDOS}"
