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
from airflow.sdk import Asset

DAGS = Path(__file__).resolve().parent.parent / "dags"
# El DAG orquesta, no calcula (ver reentrenar.py). Importar esto en un DAG es la señal de
# que el cómputo se está metiendo en Airflow.
PROHIBIDOS = {"pandas", "numpy", "sklearn", "mlflow", "src"}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS))


def test_todos_los_dags_cargan_sin_errores(dagbag):
    assert dagbag.import_errors == {}, dagbag.import_errors
    assert {"hola", "reentrenar", "monitor"} <= set(dagbag.dag_ids)


def test_reentrenar_tiene_el_orden_correcto(dagbag):
    dag = dagbag.dags["reentrenar"]  # el dict, no get_dag(): ése consulta la base
    assert dag.get_task("construir_dataset").downstream_task_ids == {"validar_datos"}
    assert dag.get_task("validar_datos").downstream_task_ids == {"entrenar"}
    assert dag.get_task("entrenar").downstream_task_ids == {"comparar_con_champion"}
    assert dag.get_task("comparar_con_champion").downstream_task_ids == {"decidir_promocion"}
    assert dag.max_active_runs == 1 and dag.schedule == [Asset("drift_sostenido")]


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
    for tid in ("entrenar", "comparar_con_champion", "decidir_promocion"):
        assert dag.get_task(tid).environment["ORQUESTADOR_RUN_ID"] == "{{ run_id }}"


def test_el_gate_rechaza_como_skip_no_como_falla(dagbag):
    """Hito 8: "no promover" es un resultado normal del gate. Pinta la tarea de rosa
    (skipped), no de rojo; un error de verdad (sin versión, MLflow caído) sí falla."""
    gate = dagbag.dags["reentrenar"].get_task("decidir_promocion")
    assert gate.command == "python -m src.promocion"
    assert 99 in gate.skip_on_exit_code and 1 not in gate.skip_on_exit_code


@pytest.mark.parametrize("archivo", sorted(DAGS.glob("*.py")), ids=lambda p: p.name)
def test_los_dags_no_importan_computo(archivo):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    importados = {n.name.split(".")[0] for nodo in ast.walk(arbol)
                  if isinstance(nodo, ast.Import) for n in nodo.names}
    importados |= {nodo.module.split(".")[0] for nodo in ast.walk(arbol)
                   if isinstance(nodo, ast.ImportFrom) and nodo.module}
    assert not importados & PROHIBIDOS, f"{archivo.name} importa {importados & PROHIBIDOS}"


def test_monitor_no_expone_la_contrasena_y_no_escribe_datos(dagbag):
    """La conexión a la base viaja en private_environment (no se muestra en la UI) y el
    dataset se monta en sólo lectura."""
    for tarea in dagbag.dags["monitor"].tasks:
        assert "MONITOR_DSN" not in tarea.environment
        # El operador lo guarda en un atributo privado a propósito: no es un template field.
        assert "MONITOR_DSN" in tarea._private_environment
        assert all(m["ReadOnly"] for m in tarea.mounts)
        assert tarea.environment["ORQUESTADOR_RUN_ID"] == "{{ run_id }}"


def test_monitor_mide_decide_y_publica_lo_que_escucha_reentrenar(dagbag):
    """El ciclo del Hito 8, como cableado: medir → decidir → evento → reentrenar."""
    monitor, reentrenar = dagbag.dags["monitor"], dagbag.dags["reentrenar"]
    assert monitor.get_task("calcular_drift").downstream_task_ids == {"evaluar_persistencia"}
    decide = monitor.get_task("evaluar_persistencia")
    assert decide.outlets == reentrenar.schedule == [Asset("drift_sostenido")]
    # "No disparar" tiene que ser SKIP, no FALLO: una tarea saltada no emite su evento,
    # y una fallida pintaría de rojo un DAG que hizo exactamente lo que debía.
    assert 99 in decide.skip_on_exit_code


def test_cada_corrida_mide_hasta_su_hora_programada(dagbag):
    """La ventana termina en `run_after` (la hora programada), no en "ahora"."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    t = dagbag.dags["monitor"].get_task("calcular_drift")
    ctx = {"dag_run": SimpleNamespace(run_after=datetime(2026, 11, 13, 5, 55, tzinfo=UTC)),
           "params": {"ventana_horas": 24}}
    assert t.render_template(t.command, ctx) == (
        "python -m src.monitoreo.drift --horas 24 --hasta 2026-11-13T05:55:00+00:00")
