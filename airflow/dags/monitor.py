"""DAG `monitor`: drift diario de las predicciones y, si es sostenido, dispara `reentrenar`.

    calcular_drift ──▶ evaluar_persistencia ══(evento drift_sostenido)══▶ DAG `reentrenar`
    (PSI → tabla        (¿empezó un episodio
     drift → Grafana)    de drift sostenido?)

Corre solo cada noche, o a mano ("Trigger DAG w/ config"):
  · `ventana_horas`: cuántas horas hacia atrás mide (24 = el día).
  · `mediciones_para_reentrenar`: cuántas mediciones seguidas con drift hacen un episodio
    (3 = tres noches). Para una demostración, 1.

Cada corrida mide SU ventana: termina en la hora PROGRAMADA de la corrida (`run_after`),
no en el momento en que se ejecuta. Una corrida atrasada, o una que se vuelve a correr
("Clear") la semana siguiente, mide el mismo día y sobrescribe su propio resultado. Sin
esto, "tres noches seguidas" no significaría nada.

`evaluar_persistencia` es la que decide (src/monitoreo/persistencia.py). Si no hay que
disparar, termina con el código 99 y Airflow la marca "skipped": una tarea saltada NO
emite su evento, así que `reentrenar` no se entera. Si hay que disparar, termina bien, el
evento `drift_sostenido` queda registrado y `reentrenar` (que lo tiene como `schedule`)
arranca. Esto es *data-aware scheduling*: un DAG no llama a otro; publica un hecho y
quien depende de él reacciona.

Misma regla que `reentrenar`: el DAG orquesta, no calcula. Las dos tareas arrancan la
imagen de entrenamiento con un comando de `src/`, probado sin Airflow (tests/).

Seguridad: la conexión a la base va en `private_environment` (Airflow no la muestra en la
UI ni en los logs) y el dataset se monta en sólo lectura.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import Asset, Param, dag
from docker.types import Mount

DOCKER_URL = os.environ.get("ML_DOCKER_URL", "tcp://docker-proxy:2375")
IMAGEN = os.environ.get("ML_IMAGEN_ENTRENAMIENTO", "ml-predict-income-train:dev")
RED = os.environ.get("ML_RED_DOCKER", "ml-predict-income_default")
VOLUMEN_DATOS = os.environ.get("ML_VOLUMEN_DATOS", "ml-predict-income-datos")
MONITOR_DSN = os.environ.get("ML_MONITOR_DSN", "")

# El hecho que publica este DAG. `reentrenar.py` declara EL MISMO nombre como su schedule
# (lo verifica airflow/tests/test_dags.py): ése es todo el acoplamiento entre los dos.
DRIFT_SOSTENIDO = Asset("drift_sostenido")
SALTAR = 99   # = src.monitoreo.persistencia.SALTAR (lo verifica tests/test_persistencia.py)


def en_contenedor(task_id: str, comando: str, **kwargs) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=IMAGEN,
        command=comando,
        docker_url=DOCKER_URL,
        api_version="auto",
        network_mode=RED,                   # ahí está predicciones-db
        mounts=[Mount(target="/app/data/processed", source=VOLUMEN_DATOS,
                      type="volume", read_only=True)],
        environment={"ORQUESTADOR_RUN_ID": "{{ run_id }}"},   # reintento = misma corrida
        private_environment={"MONITOR_DSN": MONITOR_DSN},
        mount_tmp_dir=False,
        auto_remove="success",
        force_pull=False,
        **kwargs,
    )


@dag(
    dag_id="monitor",
    description="PSI diario de las predicciones; si el drift es sostenido, dispara reentrenar",
    # 23:55 de la Ciudad de México: mide el día que está terminando.
    schedule="55 23 * * *",
    start_date=pendulum.datetime(2026, 11, 1, tz="America/Mexico_City"),
    catchup=False,           # sin rellenar días pasados: el drift de ayer ya no cambia nada
    max_active_runs=1,       # las mediciones van en orden: la persistencia lee las previas
    tags=["m7", "m8", "monitoreo"],
    doc_md=__doc__,
    params={
        "ventana_horas": Param(24, type="number", minimum=0.1, maximum=24 * 14,
                               title="Ventana (horas hacia atrás)"),
        "mediciones_para_reentrenar": Param(
            3, type="integer", minimum=1, maximum=14,
            title="Mediciones seguidas con drift para reentrenar",
            description="3 = tres noches. Para una demostración, 1."),
    },
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2),
                  "execution_timeout": timedelta(minutes=9)},   # el proxy corta a los 10
)
def monitor():
    calcular = en_contenedor(
        "calcular_drift",
        "python -m src.monitoreo.drift --horas {{ params.ventana_horas }}"
        " --hasta {{ dag_run.run_after.isoformat() }}")
    persistencia = en_contenedor(
        "evaluar_persistencia",
        "python -m src.monitoreo.persistencia"
        " --mediciones {{ params.mediciones_para_reentrenar }}",
        skip_on_exit_code=SALTAR,
        outlets=[DRIFT_SOSTENIDO],
    )
    calcular >> persistencia


monitor()
