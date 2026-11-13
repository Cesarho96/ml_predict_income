"""DAG `monitor`: drift diario de las predicciones contra la población de entrenamiento.

    calcular_drift   (PSI por feature y segmento → tabla `drift` → Grafana)

Corre solo cada noche sobre las últimas 24 h, o a mano ("Trigger DAG w/ config") con otra
ventana: para la demo, `ventana_horas` = 1 justo después de mandar tráfico.

Misma regla que `reentrenar`: el DAG orquesta, no calcula. La tarea arranca la imagen de
entrenamiento (trae pandas, el dataset y psycopg) con `python -m src.monitoreo.drift`; el
cálculo se prueba en tests/test_drift.py, sin Airflow.

Dos detalles de seguridad:
  · La conexión va en `private_environment`: Airflow NO la muestra en la UI ni en los
    logs. En `environment` cualquiera con acceso a la UI vería la contraseña en
    "Rendered Template".
  · El dataset se monta en SÓLO LECTURA: el monitoreo no tiene por qué escribir en él.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import Param, dag
from docker.types import Mount

DOCKER_URL = os.environ.get("ML_DOCKER_URL", "tcp://docker-proxy:2375")
IMAGEN = os.environ.get("ML_IMAGEN_ENTRENAMIENTO", "ml-predict-income-train:dev")
RED = os.environ.get("ML_RED_DOCKER", "ml-predict-income_default")
VOLUMEN_DATOS = os.environ.get("ML_VOLUMEN_DATOS", "ml-predict-income-datos")
MONITOR_DSN = os.environ.get("ML_MONITOR_DSN", "")


@dag(
    dag_id="monitor",
    description="PSI de las predicciones recientes contra la población de entrenamiento",
    # 23:55 de la Ciudad de México: mide el día que está terminando.
    schedule="55 23 * * *",
    start_date=pendulum.datetime(2026, 11, 1, tz="America/Mexico_City"),
    catchup=False,           # sin rellenar días pasados: el drift de ayer ya no cambia nada
    max_active_runs=1,
    tags=["m7", "monitoreo"],
    doc_md=__doc__,
    params={"ventana_horas": Param(24, type="number", minimum=0.1, maximum=24 * 14,
                                   title="Ventana (horas hacia atrás)")},
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2),
                  "execution_timeout": timedelta(minutes=9)},   # el proxy corta a los 10
)
def monitor():
    DockerOperator(
        task_id="calcular_drift",
        image=IMAGEN,
        command="python -m src.monitoreo.drift --horas {{ params.ventana_horas }}",
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
    )


monitor()
