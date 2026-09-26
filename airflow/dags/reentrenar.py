"""DAG `reentrenar`: valida los datos → entrena → compara con el champion → challenger.

    validar_datos ──▶ entrenar ──▶ comparar_con_champion
     (pandera)       (registra      (evalúa ambos sobre el mismo test;
                      versión)       marca @challenger; NUNCA toca @champion)

La regla de diseño de este archivo: **el DAG orquesta, no calcula.** Ninguna tarea
importa pandas ni scikit-learn aquí. Cada tarea arranca un contenedor con la MISMA
imagen de entrenamiento que usa `python tasks.py train` (Linux canónico, Hito 3) y le
pide un comando de `src/`. Tres consecuencias:

  · El entorno de Airflow y el del modelo no se mezclan. Airflow trae sus propias
    versiones de pandas/numpy; si el entrenamiento corriera dentro de Airflow, subir
    Airflow podría cambiar el modelo.
  · Lo que corre en el DAG se puede correr y probar sin Airflow
    (`docker compose run train`, pytest).
  · En Kubernetes (Hito 6) DockerOperator se cambia por KubernetesPodOperator y el
    resto del DAG queda igual.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import dag
from docker.types import Mount

# Configuración desde el entorno de los contenedores de Airflow (docker-compose.yml), no
# escrita aquí: el mismo DAG sirve en otra máquina cambiando variables, no código.
DOCKER_URL = os.environ.get("ML_DOCKER_URL", "tcp://docker-proxy:2375")
IMAGEN = os.environ.get("ML_IMAGEN_ENTRENAMIENTO", "ml-predict-income-train:dev")
RED = os.environ.get("ML_RED_DOCKER", "ml-predict-income_default")
VOLUMEN_DATOS = os.environ.get("ML_VOLUMEN_DATOS", "ml-predict-income-datos")
MLFLOW = os.environ.get("ML_MLFLOW_URI", "http://mlflow:5000")

# Lo que reciben las tareas que hablan con MLflow. `{{ run_id }}` lo rellena Airflow en
# cada corrida: es el linaje que une "esta corrida del DAG" con "estas versiones del
# registry" (src/train.py lo guarda como tag; src/comparar.py lo busca).
ENTORNO_MLFLOW = {
    "MLFLOW_TRACKING_URI": MLFLOW,
    "ORQUESTADOR_RUN_ID": "{{ run_id }}",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "3",
}


def en_contenedor(task_id: str, comando: str, **kwargs) -> DockerOperator:
    """Una tarea = un contenedor efímero de la imagen de entrenamiento."""
    return DockerOperator(
        task_id=task_id,
        image=IMAGEN,
        command=comando,
        docker_url=DOCKER_URL,
        api_version="auto",
        # Misma red que MLflow: el contenedor lo encuentra como http://mlflow:5000.
        network_mode=RED,
        # Los datos llegan por un volumen de Docker, en SÓLO LECTURA: ninguna tarea
        # puede modificar los datos de los que aprende.
        mounts=[Mount(target="/app/data/processed", source=VOLUMEN_DATOS,
                      type="volume", read_only=True)],
        # Por omisión DockerOperator monta una carpeta temporal del host de Airflow. Aquí
        # Docker corre en otra "máquina" (detrás del proxy), esa ruta no existe allí.
        mount_tmp_dir=False,
        # Si sale bien se borra; si falla se conserva para poder inspeccionarlo.
        auto_remove="success",
        force_pull=False,          # la imagen se construye local (tasks.py preparar)
        **kwargs,
    )


@dag(
    dag_id="reentrenar",
    description="Valida datos, entrena, compara con el champion y registra un challenger",
    schedule=None,       # la ENIGH es bienal: se dispara a mano cuando hay datos nuevos
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Mexico_City"),
    catchup=False,
    max_active_runs=1,   # dos reentrenamientos a la vez registrarían versiones cruzadas
    tags=["m5", "entrenamiento"],
    doc_md=__doc__,
    default_args={
        "retries": 0,     # un dato inválido o un modelo peor no se arreglan reintentando
        # El proxy de Docker corta conexiones a los 10 min (docker-compose.yml). Mejor un
        # timeout explícito y claro que un corte de red a mitad del entrenamiento.
        "execution_timeout": timedelta(minutes=9),
    },
)
def reentrenar():
    validar = en_contenedor("validar_datos", "python -m src.validar_datos",
                            do_xcom_push=False)
    entrenar = en_contenedor("entrenar", "python -m src.train",
                             environment=ENTORNO_MLFLOW, do_xcom_push=False)
    # La última línea que imprime `comparar` (un JSON chico con versión y veredicto) queda
    # como XCom: visible en la UI, y utilizable por una tarea futura (p. ej. notificar).
    comparar = en_contenedor("comparar_con_champion", "python -m src.comparar",
                             environment=ENTORNO_MLFLOW)

    validar >> entrenar >> comparar


reentrenar()
