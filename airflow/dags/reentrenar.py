"""DAG `reentrenar`: de la ENIGH cruda a un @challenger en el registry.

    construir_dataset ──▶ validar_datos ──▶ entrenar ──▶ comparar_con_champion
     (microdato crudo      (pandera)        (registra     (evalúa ambos sobre el mismo
      → dataset; Hito 5b)                    versión)      test; marca @challenger;
                                                           NUNCA toca @champion)

Se dispara a mano, con la edición de la ENIGH como parámetro ("Trigger DAG w/ config"):
cuando INEGI publique la 2026, se copian los CSV a data/raw, `tasks.py preparar`, y se
corre con edicion=2026.

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
from airflow.sdk import Param, dag
from docker.types import Mount

# Configuración desde el entorno de los contenedores de Airflow (docker-compose.yml), no
# escrita aquí: el mismo DAG sirve en otra máquina cambiando variables, no código.
DOCKER_URL = os.environ.get("ML_DOCKER_URL", "tcp://docker-proxy:2375")
IMAGEN = os.environ.get("ML_IMAGEN_ENTRENAMIENTO", "ml-predict-income-train:dev")
RED = os.environ.get("ML_RED_DOCKER", "ml-predict-income_default")
VOLUMEN_DATOS = os.environ.get("ML_VOLUMEN_DATOS", "ml-predict-income-datos")
VOLUMEN_CRUDOS = os.environ.get("ML_VOLUMEN_CRUDOS", "ml-predict-income-crudos")
MLFLOW = os.environ.get("ML_MLFLOW_URI", "http://mlflow:5000")

# Lo que reciben las tareas que hablan con MLflow. `{{ run_id }}` lo rellena Airflow en
# cada corrida: es el linaje que une "esta corrida del DAG" con "estas versiones del
# registry" (src/train.py lo guarda como tag; src/comparar.py lo busca).
# Todas las tareas saben de qué edición se trata: src/datos.py resuelve con ella el nombre
# del dataset que se construye, se valida, se entrena y se compara.
ENTORNO = {"EDICION": "{{ params.edicion }}"}
ENTORNO_MLFLOW = {
    **ENTORNO,
    "MLFLOW_TRACKING_URI": MLFLOW,
    "ORQUESTADOR_RUN_ID": "{{ run_id }}",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "3",
}


def en_contenedor(task_id: str, comando: str, escribe_datos: bool = False,
                  **kwargs) -> DockerOperator:
    """Una tarea = un contenedor efímero de la imagen de entrenamiento.

    Sólo la tarea que CONSTRUYE el dataset puede escribir en el volumen de datos; todas
    las demás lo montan en sólo lectura. El microdato crudo es de sólo lectura para todas.
    """
    return DockerOperator(
        task_id=task_id,
        image=IMAGEN,
        command=comando,
        docker_url=DOCKER_URL,
        api_version="auto",
        # Misma red que MLflow: el contenedor lo encuentra como http://mlflow:5000.
        network_mode=RED,
        mounts=[Mount(target="/app/data/processed", source=VOLUMEN_DATOS,
                      type="volume", read_only=not escribe_datos),
                Mount(target="/app/data/raw", source=VOLUMEN_CRUDOS,
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
    params={"edicion": Param(2024, type="integer", minimum=2016,
                             title="Edición de la ENIGH",
                             description="Año de la ENIGH en data/raw (bienal: 2024, 2026…)")},
    default_args={
        "retries": 0,     # un dato inválido o un modelo peor no se arreglan reintentando
        # El proxy de Docker corta conexiones a los 10 min (docker-compose.yml). Mejor un
        # timeout explícito y claro que un corte de red a mitad del entrenamiento.
        "execution_timeout": timedelta(minutes=9),
    },
)
def reentrenar():
    construir = en_contenedor("construir_dataset", "python -m src.pipeline.construir",
                              escribe_datos=True, environment=ENTORNO)
    validar = en_contenedor("validar_datos", "python -m src.validar_datos",
                            environment=ENTORNO, do_xcom_push=False)
    entrenar = en_contenedor("entrenar", "python -m src.train",
                             environment=ENTORNO_MLFLOW, do_xcom_push=False)
    # La última línea que imprime `comparar` (un JSON chico con versión y veredicto) queda
    # como XCom: visible en la UI, y utilizable por una tarea futura (p. ej. notificar).
    comparar = en_contenedor("comparar_con_champion", "python -m src.comparar",
                             environment=ENTORNO_MLFLOW)

    construir >> validar >> entrenar >> comparar


reentrenar()
