"""DAG de prueba: comprueba que Airflow está vivo y enseña lo mínimo de Airflow 3.

Dos tareas con la API TaskFlow (`@task`): la primera devuelve un valor, la segunda lo
recibe. Airflow lo pasa entre ellas como XCom — un valor chico que se guarda en la base de
metadatos. Sirve para textos e IDs, NO para datos: un DataFrame por XCom es el error
clásico. Los datos viajan por almacenamiento (un volumen, S3); por XCom sólo viaja su
ubicación o su identificador.

Se puede borrar cuando `reentrenar` funcione.
"""

import pendulum
from airflow.sdk import dag, task


@dag(
    dag_id="hola",
    schedule=None,          # sólo se corre a mano; no hay nada que hacer cada hora
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Mexico_City"),
    catchup=False,
    tags=["m5", "prueba"],
)
def hola():
    @task
    def saludar() -> str:
        import platform

        mensaje = f"Hola desde Airflow, corriendo en {platform.node()} ({platform.system()})"
        print(mensaje)      # aparece en el log de la tarea, en la UI
        return mensaje      # esto es lo que viaja como XCom

    @task
    def responder(mensaje: str) -> None:
        print(f"Recibí por XCom: {mensaje!r}")

    responder(saludar())


hola()
