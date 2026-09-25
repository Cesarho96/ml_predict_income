"""Carga de los modelos desde el registry de MLflow y lógica de predicción.

Separado de la capa HTTP a propósito: esta clase no sabe nada de FastAPI y se puede
usar igual desde un DAG de Airflow, un job de backtesting o una prueba.

Hito 3 — de dónde sale el modelo
--------------------------------
Hasta el Hito 2 el modelo venía horneado en la imagen (`models/*.joblib`). Ahora la
imagen sólo lleva CÓDIGO y el modelo se descarga del registry al arrancar:

    MODEL_ALIAS=champion  →  models:/ingreso_ocupados@champion
                          →  se resuelve UNA vez, al arrancar, a models:/ingreso_ocupados/3
                          →  se descarga, se carga y se sirve la versión 3 hasta que el
                             proceso muera

Tres decisiones de diseño, con su porqué:

1. **El alias se resuelve una sola vez, al arrancar.** Nunca por petición. Si cada
   petición preguntara "¿quién es champion?", mover el alias a media tarde dejaría a
   las réplicas sirviendo versiones distintas según quién preguntó primero, y un mismo
   usuario podría recibir dos respuestas a la misma pregunta. Así, cambiar el alias
   afecta sólo a los procesos NUEVOS: un rollout controlado, no un cambio en caliente.

2. **Se sirve una versión fijada y se reporta.** `/ready` y cada respuesta dicen qué
   versión exacta respondió. Sin eso, los logs de predicción del Hito 7 no se podrían
   atribuir a un modelo y el drift no tendría a quién culpar.

3. **Si el registry no responde, el arranque falla.** No hay "plan B" con un modelo
   viejo en disco. Una réplica que no sabe qué versión DEBERÍA servir no debe servir
   ninguna: es preferible un pod que no arranca (visible, con alerta) a uno que
   responde con un modelo que nadie eligió. Los pods ya vivos no se enteran de una
   caída del registry: el modelo está en memoria. Lo que se bloquea es escalar o
   desplegar, y la mitigación de eso es un registry con alta disponibilidad, no un
   fallback silencioso.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# Necesarios para deserializar: el pickle guarda la ruta `src.model.ModeloIngreso` y
# `src.preprocessing.ACategorias`. Al importarlos aquí, ANTES de cargar, el paquete
# `src` de esta imagen queda en `sys.modules` y es el que se usa —no la copia que viaja
# dentro del artefacto por `code_paths` (ver `_codigo_distinto`).
from src.model import ModeloIngreso  # noqa: F401
from src.preprocessing import ACategorias  # noqa: F401

log = logging.getLogger(__name__)

NOMBRES = {"ocupados": "ingreso_ocupados", "no_ocupados": "ingreso_no_ocupados"}

# Un pickle de scikit-learn NO es portable entre versiones menores. Comprobado: cargar
# un modelo de 1.8.0 con 1.9.1 revienta con `ModuleNotFoundError: No module named
# '_loss'`. Si va a fallar, que falle en el arranque y diciendo por qué.
VERIFICAR_VERSIONES = os.getenv("VERIFICAR_VERSIONES", "1") != "0"
CRITICAS = ("scikit-learn", "numpy", "pandas")


def _versiones_actuales() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for p in CRITICAS:
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = "?"
    return out


def comprobar_versiones(declaradas: dict[str, str]) -> list[str]:
    """Devuelve las discrepancias en las librerías cuya versión sí rompe el pickle."""
    actuales = _versiones_actuales()
    return [f"{p}: el modelo se serializó con {declaradas[p]}, aquí hay {actuales.get(p)}"
            for p in CRITICAS
            if p in declaradas and actuales.get(p) not in (declaradas[p], "?")]


# El router del producto: una sola pregunta decide qué modelo responde.
def segmento_de(trabajo_mes_pasado: bool) -> str:
    return "ocupados" if trabajo_mes_pasado else "no_ocupados"


@dataclass(frozen=True)
class Prediccion:
    segmento: str
    inferior: float
    mediana: float
    superior: float
    moneda: str = "MXN"
    periodo: str = "mensual"
    cobertura: float = 0.80
    modelo: str = ""
    version: str = ""


@dataclass
class ModeloServido:
    """Un modelo cargado y la identidad exacta de lo que se cargó."""
    segmento: str
    nombre: str
    version: str            # versión del registry, o "local" si vino de una ruta
    uri: str                # SIEMPRE fijada a versión, nunca a alias
    alias: str | None
    run_id: str | None
    pyfunc: Any             # mlflow.pyfunc.PyFuncModel: aplica la firma al predecir
    interno: ModeloIngreso  # el objeto de src/model.py: contrato, features, factor

    @property
    def etiqueta(self) -> str:
        return f"{self.nombre}/{'local' if self.version == 'local' else 'v' + self.version}"

    @property
    def features(self) -> list[str]:
        return self.interno.features

    @property
    def contrato(self) -> list[dict]:
        return self.interno.contrato

    @property
    def categoricas(self) -> dict[str, list[str]]:
        return {c["variable"]: c["valores"] for c in self.contrato
                if c["tipo"] == "categórica"}

    @property
    def numericas(self) -> list[str]:
        return [c["variable"] for c in self.contrato if c["tipo"] == "numérica"]

    @property
    def familia(self) -> str:
        return self.interno.meta.get("familia", "HistGradientBoosting")

    @property
    def cobertura(self) -> float:
        return float(self.interno.meta.get("cobertura_objetivo", 0.80))


# --------------------------------------------------------------------- resolución
def resolver(segmento: str) -> tuple[str, str, str, str | None, str | None]:
    """Traduce la configuración a una URI fijada a versión.

    Devuelve (uri_fijada, nombre, version, alias, run_id).

    Configuración, de más a menos específica:
      MODEL_URI_OCUPADOS=models:/ingreso_ocupados/3    → versión exacta
      MODEL_URI_OCUPADOS=models:/ingreso_ocupados@beta → otro alias
      MODEL_URI_OCUPADOS=/ruta/a/un/modelo             → sin registry (pruebas)
      (nada)                                           → models:/<nombre>@$MODEL_ALIAS
    """
    nombre = NOMBRES[segmento]
    explicita = os.getenv(f"MODEL_URI_{segmento.upper()}", "").strip()
    if explicita and not explicita.startswith("models:/"):
        return explicita, nombre, "local", None, None

    ref = (explicita or f"models:/{nombre}@{os.getenv('MODEL_ALIAS', 'champion').strip()}")
    ref = ref.removeprefix("models:/")
    client = MlflowClient()
    try:
        if "@" in ref:
            nombre, alias = ref.split("@", 1)
            mv = client.get_model_version_by_alias(nombre, alias)
        else:
            nombre, version = ref.split("/", 1)
            alias = None
            mv = client.get_model_version(nombre, version)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST" or "not found" in str(e).lower():
            raise RuntimeError(
                f"El registry no tiene '{ref}'. ¿Ya promoviste una versión? "
                f"→ python tasks.py promover {segmento} <versión>") from e
        raise RuntimeError(
            f"No se pudo consultar el registry en {mlflow.get_tracking_uri()} "
            f"para '{ref}': {e}") from e
    return f"models:/{nombre}/{mv.version}", nombre, str(mv.version), alias, mv.run_id


def _cache() -> Path:
    return Path(os.getenv("MODEL_CACHE_DIR") or Path(tempfile.gettempdir()) / "modelos")


def descargar(uri: str, nombre: str, version: str) -> Path:
    """Baja el artefacto a una caché local por nombre/versión.

    Una versión registrada es inmutable, así que si ya está en la caché no se vuelve a
    bajar. En Kubernetes, montar un `emptyDir` en MODEL_CACHE_DIR hace que un reinicio
    del contenedor (no del pod) arranque sin tocar la red para los artefactos.

    Se descarga a un directorio temporal y se renombra al final: una descarga
    interrumpida nunca deja una caché a medias que parezca válida.
    """
    if version == "local":
        return Path(uri)
    destino = _cache() / nombre / version
    if (destino / "MLmodel").exists():
        log.info("caché: %s v%s ya estaba en %s", nombre, version, destino)
        return destino
    parcial = destino.with_name(destino.name + ".parcial")
    shutil.rmtree(parcial, ignore_errors=True)
    parcial.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(parcial))
    if not (parcial / "MLmodel").exists():
        raise RuntimeError(f"la descarga de {uri} no contiene un MLmodel en {parcial}")
    parcial.rename(destino)
    return destino


def _codigo_distinto(ruta_modelo: Path) -> list[str]:
    """Compara el `src/` que viajó dentro del artefacto con el de esta imagen.

    `code_paths` mete una copia de `src/` en el modelo, pero al servir se usa la de la
    imagen (ya está importada). Si difieren en los módulos cuyas clases están en el
    pickle, el modelo se está deserializando con código distinto al que lo entrenó.
    No siempre rompe —un comentario nuevo no importa— pero debe ser visible.

    Se ignoran los fines de línea: un modelo entrenado en Windows empaqueta el código
    con CRLF y eso no es una diferencia real.
    """
    empaquetado = ruta_modelo / "code" / "src"
    propio = Path(__file__).resolve().parent
    def leer(p: Path) -> bytes:
        return p.read_bytes().replace(b"\r\n", b"\n")

    return [archivo for archivo in ("model.py", "preprocessing.py")
            if (empaquetado / archivo).exists() and (propio / archivo).exists()
            and leer(empaquetado / archivo) != leer(propio / archivo)]


# --------------------------------------------------------------------- predictor
class Predictor:
    """Resuelve, descarga y carga los dos modelos una sola vez; responde con un intervalo.

    El costo de cargar es de arranque, no de petición: `/ready` no devuelve 200 hasta
    que terminó. En Kubernetes eso evita que llegue tráfico a un pod que aún no puede
    responder.
    """

    def __init__(self):
        self.modelos: dict[str, ModeloServido] = {}
        self._cargado = False

    def cargar(self) -> Predictor:
        for seg in NOMBRES:
            uri, nombre, version, alias, run_id = resolver(seg)
            ruta = descargar(uri, nombre, version)
            pyfunc = mlflow.pyfunc.load_model(str(ruta))
            interno = pyfunc.unwrap_python_model()

            if VERIFICAR_VERSIONES:
                for p in comprobar_versiones(interno.meta.get("versiones_entrenamiento", {})):
                    log.warning("desajuste de versiones en %s v%s — %s", nombre, version, p)
            if distintos := _codigo_distinto(ruta):
                log.warning("el código de esta imagen difiere del que entrenó %s v%s en: %s",
                            nombre, version, ", ".join(distintos))

            self.modelos[seg] = ModeloServido(seg, nombre, version, uri, alias, run_id,
                                              pyfunc, interno)
            m = self.modelos[seg]
            log.info("modelo cargado segmento=%s modelo=%s alias=%s run=%s features=%d",
                     seg, m.etiqueta, alias, run_id, len(m.features))
        self._cargado = True
        return self

    @property
    def listo(self) -> bool:
        return self._cargado and len(self.modelos) == len(NOMBRES)

    def predecir(self, payload: dict, trabajo_mes_pasado: bool) -> Prediccion:
        seg = segmento_de(trabajo_mes_pasado)
        m = self.modelos[seg]

        faltan = [f for f in m.features if payload.get(f) is None]
        if faltan:
            raise ValueError(f"faltan campos para el segmento '{seg}': {faltan}")

        X = pd.DataFrame([{f: payload[f] for f in m.features}])
        for c in m.numericas:
            X[c] = pd.to_numeric(X[c]).astype("float64")

        # Toda la lógica —cuantiles, conformal, orden— vive en ModeloIngreso.predict,
        # la misma función que se evaluó en el entrenamiento. Hasta el Hito 2 esta clase
        # la duplicaba; dos copias de una fórmula terminan divergiendo.
        s = m.pyfunc.predict(X).iloc[0]
        return Prediccion(
            segmento=seg, inferior=float(s["inferior"]), mediana=float(s["mediana"]),
            superior=float(s["superior"]), cobertura=m.cobertura,
            modelo=m.familia, version=m.etiqueta,
        )
