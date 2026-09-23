"""Carga de los bundles y lógica de predicción.

Separado de la capa HTTP a propósito: esta clase no sabe nada de FastAPI y se puede
usar igual desde un DAG de Airflow, un job de backtesting o una prueba. La capa web
sólo traduce JSON a llamadas de aquí.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import ACategorias  # noqa: F401  (necesario para deserializar)

log = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
ARCHIVOS = {"ocupados": "modelo_ocupados_v1.joblib",
            "no_ocupados": "modelo_no_ocupados_v1.joblib"}

# Un pickle de scikit-learn NO es portable entre versiones menores. Comprobado: cargar
# un bundle de 1.8.0 con 1.9.1 revienta con `ModuleNotFoundError: No module named
# '_loss'`, porque 1.9 reorganizó módulos internos. Si va a fallar, que falle en el
# arranque y diciendo por qué, no a mitad de una petición.
VERIFICAR_VERSIONES = os.getenv("VERIFICAR_VERSIONES", "1") != "0"
CRITICAS = ("scikit-learn", "numpy", "pandas")


def _versiones_actuales() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for p in (*CRITICAS, "joblib"):
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
class Bundle:
    """Un modelo cargado, con todo lo que necesita para responder."""
    segmento: str
    features: list[str]
    cuantiles: dict[str, Any]
    factor_conformal: float
    familia: str
    version: str
    cobertura: float
    contrato: list[dict] = field(default_factory=list)

    @property
    def categoricas(self) -> dict[str, list[str]]:
        return {c["variable"]: c["valores"] for c in self.contrato
                if c["tipo"] == "categórica"}

    @property
    def numericas(self) -> dict[str, dict]:
        return {c["variable"]: c for c in self.contrato if c["tipo"] == "numérica"}


class Predictor:
    """Carga los dos bundles una sola vez y responde con un intervalo.

    El costo de cargar un modelo es de arranque, no de petición: se hace al levantar
    el proceso y `/ready` no devuelve 200 hasta que terminó. En Kubernetes eso es lo
    que evita que el tráfico llegue a un pod que todavía no puede responder.
    """

    def __init__(self, model_dir: Path | str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.bundles: dict[str, Bundle] = {}
        self._cargado = False

    def cargar(self) -> Predictor:
        for seg, archivo in ARCHIVOS.items():
            ruta = self.model_dir / archivo
            if not ruta.exists():
                raise FileNotFoundError(f"No se encontró el modelo: {ruta}")
            b = joblib.load(ruta)

            if VERIFICAR_VERSIONES and (decl := b.get("versiones_entrenamiento")):
                for problema in comprobar_versiones(decl):
                    log.warning("desajuste de versiones en %s — %s", archivo, problema)

            self.bundles[seg] = Bundle(
                segmento=b["segmento"],
                features=b["features"],
                cuantiles=b["cuantiles"],
                factor_conformal=float(b["factor_conformal"]),
                familia=b["familia"],
                version=b.get("version", "v1"),
                cobertura=float(b.get("cobertura_objetivo", 0.80)),
                contrato=b.get("contrato", []),
            )
            log.info("modelo cargado segmento=%s familia=%s features=%d",
                     seg, b["familia"], len(b["features"]))
        self._cargado = True
        return self

    @property
    def listo(self) -> bool:
        return self._cargado and len(self.bundles) == len(ARCHIVOS)

    def predecir(self, payload: dict, trabajo_mes_pasado: bool) -> Prediccion:
        seg = segmento_de(trabajo_mes_pasado)
        b = self.bundles[seg]

        faltan = [f for f in b.features if payload.get(f) is None]
        if faltan:
            raise ValueError(f"faltan campos para el segmento '{seg}': {faltan}")

        X = pd.DataFrame([{f: payload[f] for f in b.features}])
        for c in b.numericas:
            X[c] = pd.to_numeric(X[c])

        log_lo = float(b.cuantiles["0.1"].predict(X)[0])
        log_mid = float(b.cuantiles["0.5"].predict(X)[0])
        log_hi = float(b.cuantiles["0.9"].predict(X)[0])

        # Conformal: se ensancha en espacio log, así que el intervalo es proporcional.
        # Un ensanchamiento absoluto sería absurdo abajo y despreciable arriba.
        f = b.factor_conformal
        lo = float(np.exp(log_mid - f * (log_mid - log_lo)))
        hi = float(np.exp(log_mid + f * (log_hi - log_mid)))
        mid = float(np.exp(log_mid))

        # El q50 y los extremos vienen de tres modelos independientes: nada garantiza
        # que salgan ordenados en casos raros. Se ordena antes de responder.
        lo, mid, hi = sorted((lo, mid, hi))

        return Prediccion(
            segmento=seg, inferior=round(lo, 2), mediana=round(mid, 2),
            superior=round(hi, 2), cobertura=b.cobertura,
            modelo=b.familia, version=b.version,
        )
