"""El modelo tal como se despliega: tres cuantiles + corrección conformal + contrato.

Por qué existe este archivo. MLflow registra **un** modelo por nombre, pero lo que sirve
el producto son tres pipelines (q10, q50, q90) más un factor de ensanchamiento y el
contrato de entrada. Envolverlos en un `mlflow.pyfunc.PythonModel` hace que el registry
guarde una sola cosa que ya sabe responder el intervalo completo:

    m = mlflow.pyfunc.load_model("models:/ingreso_ocupados@champion")
    m.predict(df)   # -> inferior, mediana, superior

El alternativo —registrar q10, q50 y q90 por separado— deja al consumidor con la
responsabilidad de volver a pegarlos y de aplicar el factor conformal. Esa es
exactamente la lógica que no debe vivir fuera del artefacto.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd

ALFAS = ("0.1", "0.5", "0.9")


class ModeloIngreso(mlflow.pyfunc.PythonModel):
    """Devuelve el intervalo del 80% y la mediana condicional, en pesos."""

    def __init__(self, cuantiles: dict[str, Any], factor_conformal: float,
                 features: list[str], contrato: list[dict], meta: dict | None = None):
        self.cuantiles = cuantiles
        self.factor_conformal = float(factor_conformal)
        self.features = list(features)
        self.contrato = contrato
        self.meta = meta or {}

    # ---------------------------------------------------------------- interno
    @property
    def _numericas(self) -> list[str]:
        return [c["variable"] for c in self.contrato if c["tipo"] == "numérica"]

    def _preparar(self, X: pd.DataFrame) -> pd.DataFrame:
        faltan = [f for f in self.features if f not in X.columns]
        if faltan:
            raise ValueError(f"faltan columnas: {faltan}")
        X = X[self.features].copy()
        for c in self._numericas:
            X[c] = pd.to_numeric(X[c])
        return X

    # ---------------------------------------------------------------- pyfunc
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        X = self._preparar(pd.DataFrame(model_input))

        log_lo = self.cuantiles[ALFAS[0]].predict(X)
        log_mid = self.cuantiles[ALFAS[1]].predict(X)
        log_hi = self.cuantiles[ALFAS[2]].predict(X)

        # El ensanchamiento es multiplicativo en espacio log, así que el intervalo
        # escala con el nivel de ingreso. Uno aditivo sería absurdo abajo y
        # despreciable arriba.
        f = self.factor_conformal
        lo = np.exp(log_mid - f * (log_mid - log_lo))
        hi = np.exp(log_mid + f * (log_hi - log_mid))
        mid = np.exp(log_mid)

        # Tres modelos independientes no garantizan orden en casos raros.
        inf = np.minimum.reduce([lo, mid, hi])
        sup = np.maximum.reduce([lo, mid, hi])
        med = np.clip(mid, inf, sup)

        return pd.DataFrame({"inferior": np.round(inf, 2),
                             "mediana": np.round(med, 2),
                             "superior": np.round(sup, 2)})


def derivar_contrato(X_train: pd.DataFrame, cat: list[str],
                     preguntas: dict[str, str] | None = None) -> list[dict]:
    """Deriva el contrato de entrada de las filas de TRAIN, no de un archivo aparte.

    Es la propiedad que importa: los valores que la API acepta son exactamente los que
    el modelo vio. Si un reentrenamiento cambia las categorías, el contrato cambia con
    él en la misma corrida. Un contrato escrito a mano se desincroniza el primer día.

    El texto de cada pregunta es lo único que no se deriva de los datos: lo decidió una
    persona en el notebook 03 y llega en `preguntas`.
    """
    preguntas = preguntas or {}
    contrato = []
    for c in X_train.columns:
        if c in cat:
            contrato.append({"variable": c, "tipo": "categórica",
                             "valores": sorted(X_train[c].dropna().unique().tolist()),
                             "pregunta": preguntas.get(c, "")})
        else:
            contrato.append({"variable": c, "tipo": "numérica",
                             "min": float(np.nanmin(X_train[c])),
                             "max": float(np.nanmax(X_train[c])),
                             "mediana": float(np.nanmedian(X_train[c])),
                             "acepta_nulos": bool(X_train[c].isna().any()),
                             "pregunta": preguntas.get(c, "")})
    return contrato


def ejemplo_de_entrada(contrato: list[dict], features: list[str]) -> pd.DataFrame:
    """Una fila válida derivada del propio contrato.

    MLflow la usa para inferir la firma del modelo. Que salga del contrato y no de una
    constante escrita a mano significa que firma y validación no se pueden separar.
    """
    fila = {}
    for c in contrato:
        if c["variable"] not in features:
            continue
        fila[c["variable"]] = (c["valores"][0] if c["tipo"] == "categórica"
                               else float(c["mediana"]))
    return pd.DataFrame([fila])[features]
