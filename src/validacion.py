"""Comprobaciones de sensatez que un modelo debe pasar ANTES de registrarse.

Antes vivían en `tests/test_predictor.py` y corrían contra el `.joblib` versionado en
git. Con el Hito 3 el modelo ya no está en el repo —vive en el registry—, así que una
prueba unitaria ya no puede verlo. Y está bien: estas comprobaciones no prueban el
CÓDIGO, prueban un MODELO concreto. Su lugar es el pipeline de entrenamiento, justo
antes de `log_model`: un modelo que no las pasa no llega al registry y, por lo tanto,
nadie puede promoverlo por accidente.

No son un gate de calidad (eso es el Hito 8: el challenger contra el champion con la
banda de ruido en mente). Son lo mínimo: que el modelo no esté roto de una forma que
cualquier persona del dominio vería en cinco segundos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import ModeloIngreso, ejemplo_de_entrada


class ModeloInvalido(Exception):
    """El modelo no pasa las comprobaciones de sensatez y no debe registrarse."""


def validar(modelo: ModeloIngreso, X_test: pd.DataFrame) -> list[str]:
    """Devuelve la lista de comprobaciones que pasaron. Lanza `ModeloInvalido` si alguna falla."""
    ok = []
    salida = modelo.predict(None, X_test)
    inf, med, sup = (salida[c].to_numpy() for c in ("inferior", "mediana", "superior"))

    if not np.isfinite(salida.to_numpy()).all():
        raise ModeloInvalido("hay predicciones no finitas en test")
    ok.append("predicciones finitas")

    if not ((inf <= med) & (med <= sup)).all():
        raise ModeloInvalido("hay intervalos desordenados en test")
    ok.append("intervalos ordenados")

    if (inf <= 0).any():
        raise ModeloInvalido("hay límites inferiores ≤ 0 pesos")
    ok.append("límite inferior positivo")

    # Dominio: el retorno a la escolaridad es positivo. Se evalúa sobre el perfil
    # mediano del contrato, no sobre un caso escrito a mano, para que aplique igual a
    # los dos segmentos y a cualquier reentrenamiento.
    if "anios_escolaridad" in modelo.features:
        base = ejemplo_de_entrada(modelo.contrato, modelo.features)
        bajo = modelo.predict(None, base.assign(anios_escolaridad=6.0))["mediana"].iloc[0]
        alto = modelo.predict(None, base.assign(anios_escolaridad=17.0))["mediana"].iloc[0]
        if not alto > bajo:
            raise ModeloInvalido(
                f"17 años de escolaridad (${alto:,.0f}) no supera a 6 (${bajo:,.0f})")
        ok.append(f"escolaridad 6→17 años: ${bajo:,.0f}→${alto:,.0f}")

    return ok
