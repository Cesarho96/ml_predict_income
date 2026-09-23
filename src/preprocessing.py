"""Transformadores que viajan DENTRO del pipeline serializado.

Este módulo existe por una razón concreta y es la primera lección del despliegue:
`joblib`/`pickle` no guardan el código de una clase, guardan su **ruta de importación**.
Mientras `ACategorias` vivió en una celda del notebook, esa ruta era
`__main__.ACategorias`, y el bundle sólo se podía abrir desde ese mismo kernel. En
cualquier otro proceso —un worker de FastAPI, un contenedor, un DAG de Airflow— la
carga falla con:

    AttributeError: Can't get attribute 'ACategorias' on <module '__main__'>

La regla general: **todo lo que se serialice debe vivir en un módulo importable y
versionado**, nunca en un notebook. `scripts/migrate_bundles.py` hace la migración de
los bundles que ya existen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ACategorias"]


class ACategorias(BaseEstimator, TransformerMixin):
    """Deja el DataFrame intacto pero con las columnas indicadas como dtype `category`.

    Las categorías se fijan en `fit`: un nivel no visto se convierte en NaN en
    `transform`, que es exactamente lo que los modelos de boosting saben rutear. Esto
    degrada en vez de tronar, pero la API valida antes de llegar aquí: una categoría
    desconocida es un error del cliente, no algo que el modelo deba adivinar.
    """

    def __init__(self, cat=()):
        # `clone` de scikit-learn exige que __init__ guarde los parámetros sin modificarlos.
        self.cat = cat

    def fit(self, X: pd.DataFrame, y=None) -> ACategorias:
        self.cat_ = list(self.cat)
        self.categorias_ = {
            c: pd.Index(sorted(X[c].astype(str).dropna().unique())) for c in self.cat_
        }
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categorias_[c])
        return X
