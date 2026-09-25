"""Modelos sintéticos para las pruebas.

Desde el Hito 3 el modelo real no está en el repo: vive en el registry. Las pruebas no
pueden depender de él —en CI no hay registry ni microdatos de la ENIGH— y tampoco
deberían: prueban el CÓDIGO (carga, contrato, ruteo, HTTP), no un modelo concreto. Lo
que valida al modelo real está en `src/validacion.py` y corre al entrenar.

Así que aquí se fabrican modelos diminutos con la MISMA estructura que los reales
(mismas features, mismas categorías que usan las pruebas, mismo `ModeloIngreso`) y se
guardan en formato MLflow. Entrenan en menos de un segundo.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.model import ALFAS, ModeloIngreso, derivar_contrato, ejemplo_de_entrada
from src.predictor import _versiones_actuales
from src.preprocessing import ACategorias

SEGMENTOS = {
    "ocupados": {
        "features": ["horas_trabajadas", "formalidad", "cve_entidad", "anios_escolaridad",
                     "sinco_grupo", "posicion_ocupacion", "es_jefe_hogar",
                     "sector_agrupado", "es_mujer", "tam_empresa_grupo"],
        "categorias": {
            "formalidad": ["1 Informal sin prestaciones",
                           "2 Informal con algunas prestaciones",
                           "3 Formal (con seguridad social)"],
            "cve_entidad": ["09", "15", "19"],
            "sinco_grupo": ["2 Profesionistas y técnicos",
                            "9 Trabajadores en actividades elementales y de apoyo"],
            "posicion_ocupacion": ["Subordinado remunerado", "Trabajador por cuenta propia"],
            "sector_agrupado": ["1 Agropecuario", "6 Servicios profesionales y financieros"],
            "tam_empresa_grupo": ["1 Micro (1-10)", "3 Mediana (51-250)"],
        },
        "numericas": {"horas_trabajadas": (1, 80), "anios_escolaridad": (0, 22),
                      "es_jefe_hogar": (0, 1), "es_mujer": (0, 1)},
    },
    "no_ocupados": {
        "features": ["actividad_no_ocupado", "es_jefe_hogar", "anios_escolaridad", "edad",
                     "cve_entidad", "integrantes_hogar", "situacion_conyugal"],
        "categorias": {
            "actividad_no_ocupado": ["Pensionado(a)/jubilado(a)", "Quehaceres del hogar"],
            "cve_entidad": ["09", "15", "19"],
            "situacion_conyugal": ["Casado(a)", "Soltero(a)"],
        },
        "numericas": {"es_jefe_hogar": (0, 1), "anios_escolaridad": (0, 22),
                      "edad": (18, 90), "integrantes_hogar": (1, 10)},
    },
}


def fabricar_modelo(segmento: str, efecto_escolaridad: float = 0.08,
                    factor: float = 1.1, semilla: int = 0) -> ModeloIngreso:
    """Un ModeloIngreso real, entrenado en datos sintéticos con un efecto conocido."""
    spec = SEGMENTOS[segmento]
    rng = np.random.default_rng(semilla)
    n = 600
    X = pd.DataFrame({
        f: (rng.choice(spec["categorias"][f], n) if f in spec["categorias"]
            else rng.integers(*spec["numericas"][f], endpoint=True, size=n).astype("float64"))
        for f in spec["features"]})
    y = 8.0 + efecto_escolaridad * X["anios_escolaridad"] + rng.normal(0, 0.4, n)

    cat = list(spec["categorias"])
    cuantiles = {}
    for alfa in ALFAS:
        cuantiles[alfa] = Pipeline([
            ("prep", ACategorias(cat)),
            ("modelo", HistGradientBoostingRegressor(
                loss="quantile", quantile=float(alfa), max_iter=40,
                min_samples_leaf=20, categorical_features="from_dtype", random_state=0)),
        ]).fit(X, y)
    return ModeloIngreso(cuantiles, factor, spec["features"], derivar_contrato(X, cat),
                         meta={"segmento": segmento, "familia": "HistGradientBoosting",
                               "cobertura_objetivo": 0.80,
                               "versiones_entrenamiento": _versiones_actuales()})


def firma_de(m: ModeloIngreso):
    ejemplo = ejemplo_de_entrada(m.contrato, m.features)
    return infer_signature(ejemplo, m.predict(None, ejemplo)), ejemplo


# Requisitos explícitos: si no se dan, MLflow los INFIERE levantando un subproceso que
# carga el modelo y rastrea imports. Son ~5 s por modelo; en las pruebas no aportan nada.
REQUISITOS = [f"{p}=={v}" for p, v in _versiones_actuales().items()]


def guardar(m: ModeloIngreso, ruta: Path, code_paths: list[str] | None = None) -> Path:
    firma, ejemplo = firma_de(m)
    mlflow.pyfunc.save_model(str(ruta), python_model=m, signature=firma,
                             input_example=ejemplo, code_paths=code_paths,
                             pip_requirements=REQUISITOS)
    return ruta
