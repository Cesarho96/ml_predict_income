"""Fixtures compartidas. Los modelos sintéticos se fabrican en `tests/fabrica.py`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fabrica import SEGMENTOS, fabricar_modelo, guardar


@pytest.fixture(scope="session")
def rutas_modelos(tmp_path_factory) -> dict[str, Path]:
    """Los dos modelos sintéticos guardados en disco, en formato MLflow."""
    base = tmp_path_factory.mktemp("modelos")
    return {seg: guardar(fabricar_modelo(seg), base / seg) for seg in SEGMENTOS}


@pytest.fixture(scope="session", autouse=True)
def modelos_por_ruta(rutas_modelos):
    """Por defecto, la API carga los modelos sintéticos por ruta: sin registry.

    Es el mismo mecanismo que usarías para servir un modelo sin MLflow en un entorno
    aislado: MODEL_URI_<SEGMENTO> apuntando a una carpeta.
    """
    previo = {k: os.environ.get(k) for k in ("MODEL_URI_OCUPADOS", "MODEL_URI_NO_OCUPADOS")}
    os.environ["MODEL_URI_OCUPADOS"] = str(rutas_modelos["ocupados"])
    os.environ["MODEL_URI_NO_OCUPADOS"] = str(rutas_modelos["no_ocupados"])
    yield
    for k, v in previo.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
