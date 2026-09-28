"""Dónde están los datos y de qué edición de la ENIGH se trata. Un solo lugar.

La edición llega por la variable EDICION (el DAG la pasa como parámetro de la corrida);
sin ella es 2024. Todo lo que lee o escribe el dataset —construir, validar, entrenar,
comparar— resuelve el nombre del archivo aquí, así que cambiar de edición es cambiar una
variable, no buscar "2024" por todo el repo.
"""

from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PROC = Path(os.getenv("ML_DATOS", RAIZ / "data" / "processed"))
CRUDOS = Path(os.getenv("ML_CRUDOS", RAIZ / "data" / "raw"))
EDICION = int(os.getenv("EDICION", "2024"))
SPLIT = "split_upm_3way.csv"


def dataset(edicion: int = EDICION) -> str:
    return f"enigh{edicion}_features_v2.parquet"
