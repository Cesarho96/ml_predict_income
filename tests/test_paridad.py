"""Paridad: el pipeline reproduce EXACTAMENTE el dataset que produjeron los notebooks.

    python tasks.py paridad

Es la prueba que autoriza a jubilar los notebooks 01–02 como fuente del dataset. Compara
la salida de `src/pipeline` sobre la ENIGH 2024 cruda contra
`data/processed/enigh2024_features_v2.parquet` (el del notebook 02): mismas filas, mismo
orden, mismas columnas, mismos tipos, mismos valores. No "casi iguales": iguales.

Necesita los microdatos crudos (842 MB, no versionados): en CI se salta sola.
"""

import shutil

import pandas as pd
import pytest

from src import datos
from src.pipeline.construir import construir

CRUDOS_2024 = datos.RAIZ / "data" / "raw" / "conjunto_de_datos_enigh2024_ns_csv"
NOTEBOOK = datos.RAIZ / "data" / "processed" / "enigh2024_features_v2.parquet"

pytestmark = [
    pytest.mark.paridad,
    pytest.mark.skipif(not (CRUDOS_2024.exists() and NOTEBOOK.exists()),
                       reason="sin microdatos ENIGH 2024 crudos en data/raw"),
]


def test_pipeline_igual_al_notebook(tmp_path):
    for f in ("features_ocupados.json", "features_no_ocupados.json", datos.SPLIT):
        shutil.copy(datos.RAIZ / "data" / "processed" / f, tmp_path / f)

    meta = construir(2024, crudos=datos.RAIZ / "data" / "raw", proc=tmp_path)

    nuevo = pd.read_parquet(tmp_path / datos.dataset(2024))
    viejo = pd.read_parquet(NOTEBOOK)[list(nuevo.columns)]
    pd.testing.assert_frame_equal(nuevo, viejo)          # estricto: valores Y tipos
    assert meta["upm_nuevas_particionadas"] == 0         # 2024 ya estaba particionada
