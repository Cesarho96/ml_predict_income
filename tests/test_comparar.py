"""`comparar` marca el @challenger con el veredicto correcto y NUNCA toca @champion."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.comparar import EXPERIMENTO, comparar, veredicto
from src.predictor import NOMBRES
from tests.fabrica import REQUISITOS, SEGMENTOS, fabricar_modelo, firma_de

ETIQUETA = {"ocupados": "Ocupados", "no_ocupados": "No ocupados"}


# ------------------------------------------------------------------ regla de decisión
@pytest.mark.parametrize("delta,cobertura,esperado", [
    (-1.00, 80, "mejor"),          # mejora mayor que la banda
    (-0.50, 80, "equivalente"),    # "mejora" dentro del ruido: empate
    (+0.50, 80, "equivalente"),
    (+1.00, 80, "peor"),
    (-2.00, 75, "peor (cobertura)"),  # mejor MdAPE no compensa un intervalo que miente
    (None, 80, "sin champion"),
])
def test_veredicto(delta, cobertura, esperado):
    assert veredicto(delta, 0.78, cobertura) == esperado


# ------------------------------------------------------------------ de punta a punta
def _datos(tmp_path):
    """Parquet + split con la forma que lee src.train.cargar_datos, para ambos segmentos."""
    rng = np.random.default_rng(1)
    partes = []
    for slug, spec in SEGMENTOS.items():
        n = 900
        d = pd.DataFrame({
            f: (rng.choice(spec["categorias"][f], n) if f in spec["categorias"]
                else rng.integers(*spec["numericas"][f], endpoint=True, size=n).astype(float))
            for f in spec["features"]})
        d["ingreso_mensual"] = np.exp(8 + 0.08 * d["anios_escolaridad"] + rng.normal(0, .4, n))
        d["segmento"] = ETIQUETA[slug]
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    df["factor"] = 1
    df["upm"] = [f"{i:07d}" for i in range(len(df))]
    df.to_parquet(tmp_path / "enigh2024_features_v2.parquet")
    pd.DataFrame({"upm": df["upm"], "particion": np.where(np.arange(len(df)) % 3, "train", "test")}
                 ).to_csv(tmp_path / "split_upm_3way.csv", index=False)
    return tmp_path


def _registrar(tags: dict, factor: float = 1.4):
    for slug, nombre in NOMBRES.items():
        m = fabricar_modelo(slug, factor=factor)
        firma, ejemplo = firma_de(m)
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(EXPERIMENTO)
                              .experiment_id):
            mlflow.set_tags({"segmento": slug, **tags})
            mlflow.pyfunc.log_model(name="modelo", python_model=m, signature=firma,
                                    input_example=ejemplo, pip_requirements=REQUISITOS,
                                    registered_model_name=nombre)


def _aliases(client, nombre) -> dict[str, str]:
    # MLflow devuelve la versión del alias como int o str según el backend: se normaliza.
    return {k: str(v) for k, v in client.get_registered_model(nombre).aliases.items()}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(EXPERIMENTO, artifact_location=(tmp_path / "art").as_uri())
    return MlflowClient()


def test_con_champion_marca_challenger_y_no_toca_champion(registry, tmp_path):
    datos = _datos(tmp_path)
    _registrar({})                                   # v1: el que hoy sirve
    for nombre in NOMBRES.values():
        registry.set_registered_model_alias(nombre, "champion", "1")
    _registrar({"orquestador_run_id": "rid-1"})      # v2: lo que entrenó el DAG

    resultados = comparar("rid-1", proc=datos, client=registry)

    for r in resultados:
        # Mismo modelo, mismos datos: empate exacto, y un empate también es candidato.
        assert (r.version_nueva, r.version_champion, r.delta_pp) == ("2", "1", 0.0)
        assert r.veredicto == "equivalente" and r.marcado_challenger
        assert _aliases(registry, r.nombre) == {"champion": "1", "challenger": "2"}
        tags = registry.get_model_version(r.nombre, "2").tags
        assert tags["veredicto"] == "equivalente" and tags["orquestador_run_id"] == "rid-1"


def test_sin_champion_tambien_deja_candidato(registry, tmp_path):
    datos = _datos(tmp_path)
    _registrar({"orquestador_run_id": "rid-1"})
    for r in comparar("rid-1", proc=datos, client=registry):
        assert r.veredicto == "sin champion" and r.marcado_challenger
        assert _aliases(registry, r.nombre) == {"challenger": "1"}


def test_encuentra_su_corrida_aunque_haya_otra_mas_nueva(registry, tmp_path):
    """Linaje, no "la versión más nueva": un entrenamiento manual posterior no se cuela."""
    datos = _datos(tmp_path)
    _registrar({"orquestador_run_id": "rid-1"})      # v1, del DAG
    _registrar({})                                   # v2, manual y más nuevo
    for r in comparar("rid-1", proc=datos, client=registry):
        assert r.version_nueva == "1"
