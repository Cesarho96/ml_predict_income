"""El gate de promoción (Hito 8): promueve sólo con toda la evidencia, y sabe decir que no.

Lo que se prueba es lo que haría peligroso automatizar la promoción: promover un empate,
un modelo irreproducible o uno comparado contra un champion que ya cambió; promover dos
veces al reintentar; o promover sin dejar rastro ni camino de regreso.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from src import promocion
from src.comparar import EXPERIMENTO, comparar
from src.predictor import NOMBRES
from src.promocion import Evidencia, criterios, decidir, mover_champion
from tests.fabrica import REQUISITOS, fabricar_modelo, firma_de
from tests.test_comparar import _aliases, _datos

# ------------------------------------------------------------------ la regla, sin registry
BUENA = Evidencia(version="2", veredicto="mejor", cobertura=80.4, champion_comparado="1",
                  git_sha="7497b42", origen="airflow", edicion=2024, degradado=False)


def fallan(champion: str | None = "1", ed_champ: int | None = 2024, **cambios) -> set[str]:
    return {c.nombre for c in criterios(replace(BUENA, **cambios), champion, ed_champ)
            if not c.cumple}


def test_con_toda_la_evidencia_se_promueve():
    assert fallan() == set()


@pytest.mark.parametrize("cambio,falla", [
    ({"veredicto": "equivalente"}, "veredicto"),     # el caso del drift: empate
    ({"veredicto": "peor"}, "veredicto"),
    ({"veredicto": None}, "veredicto"),              # comparar no dejó evidencia
    ({"cobertura": 77.9}, "cobertura"),              # el intervalo ya no cubre el 80 %
    ({"cobertura": None}, "cobertura"),
    ({"git_sha": "7497b42-dirty"}, "commit limpio"),
    ({"git_sha": "desconocido"}, "commit limpio"),
    ({"origen": "manual"}, "orquestador"),
    ({"edicion": 2022}, "edición"),                  # nunca regresar a datos más viejos
    ({"degradado": True}, "no degradado"),
])
def test_cada_criterio_por_si_solo_rechaza(cambio, falla):
    assert fallan(**cambio) == {falla}


def test_un_veredicto_contra_otro_champion_ya_no_vale():
    """Alguien movió @champion entre `comparar` y el gate: la comparación es vieja."""
    assert fallan(champion="3") == {"champion actual"}


def test_el_primer_champion_lo_nombra_una_persona():
    assert "champion actual" in fallan(champion=None, veredicto="sin champion")


def test_un_champion_sin_edicion_registrada_no_bloquea():
    assert fallan(ed_champ=None) == set()     # versiones previas al Hito 5b


def test_se_reportan_todas_las_razones_no_solo_la_primera():
    assert fallan(veredicto="equivalente", git_sha="x-dirty", origen="manual") == {
        "veredicto", "commit limpio", "orquestador"}


# ------------------------------------------------------------------ registry real (SQLite)
LIMPIO = {"origen": "airflow", "git_sha": "7497b42"}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(EXPERIMENTO, artifact_location=(tmp_path / "art").as_uri())
    return MlflowClient()


def _registrar(tags: dict, efecto: float = 0.08) -> None:
    """Una versión por segmento, con la forma de lo que registra src/train.py."""
    exp = mlflow.get_experiment_by_name(EXPERIMENTO).experiment_id
    for slug, nombre in NOMBRES.items():
        m = fabricar_modelo(slug, efecto_escolaridad=efecto, factor=1.4)
        firma, ejemplo = firma_de(m)
        with mlflow.start_run(experiment_id=exp):
            mlflow.set_tags({"segmento": slug, **tags})
            mlflow.log_param("edicion_enigh", 2024)
            mlflow.pyfunc.log_model(name="modelo", python_model=m, signature=firma,
                                    input_example=ejemplo, pip_requirements=REQUISITOS,
                                    registered_model_name=nombre)


def _como_comparar(client, veredicto: str, version="2", champion="1", segmentos=NOMBRES):
    """Los tags que deja src/comparar.py, sin evaluar modelos (eso lo prueba test_comparar)."""
    for nombre in (NOMBRES[s] for s in segmentos):
        for k, v in {"veredicto": veredicto, "cobertura_test": "80.4",
                     "champion_comparado": champion}.items():
            client.set_model_version_tag(nombre, version, k, v)
        client.set_registered_model_alias(nombre, "challenger", version)


def _escenario(client) -> None:
    """v1 sirviendo; v2 recién entrenada por el DAG (corrida rid-1)."""
    _registrar({})
    for nombre in NOMBRES.values():
        client.set_registered_model_alias(nombre, "champion", "1")
    _registrar({**LIMPIO, "orquestador_run_id": "rid-1"})


def test_promueve_deja_anterior_retira_challenger_y_deja_rastro(registry):
    _escenario(registry)
    _como_comparar(registry, "mejor")

    for d in decidir("rid-1", client=registry):
        assert d.promovida and d.champion_antes == "1"
        assert _aliases(registry, d.nombre) == {"champion": "2", "anterior": "1"}
        t = registry.get_model_version(d.nombre, "2").tags
        assert (t["gate"], t["promovida_por"], t["reemplaza"]) == ("promovida", "gate", "1")
        assert t["gate_corrida"] == "rid-1"
        assert registry.get_model_version(d.nombre, "1").tags["reemplazada_por"] == "2"


def test_un_empate_no_se_promueve_y_dice_por_que(registry):
    _escenario(registry)
    _como_comparar(registry, "equivalente")

    for d in decidir("rid-1", client=registry):
        assert not d.promovida
        assert _aliases(registry, d.nombre) == {"champion": "1", "challenger": "2"}
        t = registry.get_model_version(d.nombre, "2").tags
        assert t["gate"] == "rechazada" and "equivalente" in t["gate_motivo"]


def test_simular_no_escribe_nada(registry):
    _escenario(registry)
    _como_comparar(registry, "mejor")

    assert all(d.promovida for d in decidir("rid-1", client=registry, simular=True))
    for nombre in NOMBRES.values():
        assert _aliases(registry, nombre) == {"champion": "1", "challenger": "2"}
        assert "gate" not in registry.get_model_version(nombre, "2").tags


def test_reintentar_la_misma_corrida_no_promueve_dos_veces(registry):
    """Si Airflow reintenta, @anterior NO debe pasar a apuntar al champion recién puesto."""
    _escenario(registry)
    _como_comparar(registry, "mejor")
    decidir("rid-1", client=registry)

    for d in decidir("rid-1", client=registry):
        assert d.promovida and "reintento" in d.motivo
        assert _aliases(registry, d.nombre) == {"champion": "2", "anterior": "1"}


def test_cada_segmento_se_decide_por_separado(registry):
    _escenario(registry)
    _como_comparar(registry, "mejor", segmentos=["ocupados"])
    _como_comparar(registry, "equivalente", segmentos=["no_ocupados"])

    decidir("rid-1", client=registry)
    assert _aliases(registry, "ingreso_ocupados")["champion"] == "2"
    assert _aliases(registry, "ingreso_no_ocupados")["champion"] == "1"


def test_promover_a_mano_deja_el_mismo_rastro(registry):
    """scripts/promover.py usa mover_champion: @anterior y sin @challenger colgado."""
    _escenario(registry)
    _como_comparar(registry, "equivalente")
    nombre = NOMBRES["ocupados"]

    assert mover_champion(registry, nombre, "2", "manual", "revisado") == "1"
    assert _aliases(registry, nombre) == {"champion": "2", "anterior": "1"}
    assert registry.get_model_version(nombre, "2").tags["promovida_por"] == "manual"


def test_de_punta_a_punta_con_comparar(registry, tmp_path):
    """El ciclo con modelos de verdad: uno claramente mejor pasa; el mismo otra vez, no."""
    datos = _datos(tmp_path)
    _registrar({}, efecto=0.0)          # v1, champion malo: ignora la escolaridad
    for nombre in NOMBRES.values():
        registry.set_registered_model_alias(nombre, "champion", "1")

    _registrar({**LIMPIO, "orquestador_run_id": "rid-1"})              # v2: el bueno
    for r in comparar("rid-1", proc=datos, client=registry):
        assert r.veredicto == "mejor", r
    assert all(d.promovida for d in decidir("rid-1", client=registry))

    _registrar({**LIMPIO, "orquestador_run_id": "rid-2"})              # v3 = v2 otra vez
    comparar("rid-2", proc=datos, client=registry)
    for d in decidir("rid-2", client=registry):
        assert not d.promovida and d.champion_antes == "2"
        assert "equivalente" in d.motivo
        assert _aliases(registry, d.nombre) == {"champion": "2", "anterior": "1",
                                                "challenger": "3"}


# ------------------------------------------------------------------ como lo corre Airflow
def test_el_codigo_de_salida_distingue_promover_de_rechazar(registry):
    _escenario(registry)
    _como_comparar(registry, "equivalente")
    with pytest.raises(SystemExit) as e:
        promocion.main(["--corrida", "rid-1"])
    assert e.value.code == promocion.SALTAR                 # rechazo → skipped

    _registrar({**LIMPIO, "orquestador_run_id": "rid-2"})   # v3
    _como_comparar(registry, "mejor", version="3")
    with pytest.raises(SystemExit) as e:
        promocion.main(["--corrida", "rid-2"])
    assert e.value.code == 0                                # promovió → verde

    with pytest.raises(SystemExit) as e:                    # sin versión → FALLA
        promocion.main(["--corrida", "no-existe"])
    assert isinstance(e.value.code, str) and "no se pudo decidir" in e.value.code


def test_el_dag_salta_con_el_mismo_codigo_que_usa_el_modulo():
    """Si estos dos números difieren, un rechazo se vuelve un FALLO del DAG."""
    dag = Path(__file__).resolve().parent.parent / "airflow" / "dags" / "reentrenar.py"
    m = re.search(r"^SALTAR = (\d+)", dag.read_text(encoding="utf-8"), re.MULTILINE)
    assert m and int(m.group(1)) == promocion.SALTAR
