"""El criterio de "hecho" del Hito 3, como prueba automática.

    Hecho cuando: cambiar el alias en MLflow cambia lo que sirven los pods NUEVOS
    —y no lo que sirven los que ya están vivos.

Usa un registry real de MLflow sobre SQLite en una carpeta temporal: el mismo código de
resolución que corre en producción, sin levantar ningún servidor.
"""

import mlflow
import pytest
from mlflow import MlflowClient

from src.predictor import NOMBRES, Predictor
from tests.fabrica import REQUISITOS, fabricar_modelo, firma_de

OCUPADO = dict(horas_trabajadas=48, formalidad="3 Formal (con seguridad social)",
               cve_entidad="09", anios_escolaridad=16,
               sinco_grupo="2 Profesionistas y técnicos",
               posicion_ocupacion="Subordinado remunerado", es_jefe_hogar=1,
               sector_agrupado="6 Servicios profesionales y financieros",
               es_mujer=0, tam_empresa_grupo="3 Mediana (51-250)")


@pytest.fixture(scope="module")
def _registro(tmp_path_factory):
    """Se crea UNA vez por módulo: registrar es lo lento. v2 tiene un intervalo más ancho."""
    base = tmp_path_factory.mktemp("registry")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{base / 'mlflow.db'}")
        exp = mlflow.create_experiment("pruebas", artifact_location=(base / "art").as_uri())
        for factor in (1.1, 1.6):                       # v1, v2
            for seg, nombre in NOMBRES.items():
                m = fabricar_modelo(seg, factor=factor)
                firma, ejemplo = firma_de(m)
                with mlflow.start_run(experiment_id=exp):
                    mlflow.pyfunc.log_model(name="modelo", python_model=m, signature=firma,
                                            input_example=ejemplo,
                                            pip_requirements=REQUISITOS,
                                            registered_model_name=nombre)
    return f"sqlite:///{base / 'mlflow.db'}"


@pytest.fixture
def registry(_registro, tmp_path, monkeypatch):
    """Cada prueba arranca sin aliases, con su propia caché y sin rutas locales."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _registro)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("MODEL_URI_OCUPADOS", "MODEL_URI_NO_OCUPADOS", "MODEL_ALIAS"):
        monkeypatch.delenv(var, raising=False)
    client = MlflowClient()
    for nombre in NOMBRES.values():
        for alias in client.get_registered_model(nombre).aliases:
            client.delete_registered_model_alias(nombre, alias)
    return client


def test_sin_champion_el_arranque_falla_y_dice_que_hacer(registry):
    with pytest.raises(RuntimeError, match="promover"):
        Predictor().cargar()


def test_mover_el_alias_cambia_los_pods_nuevos_no_los_vivos(registry):
    for nombre in NOMBRES.values():
        registry.set_registered_model_alias(nombre, "champion", "1")

    vivo = Predictor().cargar()
    m = vivo.modelos["ocupados"]
    assert (m.version, m.alias, m.uri) == ("1", "champion", "models:/ingreso_ocupados/1")
    antes = vivo.predecir(OCUPADO, True)
    assert antes.version == "ingreso_ocupados/v1"

    # Promoción: champion pasa a v2.
    registry.set_registered_model_alias("ingreso_ocupados", "champion", "2")

    # El proceso que ya estaba vivo sigue sirviendo v1: el alias se resolvió al arrancar.
    assert vivo.predecir(OCUPADO, True) == antes

    # Un proceso nuevo sirve v2, y se nota: su intervalo es más ancho.
    nuevo = Predictor().cargar()
    despues = nuevo.predecir(OCUPADO, True)
    assert nuevo.modelos["ocupados"].version == "2"
    assert despues.version == "ingreso_ocupados/v2"
    assert despues.superior - despues.inferior > antes.superior - antes.inferior

    # Rollback: el mismo gesto en sentido contrario.
    registry.set_registered_model_alias("ingreso_ocupados", "champion", "1")
    assert Predictor().cargar().predecir(OCUPADO, True) == antes


def test_version_fijada_ignora_el_alias(registry, monkeypatch):
    """MODEL_URI_* con versión exacta: para reproducir un incidente con el modelo de ese día."""
    registry.set_registered_model_alias("ingreso_no_ocupados", "champion", "1")
    monkeypatch.setenv("MODEL_URI_OCUPADOS", "models:/ingreso_ocupados/2")
    p = Predictor().cargar()
    assert p.modelos["ocupados"].version == "2" and p.modelos["ocupados"].alias is None


def test_la_cache_no_vuelve_a_descargar(registry, tmp_path):
    for nombre in NOMBRES.values():
        registry.set_registered_model_alias(nombre, "champion", "1")
    Predictor().cargar()
    marca = tmp_path / "cache" / "ingreso_ocupados" / "1" / "MLmodel"
    assert marca.exists()
    antes = marca.stat().st_mtime_ns
    Predictor().cargar()
    assert marca.stat().st_mtime_ns == antes
