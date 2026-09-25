"""Pruebas de la capa de predicción, sin HTTP de por medio."""
import math
import shutil

import pandas as pd
import pytest

from src.model import ejemplo_de_entrada
from src.predictor import Predictor, _codigo_distinto, comprobar_versiones, segmento_de
from src.preprocessing import ACategorias
from src.validacion import ModeloInvalido, validar
from tests.fabrica import SEGMENTOS, fabricar_modelo, guardar

OCUPADO = dict(horas_trabajadas=48, formalidad="3 Formal (con seguridad social)",
               cve_entidad="09", anios_escolaridad=16,
               sinco_grupo="2 Profesionistas y técnicos",
               posicion_ocupacion="Subordinado remunerado", es_jefe_hogar=1,
               sector_agrupado="6 Servicios profesionales y financieros",
               es_mujer=0, tam_empresa_grupo="3 Mediana (51-250)")


def test_router():
    assert segmento_de(True) == "ocupados"
    assert segmento_de(False) == "no_ocupados"


def test_carga_por_ruta_sin_registry():
    """MODEL_URI_* con una ruta: sirve sin MLflow server. Así corren estas pruebas en CI."""
    p = Predictor().cargar()
    assert p.listo
    assert p.modelos["ocupados"].version == "local"
    assert len(p.modelos["ocupados"].features) == 10
    assert len(p.modelos["no_ocupados"].features) == 7


def test_intervalo_es_finito_y_ordenado():
    p = Predictor().cargar()
    r = p.predecir(dict(actividad_no_ocupado="Pensionado(a)/jubilado(a)",
                        es_jefe_hogar=1, anios_escolaridad=12, edad=70,
                        cve_entidad="19", integrantes_hogar=2,
                        situacion_conyugal="Casado(a)"), False)
    assert all(math.isfinite(v) for v in (r.inferior, r.mediana, r.superior))
    assert r.inferior <= r.mediana <= r.superior


def test_campos_enteros_no_rompen_la_firma():
    """La firma de MLflow declara double; la API manda `es_jefe_hogar=1` como int."""
    r = Predictor().cargar().predecir(OCUPADO, True)
    assert r.mediana > 0


def test_deteccion_de_desajuste_de_versiones():
    """Historia: sklearn 1.9.1 no abre un pickle de 1.8.0 (`No module named '_loss'`)."""
    from src.predictor import _versiones_actuales

    assert comprobar_versiones(_versiones_actuales()) == []
    problemas = comprobar_versiones({"scikit-learn": "0.1.0"})
    assert len(problemas) == 1 and "scikit-learn" in problemas[0]


def test_las_categorias_no_arrastran_arrow():
    """Las categorías deben ser listas de Python, no `pd.Index`.

    Guardarlas como `pd.Index` metía pyarrow (~150 MB) en la imagen de servicio,
    porque en pandas 3.0 un Index de strings vive sobre Arrow.
    """
    prep = ACategorias(["a"]).fit(pd.DataFrame({"a": ["x", "y", "x"]}))
    for col, valores in prep.categorias_.items():
        assert isinstance(valores, list), (
            f"'{col}' guarda {type(valores).__name__}; debe ser list o vuelve pyarrow")


# ------------------------------------------------------------ gate de validación
@pytest.mark.parametrize("segmento", list(SEGMENTOS))
def test_un_modelo_sensato_pasa_la_validacion(segmento):
    m = fabricar_modelo(segmento)
    ok = validar(m, ejemplo_de_entrada(m.contrato, m.features))
    assert any("escolaridad" in x for x in ok)


def test_un_modelo_al_reves_no_se_registra():
    """Si más escolaridad baja el ingreso, el modelo está roto: `validar` lo detiene."""
    m = fabricar_modelo("ocupados", efecto_escolaridad=-0.15)
    with pytest.raises(ModeloInvalido, match="escolaridad"):
        validar(m, ejemplo_de_entrada(m.contrato, m.features))


# ------------------------------------------------------------ código empaquetado
def test_detecta_codigo_distinto_al_que_entreno(tmp_path):
    """El `src/` que viaja con el modelo difiere del de la imagen → debe avisar."""
    import src
    origen = tmp_path / "src"
    shutil.copytree(src.__path__[0], origen, ignore=shutil.ignore_patterns("__pycache__"))

    # Mismo código con fines de línea de Windows: NO es una diferencia. (Se normaliza
    # primero: en un checkout de Windows el archivo ya puede venir con CRLF.)
    for f in ("model.py", "preprocessing.py"):
        crudo = (origen / f).read_bytes().replace(b"\r\n", b"\n")
        (origen / f).write_bytes(crudo.replace(b"\n", b"\r\n"))
    ruta = guardar(fabricar_modelo("ocupados"), tmp_path / "crlf", code_paths=[str(origen)])
    assert _codigo_distinto(ruta) == []

    # Un cambio real en una clase que está dentro del pickle: SÍ lo es.
    (origen / "model.py").write_text((origen / "model.py").read_text() + "\n# cambio\n")
    ruta = guardar(fabricar_modelo("ocupados"), tmp_path / "distinto", code_paths=[str(origen)])
    assert _codigo_distinto(ruta) == ["model.py"]
