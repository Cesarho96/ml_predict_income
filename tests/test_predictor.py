"""Pruebas de la capa de predicción, sin HTTP de por medio."""
import math

from src.predictor import Predictor, segmento_de


def test_router():
    assert segmento_de(True) == "ocupados"
    assert segmento_de(False) == "no_ocupados"


def test_bundles_cargan_en_proceso_limpio():
    """Si esto falla, el pickle sigue apuntando a `__main__` (ver scripts/migrate_bundles.py)."""
    p = Predictor().cargar()
    assert p.listo
    assert len(p.bundles["ocupados"].features) == 10
    assert len(p.bundles["no_ocupados"].features) == 7


def test_mas_escolaridad_no_baja_la_prediccion():
    """Prueba de sensatez del dominio: el retorno a la escolaridad es positivo."""
    p = Predictor().cargar()
    base = dict(trabajo_mes_pasado=True, horas_trabajadas=48,
                formalidad="3 Formal (con seguridad social)", cve_entidad="09",
                sinco_grupo="2 Profesionistas y técnicos",
                posicion_ocupacion="Subordinado remunerado", es_jefe_hogar=1,
                sector_agrupado="6 Servicios profesionales y financieros",
                es_mujer=0, tam_empresa_grupo="3 Mediana (51-250)")
    bajo = p.predecir({**base, "anios_escolaridad": 6}, True).mediana
    alto = p.predecir({**base, "anios_escolaridad": 17}, True).mediana
    assert alto > bajo, f"escolaridad 17 ({alto}) debería superar a 6 ({bajo})"


def test_intervalo_es_finito_y_ordenado():
    p = Predictor().cargar()
    r = p.predecir(dict(trabajo_mes_pasado=False,
                        actividad_no_ocupado="Pensionado(a)/jubilado(a)",
                        es_jefe_hogar=1, anios_escolaridad=12, edad=70,
                        cve_entidad="19", integrantes_hogar=2,
                        situacion_conyugal="Casado(a)"), False)
    assert all(math.isfinite(v) for v in (r.inferior, r.mediana, r.superior))
    assert r.inferior <= r.mediana <= r.superior


def test_deteccion_de_desajuste_de_versiones():
    """El bundle declara con qué versiones se serializó; un desajuste debe ser visible.

    Esta prueba existe por un fallo real: sklearn 1.9.1 no puede abrir un pickle de
    1.8.0 (`ModuleNotFoundError: No module named '_loss'`). Es un fallo de arranque,
    no de petición, y debe decir por qué.
    """
    from src.predictor import comprobar_versiones

    assert comprobar_versiones({"scikit-learn": "1.8.0", "numpy": "2.4.4",
                                "pandas": "3.0.2"}) == []
    problemas = comprobar_versiones({"scikit-learn": "0.1.0"})
    assert len(problemas) == 1 and "scikit-learn" in problemas[0]


def test_las_categorias_no_arrastran_arrow():
    """Las categorías deben ser listas de Python, no `pd.Index`.

    Historia: guardarlas como `pd.Index` metía pyarrow (~150 MB) en la imagen de
    servicio, porque en pandas 3.0 un Index de strings vive sobre Arrow. Una lista hace
    exactamente lo mismo. Esta prueba impide que la optimización se revierta sin querer.
    """
    p = Predictor().cargar()
    prep = p.bundles["ocupados"].cuantiles["0.5"].named_steps["prep"]
    for col, valores in prep.categorias_.items():
        assert isinstance(valores, list), (
            f"'{col}' guarda {type(valores).__name__}; debe ser list o vuelve pyarrow")
