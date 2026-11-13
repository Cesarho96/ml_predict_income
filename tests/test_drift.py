"""Drift con PSI (Hito 7c). Datos sintéticos: CI no tiene la ENIGH.

Lo que se prueba es lo que haría que el monitoreo MIENTA: un PSI mal calculado, una
referencia sin ponderar que "ve" drift donde sólo hay diseño muestral, una muestra chica
que grita "actuar" por azar, o una categoría nueva que pasa sin hacer ruido.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.monitoreo import drift
from src.monitoreo.drift import Feature, aprender_feature, comparar, psi

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ la fórmula
def test_psi_de_dos_distribuciones_iguales_es_cero():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0)


def test_psi_coincide_con_la_formula_a_mano():
    ref, act = np.array([0.5, 0.5]), np.array([0.8, 0.2])
    a_mano = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(ref, act) == pytest.approx(a_mano)            # ≈ 0.416 → actuar


def test_un_bin_vacio_no_da_infinito():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))


def test_niveles_y_muestra_minima():
    assert drift.n_minimo(2) == 300 and drift.n_minimo(32) == 620
    assert drift.nivel(0.05, 1000, 300, 10) == "estable"
    assert drift.nivel(0.15, 1000, 300, 10) == "vigilar"
    assert drift.nivel(0.40, 1000, 300, 10) == "actuar"
    assert drift.nivel(0.40, 100, 620, 32) == "insuficiente"  # chico y tibio: no opina
    # Chico pero contundente: 335 peticiones con PSI 7 no son azar (ruido ≈ 0.09).
    assert drift.nivel(7.0, 335, 620, 32) == "actuar"
    assert drift.nivel(float("nan"), 0, 300, 10) == "insuficiente"


def test_con_pocas_peticiones_el_psi_es_ruido():
    """Por qué existe n_minimo: sin drift real, 32 categorías y 100 peticiones dan un
    PSI alto por puro azar. Con suficientes peticiones, baja."""
    cats = [f"{i:02d}" for i in range(1, 33)]
    ref = aprender_feature("cve_entidad", pd.Series(cats * 100), pd.Series([1.0] * 3200))
    poco = comparar(ref, pd.Series(RNG.choice(cats, 100)), "s")
    mucho = comparar(ref, pd.Series(RNG.choice(cats, 20_000)), "s")
    assert poco.psi > 0.15 and poco.nivel == "insuficiente"
    assert mucho.psi < 0.01 and mucho.nivel == "estable"


# ------------------------------------------------------------------ la referencia
def test_la_referencia_se_pondera_por_el_factor_de_expansion():
    """La encuesta sobre-muestrea al estado A (mitad de la muestra) pero cada uno de esos
    hogares representa a pocas personas: en la POBLACIÓN, A es el 10 %. Tráfico que se
    parece a la población no debe marcar drift; contra la muestra cruda, sí lo marcaría."""
    estados = pd.Series(["A"] * 500 + ["B"] * 500)
    factor = pd.Series([1.0] * 500 + [9.0] * 500)            # A: 500 personas; B: 4500
    poblacion = pd.Series(RNG.choice(["A", "B"], 5000, p=[0.1, 0.9]))

    ponderada = aprender_feature("estado", estados, factor)
    cruda = aprender_feature("estado", estados, pd.Series([1.0] * 1000))
    assert comparar(ponderada, poblacion, "s").nivel == "estable"
    assert comparar(cruda, poblacion, "s").nivel == "actuar"


def test_numericas_se_parten_en_deciles_ponderados():
    x = pd.Series(RNG.normal(40, 10, 50_000))
    f = aprender_feature("horas", x, pd.Series(np.ones(len(x))))
    assert f.tipo == "numerica" and len(f.bins) == 10
    assert np.allclose(f.p_ref, 0.1, atol=0.01)               # ~10 % por bin


def test_binarias_son_categoricas_y_hablan_igual():
    """La referencia guarda 0/1 como int8; la API los manda como 1, 1.0 o True."""
    f = aprender_feature("es_mujer", pd.Series([0, 1] * 50, dtype="int8"),
                         pd.Series([1.0] * 100))
    assert f.tipo == "categorica" and sorted(f.bins) == ["0", "1"]
    r = comparar(f, pd.Series([1, 1.0, True, 0, 0.0, False] * 100), "s")
    assert r.psi == pytest.approx(0, abs=1e-9)


# ------------------------------------------------------------------ la comparación
def test_una_categoria_nueva_no_pasa_en_silencio():
    f = aprender_feature("formalidad", pd.Series(["formal", "informal"] * 500),
                         pd.Series([1.0] * 1000))
    r = comparar(f, pd.Series(["formal"] * 400 + ["NUEVA"] * 400), "s")
    nueva = next(d for d in r.detalle if d["bin"] == "NUEVA")
    assert nueva["p_ref"] == 0 and nueva["p_actual"] == 0.5
    assert r.nivel == "actuar"


def test_faltantes_tienen_su_propio_bin():
    f = aprender_feature("horas", pd.Series(RNG.normal(40, 10, 5000)),
                         pd.Series(np.ones(5000)))
    r = comparar(f, pd.Series([None] * 500), "s")
    assert any(d["bin"] == drift.FALTANTE and d["p_actual"] == 1.0 for d in r.detalle)


def test_calcular_cubre_cada_feature_de_cada_segmento_aunque_no_haya_trafico():
    ref = {"ocupados": [Feature("x", "categorica", ["a", "b"], np.array([0.5, 0.5]))],
           "no_ocupados": [Feature("y", "categorica", ["c"], np.array([1.0]))]}
    actual = pd.DataFrame({"segmento": ["ocupados"] * 400, "x": ["a", "b"] * 200})
    res = {(r.segmento, r.feature): r for r in drift.calcular(ref, actual)}
    assert res[("ocupados", "x")].nivel == "estable"
    sin = res[("no_ocupados", "y")]
    assert sin.n_actual == 0 and np.isnan(sin.psi) and sin.nivel == "insuficiente"


# ------------------------------------------------------------------ Postgres real
DSN_API = os.getenv("PREDICCIONES_DSN_PRUEBA")
DSN_MONITOR = os.getenv("PREDICCIONES_DSN_MONITOR_PRUEBA")
DSN_LECTOR = os.getenv("PREDICCIONES_DSN_LECTOR_PRUEBA")
con_postgres = pytest.mark.skipif(not (DSN_API and DSN_MONITOR and DSN_LECTOR),
                                  reason="sin Postgres de prueba (CI lo tiene)")


@pytest.mark.postgres
@con_postgres
def test_lee_predicciones_y_guarda_resultados_sin_duplicar():
    import psycopg
    from psycopg.types.json import Jsonb

    marca = datetime.now(UTC) + timedelta(days=3650)       # una ventana sólo nuestra
    with psycopg.connect(DSN_API, autocommit=True) as con:
        for estado in ("09", "09", "15"):
            con.execute(
                "INSERT INTO predicciones VALUES (%s, %s, 'ocupados', 'm', '4', 'c', 'p', "
                "%s, 1, 2, 3, 1)", (uuid.uuid4(), marca, Jsonb({"cve_entidad": estado})))

    ref = {"ocupados": [aprender_feature("cve_entidad", pd.Series(["09", "15"]),
                                         pd.Series([1.0, 1.0]))]}
    corrida = f"prueba-{uuid.uuid4().hex[:6]}"
    with psycopg.connect(DSN_MONITOR, autocommit=True) as con:
        actual = drift.leer_predicciones(con, marca - timedelta(seconds=1),
                                         marca + timedelta(seconds=1), 1000)
        assert sorted(actual["cve_entidad"]) == ["09", "09", "15"]
        resultados = drift.calcular(ref, actual)
        drift.guardar(con, corrida, marca, marca, resultados)
        drift.guardar(con, corrida, marca, marca, resultados)    # un reintento de Airflow
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            con.execute("DELETE FROM predicciones")              # monitor no borra

    with psycopg.connect(DSN_LECTOR) as con:
        (n,) = con.execute("SELECT count(*) FROM drift WHERE corrida = %s",
                           [corrida]).fetchone()
    assert n == 1                                                # idempotente
