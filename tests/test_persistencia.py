"""¿Cuándo dispara el drift un reentrenamiento? (Hito 8)

Lo que se prueba es lo que convertiría el ciclo en un problema: reentrenar por un día
raro, reentrenar CADA noche mientras dure el drift, o dejar que una noche sin tráfico
borre o fabrique un episodio.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.monitoreo import persistencia
from src.monitoreo.persistencia import Medicion, decidir

T0 = datetime(2026, 11, 1, 5, 55, tzinfo=UTC)


def noches(*actuar: int | None) -> list[Medicion]:
    """Una medición por noche. Un número = features en 'actuar'; None = noche sin tráfico."""
    return [Medicion(f"n{i}", T0 + timedelta(days=i), a or 0, a is not None)
            for i, a in enumerate(actuar)]


def ultima(h: list[Medicion]) -> str:
    return h[-1].corrida


# ------------------------------------------------------------------ las tres reglas
def test_un_dia_raro_no_dispara():
    h = noches(0, 0, 4)
    assert not decidir(h, ultima(h), 3).disparar


def test_tres_noches_seguidas_disparan():
    h = noches(0, 0, 2, 4, 1)
    d = decidir(h, ultima(h), 3)
    assert d.disparar and d.racha == ["n2", "n3", "n4"]


def test_si_el_drift_sigue_no_se_reentrena_cada_noche():
    """Disparo por flanco: la cuarta noche con drift ya no dispara."""
    h = noches(0, 2, 4, 1, 3)
    assert decidir(h[:4], "n3", 3).disparar          # empieza el episodio
    assert not decidir(h, "n4", 3).disparar          # sigue: no otra vez


def test_si_el_drift_se_va_y_vuelve_es_otro_episodio():
    h = noches(2, 2, 2, 0, 3, 3, 3)
    assert decidir(h, "n2", 3).disparar
    assert decidir(h, "n6", 3).disparar


def test_una_noche_sin_trafico_no_corta_la_racha_ni_la_fabrica():
    h = noches(0, 2, None, 3, 4)                     # n2 sin tráfico: se ignora
    assert decidir(h, "n4", 3).disparar
    assert not decidir(h, "n2", 3).disparar          # la medición vacía no decide nada


def test_sin_historia_suficiente_no_dispara():
    h = noches(4, 4)
    d = decidir(h, ultima(h), 3)
    assert not d.disparar and "se necesitan 3" in d.motivo


def test_con_una_medicion_basta_si_asi_se_configura():
    """La demostración: mediciones_para_reentrenar = 1."""
    h = noches(0, 4)
    assert decidir(h, "n1", 1).disparar
    assert not decidir(noches(4, 4), "n1", 1).disparar   # ya venía de antes


def test_decide_respecto_a_su_corrida_no_a_lo_mas_nuevo():
    """Reintentar una corrida vieja da SU decisión, aunque después haya más mediciones."""
    h = noches(0, 0, 3, 3, 3, 0)
    assert decidir(h, "n4", 3).disparar
    assert decidir(h, "n4", 3) == decidir(h[:5], "n4", 3)


def test_una_corrida_sin_medicion_es_un_error():
    with pytest.raises(ValueError, match="no hay medición"):
        decidir(noches(1), "otra", 1)


# ------------------------------------------------------------------ contrato con el DAG
def test_el_dag_salta_con_el_mismo_codigo_que_usa_el_modulo():
    """Si estos dos números difieren, 'no disparar' se vuelve un FALLO del DAG."""
    dag = (Path(__file__).resolve().parent.parent / "airflow" / "dags" / "monitor.py")
    m = re.search(r"^SALTAR = (\d+)", dag.read_text(encoding="utf-8"), re.MULTILINE)
    assert m and int(m.group(1)) == persistencia.SALTAR


# ------------------------------------------------------------------ Postgres real
DSN_MONITOR = os.getenv("PREDICCIONES_DSN_MONITOR_PRUEBA")


@pytest.mark.postgres
@pytest.mark.skipif(not DSN_MONITOR, reason="sin Postgres de prueba (CI lo tiene)")
def test_lee_la_historia_de_la_tabla_drift():
    import psycopg
    from psycopg.types.json import Jsonb

    base = datetime.now(UTC) + timedelta(days=7300)              # fechas sólo nuestras
    sufijo = uuid.uuid4().hex[:6]
    filas = []
    for i, (actuar, n) in enumerate([(0, 500), (2, 500), (0, 0)]):
        for j, feature in enumerate(("cve_entidad", "anios_escolaridad", "edad")):
            nivel = "actuar" if j < actuar else ("insuficiente" if n == 0 else "estable")
            filas.append((f"p{i}-{sufijo}", base, base, base + timedelta(days=i), "ocupados",
                          feature, "categorica", 0.5, n, 300, nivel, Jsonb([])))
    with psycopg.connect(DSN_MONITOR, autocommit=True) as con:
        con.cursor().executemany(
            "INSERT INTO drift (corrida, calculado, ventana_inicio, ventana_fin, segmento, "
            "feature, tipo, psi, n_actual, n_minimo, nivel, detalle) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", filas)
        historia = {m.corrida: m for m in persistencia.leer_historia(con)}

    assert historia[f"p0-{sufijo}"].actuar == 0 and historia[f"p0-{sufijo}"].con_datos
    assert historia[f"p1-{sufijo}"].actuar == 2
    assert not historia[f"p2-{sufijo}"].con_datos               # sin peticiones
    propias = [m for c, m in historia.items() if c.endswith(sufijo)]   # otras pruebas
    assert decidir(propias, f"p1-{sufijo}", 1).disparar              # escriben aquí
