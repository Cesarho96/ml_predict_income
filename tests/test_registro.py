"""Registro de predicciones (Hito 7b). Lo esencial: /predict no depende de la base.

Casi todo usa un destino FALSO (sin Postgres) para probar la lógica de cola, lotes y
descartes. La última sección escribe en un Postgres real con los roles reales del
esquema; corre donde exista PREDICCIONES_DSN_PRUEBA (CI la define).
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import registro as reg
from app.registro import Registro, RegistroPredicciones
from tests.test_api import OCUPADO


def valor(nombre: str, **etiquetas) -> float:
    return sum(s.value for m in REGISTRY.collect() for s in m.samples
               if s.name == nombre and all(s.labels.get(k) == v for k, v in etiquetas.items()))


def un_registro(**cambios) -> Registro:
    base = dict(id=uuid.uuid4(), ts=datetime.now(UTC), segmento="ocupados",
                modelo="ingreso_ocupados", version="4", commit_api="abc1234", pod="api-x",
                features={"cve_entidad": "09", "anios_escolaridad": 16.0},
                inferior=10_000.0, mediana=20_000.0, superior=40_000.0, latencia_ms=3.2)
    return Registro(**{**base, **cambios})


class Falso:
    """Destino en memoria. `fallar` = cuántas llamadas a escribir() fallan primero."""
    def __init__(self, fallar: int = 0):
        self.lotes: list[list[Registro]] = []
        self.fallar = fallar
        self.llamadas = 0
        self.cerrado = False

    def escribir(self, lote):
        self.llamadas += 1
        if self.llamadas <= self.fallar:
            raise ConnectionError("base caída")
        self.lotes.append(list(lote))

    def cerrar(self):
        self.cerrado = True

    @property
    def filas(self):
        return [r for lote in self.lotes for r in lote]


# ------------------------------------------------------------------ cola y lotes
def test_escribe_en_lotes_y_vacia_al_apagar():
    destino = Falso()
    r = RegistroPredicciones(destino, espera_s=0.2)
    antes = valor("prediction_log_written_total")
    r.iniciar()
    for _ in range(7):
        r.registrar(un_registro())
    r.detener()
    assert len(destino.filas) == 7
    assert len(destino.lotes) < 7                  # en lotes, no fila por fila
    assert valor("prediction_log_written_total") == antes + 7
    assert destino.cerrado


def test_cola_llena_descarta_y_cuenta_sin_bloquear():
    r = RegistroPredicciones(Falso(), capacidad=2)  # sin iniciar: nadie vacía la cola
    antes = valor("prediction_log_dropped_total", reason="cola_llena")
    for _ in range(5):
        r.registrar(un_registro())                  # no bloquea ni lanza
    assert r.cola.qsize() == 2
    assert valor("prediction_log_dropped_total", reason="cola_llena") == antes + 3


def test_un_fallo_se_reintenta_una_vez():
    destino = Falso(fallar=1)
    r = RegistroPredicciones(destino, espera_s=0.1)
    r.iniciar()
    r.registrar(un_registro())
    r.detener()
    assert len(destino.filas) == 1 and destino.llamadas == 2


def test_dos_fallos_descartan_el_lote_y_lo_cuentan():
    destino = Falso(fallar=99)
    r = RegistroPredicciones(destino, espera_s=0.1)
    antes = valor("prediction_log_dropped_total", reason="error_base")
    r.iniciar()
    for _ in range(3):
        r.registrar(un_registro())
    r.detener()
    assert destino.filas == []
    assert valor("prediction_log_dropped_total", reason="error_base") == antes + 3


def test_apagar_con_la_base_atorada_no_cuelga_y_cuenta_lo_pendiente():
    suelta = threading.Event()

    class Atorado(Falso):
        def escribir(self, lote):
            suelta.wait(5)
            super().escribir(lote)

    r = RegistroPredicciones(Atorado(), lote_max=1, espera_s=0.05)
    antes = valor("prediction_log_dropped_total", reason="apagado")
    r.iniciar()
    for _ in range(4):
        r.registrar(un_registro())
    r.detener(timeout_s=0.3)                       # no espera a la base para siempre
    assert valor("prediction_log_dropped_total", reason="apagado") > antes
    suelta.set()


def test_los_motivos_de_descarte_existen_en_cero():
    """Sin esto, una alerta sobre rate(prediction_log_dropped_total) nunca vería nada."""
    for motivo in reg.MOTIVOS:
        assert any(s.labels.get("reason") == motivo for m in REGISTRY.collect()
                   for s in m.samples if s.name == "prediction_log_dropped_total")


def test_sin_dsn_queda_desactivado(monkeypatch):
    monkeypatch.delenv("PREDICCIONES_DSN", raising=False)
    assert reg.desde_entorno() is None
    assert valor("prediction_log_enabled") == 0


# ------------------------------------------------------------------ desde /predict
@pytest.fixture
def api_con_registro(monkeypatch):
    destino = Falso()
    monkeypatch.setattr("app.main.desde_entorno",
                        lambda: RegistroPredicciones(destino, espera_s=0.05))
    from app.main import app
    with TestClient(app) as c:
        yield c, destino


def test_predict_registra_lo_que_vio_el_modelo(api_con_registro):
    client, destino = api_con_registro
    resp = client.post("/predict", json=OCUPADO)
    assert resp.status_code == 200
    id_ = resp.headers["X-Prediccion-Id"]
    client.app.state.registro.detener()
    (fila,) = destino.filas
    assert str(fila.id) == id_
    assert fila.segmento == "ocupados" and fila.mediana == resp.json()["mediana"]
    assert fila.features["cve_entidad"] == "09"
    assert "edad" not in fila.features             # sólo las features de SU modelo


def test_predict_responde_aunque_la_base_este_caida(monkeypatch):
    destino = Falso(fallar=10**6)
    monkeypatch.setattr("app.main.desde_entorno",
                        lambda: RegistroPredicciones(destino, espera_s=0.05))
    from app.main import app
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/predict", json=OCUPADO).status_code == 200


def test_un_422_no_se_registra(api_con_registro):
    client, destino = api_con_registro
    client.post("/predict", json={**OCUPADO, "formalidad": "no existe"})
    client.app.state.registro.detener()
    assert destino.filas == []


# ------------------------------------------------------------------ Postgres real
DSN = os.getenv("PREDICCIONES_DSN_PRUEBA")        # como api_escritura
DSN_LECTOR = os.getenv("PREDICCIONES_DSN_LECTOR_PRUEBA")
con_postgres = pytest.mark.skipif(not (DSN and DSN_LECTOR),
                                  reason="sin PREDICCIONES_DSN_PRUEBA (CI la define)")


@pytest.mark.postgres
@con_postgres
def test_escribe_en_postgres_con_el_esquema_real():
    import psycopg

    r = RegistroPredicciones(reg.DestinoPostgres(DSN), espera_s=0.1)
    ids = [uuid.uuid4() for _ in range(3)]
    r.iniciar()
    for i in ids:
        r.registrar(un_registro(id=i))
    r.detener()
    with psycopg.connect(DSN_LECTOR) as con:
        filas = con.execute(
            "SELECT id, features->>'cve_entidad', version FROM predicciones WHERE id = ANY(%s)",
            [ids]).fetchall()
    assert sorted(f[0] for f in filas) == sorted(ids)
    assert {(f[1], f[2]) for f in filas} == {("09", "4")}


@pytest.mark.postgres
@con_postgres
def test_la_api_no_puede_leer_ni_borrar():
    """Mínimo privilegio: con la contraseña de la API sólo se puede INSERTAR."""
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as con:
        for sql in ("SELECT count(*) FROM predicciones", "DELETE FROM predicciones",
                    "DROP TABLE predicciones"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                con.execute(sql)
