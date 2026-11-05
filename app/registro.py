"""Registro de predicciones (Hito 7b): cada /predict queda en Postgres, sin frenar ni
tumbar /predict.

    /predict ──encola (µs)──▶ cola en memoria, acotada ──hilo escritor──▶ Postgres, en lotes

Por qué así:
  · /predict NUNCA espera a la base: encola y responde. Si la base tarda o se cae, el
    usuario no se entera.
  · La cola es ACOTADA. Con la base caída se llena, y lo que ya no cabe se DESCARTA y se
    cuenta (`prediction_log_dropped_total`). Una cola sin límite convertiría una caída de
    la base en un OOMKilled de la API: registrar importa, servir importa más.
  · Escribe en LOTES (hasta 500 filas, o lo juntado en 1 s): una transacción por lote,
    no una por predicción.
  · Un lote que falla se reintenta UNA vez con conexión nueva; si vuelve a fallar se
    descarta y se cuenta. Sin reintentos infinitos: un lote atorado frenaría a todos los
    que vienen detrás.
  · Al apagar (rollout, escalado hacia abajo) vacía la cola durante hasta 5 s.

Configuración: PREDICCIONES_DSN. Sin ella el registro queda desactivado (pruebas,
`tasks.py api`) y `prediction_log_enabled` = 0. En Kubernetes viene de un Secret
OBLIGATORIO: un error de configuración debe fallar al desplegar; una caída de la base,
degradar sin tumbar el servicio.

Qué NO se guarda: IP, user-agent, ni nada fuera del contrato del modelo. Lo mínimo para
medir drift, que es para lo único que existe esta tabla.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from prometheus_client import Counter, Gauge, Histogram

log = logging.getLogger("registro")

# --------------------------------------------------------------------- métricas
ESCRITAS = Counter("prediction_log_written_total", "Predicciones guardadas en la base")
DESCARTADAS = Counter("prediction_log_dropped_total", "Predicciones que NO se guardaron",
                      ["reason"])
EN_COLA = Gauge("prediction_log_queue_size", "Predicciones esperando a escribirse")
ACTIVO = Gauge("prediction_log_enabled", "1 si este pod registra predicciones")
LOTE = Histogram("prediction_log_batch_seconds", "Tiempo de escribir un lote",
                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))

# Las series deben EXISTIR en 0 desde el arranque. Una alerta con rate() sobre una serie
# que nunca apareció no ve "0", ve "nada", y nunca dispara.
MOTIVOS = ("cola_llena", "error_base", "apagado")
for _m in MOTIVOS:
    DESCARTADAS.labels(_m)


@dataclass(frozen=True)
class Registro:
    """Una predicción servida: lo que entró, lo que salió y quién la produjo."""
    id: uuid.UUID
    ts: datetime
    segmento: str
    modelo: str
    version: str
    commit_api: str
    pod: str
    features: dict
    inferior: float
    mediana: float
    superior: float
    latencia_ms: float


class Destino(Protocol):
    """A dónde se escriben los lotes. Postgres en producción; uno falso en las pruebas."""
    def escribir(self, lote: list[Registro]) -> None: ...
    def cerrar(self) -> None: ...


class DestinoPostgres:
    SQL = ("INSERT INTO predicciones (id, ts, segmento, modelo, version, commit_api, pod, "
           "features, inferior, mediana, superior, latencia_ms) "
           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._con = None

    def _conexion(self):
        if self._con is None or self._con.closed:
            import psycopg  # tardío: sin DSN, psycopg ni se carga

            self._con = psycopg.connect(
                self.dsn, autocommit=True, connect_timeout=3,
                application_name="api-ingreso",
                # Un INSERT colgado no debe congelar al escritor: 5 s y se aborta.
                options="-c statement_timeout=5000")
        return self._con

    def escribir(self, lote: list[Registro]) -> None:
        from psycopg.types.json import Jsonb

        con = self._conexion()
        try:
            with con.transaction(), con.cursor() as cur:
                cur.executemany(self.SQL, [
                    (r.id, r.ts, r.segmento, r.modelo, r.version, r.commit_api, r.pod,
                     Jsonb(r.features), r.inferior, r.mediana, r.superior, r.latencia_ms)
                    for r in lote])
        except Exception:
            self.cerrar()  # la conexión puede haber quedado rota: el reintento abre otra
            raise

    def cerrar(self) -> None:
        if self._con is not None:
            # Cerrar una conexión que ya está rota no debe lanzar.
            with contextlib.suppress(Exception):
                self._con.close()
            self._con = None


class RegistroPredicciones:
    def __init__(self, destino: Destino, capacidad: int = 10_000, lote_max: int = 500,
                 espera_s: float = 1.0):
        self.destino = destino
        self.cola: queue.Queue[Registro] = queue.Queue(maxsize=capacidad)
        self.lote_max = lote_max
        self.espera_s = espera_s
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="registro-predicciones",
                                      daemon=True)
        self._ultimo_aviso = 0.0

    # ---------------------------------------------------------------- lado de /predict
    def registrar(self, r: Registro) -> None:
        """Lo llama /predict. Nunca bloquea ni lanza."""
        try:
            self.cola.put_nowait(r)
        except queue.Full:
            DESCARTADAS.labels("cola_llena").inc()
        EN_COLA.set(self.cola.qsize())

    # ---------------------------------------------------------------- ciclo de vida
    def iniciar(self) -> None:
        ACTIVO.set(1)
        self._hilo.start()
        log.info("registro de predicciones activo")

    def detener(self, timeout_s: float = 5.0) -> None:
        """Vacía la cola hasta `timeout_s`; lo que quede se cuenta como perdido."""
        self._parar.set()
        self._hilo.join(timeout_s)
        if pendientes := self.cola.qsize():
            DESCARTADAS.labels("apagado").inc(pendientes)
            log.warning("apagando con %d predicciones sin guardar", pendientes)
        self.destino.cerrar()
        ACTIVO.set(0)

    # ---------------------------------------------------------------- hilo escritor
    def _bucle(self) -> None:
        while not (self._parar.is_set() and self.cola.empty()):
            if lote := self._tomar_lote():
                self._escribir(lote)
            EN_COLA.set(self.cola.qsize())

    def _tomar_lote(self) -> list[Registro]:
        """Espera la primera predicción; luego junta las que lleguen hasta llenar el lote
        o cumplir `espera_s`. Con poco tráfico escribe cada segundo; con mucho, cada 500."""
        try:
            lote = [self.cola.get(timeout=self.espera_s)]
        except queue.Empty:
            return []
        limite = time.monotonic() + self.espera_s
        while len(lote) < self.lote_max:
            restante = limite - time.monotonic()
            if restante <= 0 or self._parar.is_set() and self.cola.empty():
                break
            try:
                lote.append(self.cola.get(timeout=restante))
            except queue.Empty:
                break
        return lote

    def _escribir(self, lote: list[Registro]) -> None:
        for intento in (1, 2):
            t0 = time.perf_counter()
            try:
                self.destino.escribir(lote)
            except Exception as e:  # noqa: BLE001 — cualquier falla de la base se cuenta
                if intento == 1:
                    continue
                DESCARTADAS.labels("error_base").inc(len(lote))
                self._avisar(e, len(lote))
                return
            LOTE.observe(time.perf_counter() - t0)
            ESCRITAS.inc(len(lote))
            return

    def _avisar(self, e: Exception, n: int) -> None:
        """Con la base caída fallaría un lote por segundo: un aviso cada 30 s basta."""
        ahora = time.monotonic()
        if ahora - self._ultimo_aviso >= 30:
            self._ultimo_aviso = ahora
            log.warning("no se pudo guardar un lote de %d predicciones: %s: %s",
                        n, type(e).__name__, str(e).splitlines()[0] if str(e) else "")


def desde_entorno() -> RegistroPredicciones | None:
    """El registro que pide la configuración, o None si no hay PREDICCIONES_DSN."""
    dsn = os.getenv("PREDICCIONES_DSN", "").strip()
    if not dsn:
        ACTIVO.set(0)
        log.info("registro de predicciones desactivado (sin PREDICCIONES_DSN)")
        return None
    return RegistroPredicciones(DestinoPostgres(dsn))
