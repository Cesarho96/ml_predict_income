"""Métricas de Prometheus para la API (Hito 7).

Prometheus no recibe nada: cada 15 s llama a GET /metrics de CADA pod y guarda lo que
lee. Aquí sólo se definen los contadores y se actualizan; guardar la historia, calcular
tasas y alertar es trabajo de Prometheus.

Tres familias:
  · HTTP (RED: Rate, Errors, Duration): cuántas peticiones, con qué código, cuánto
    tardan. Por ruta PLANTILLA (`/contrato/{segmento}`), nunca por la ruta literal.
  · Modelo: qué versión sirve este pod (`income_model_info`, valor siempre 1: la
    información va en las etiquetas) y cuántas predicciones hizo cada versión.
  · Salida: distribución del ingreso predicho. Si la mediana servida se mueve sin que el
    modelo cambie, cambió QUIÉN pide predicciones: la primera señal de drift.

Regla de oro de las etiquetas: sólo valores de un conjunto chico y conocido. Nunca el
ingreso, un id de petición ni texto del usuario. Cada combinación distinta de etiquetas
es una serie de tiempo nueva en Prometheus; una etiqueta con valores libres lo tumba.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.routing import Match

# --------------------------------------------------------------------- HTTP (RED)
PETICIONES = Counter(
    "http_requests_total", "Peticiones HTTP atendidas", ["method", "route", "status"])
DURACION = Histogram(
    "http_request_duration_seconds", "Duración de las peticiones HTTP", ["method", "route"],
    # Cubos alrededor de lo medido en el Hito 6 (p50 ~90 ms, p95 ~250 ms bajo carga).
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))

# --------------------------------------------------------------------- modelo
APP = Gauge("app_info", "Versión de la API que corre en este pod (siempre 1)",
            ["version", "commit"])
MODELO = Gauge("income_model_info", "Modelo que sirve este pod (siempre 1)",
               ["segment", "model", "version", "alias", "run_id"])
PREDICCIONES = Counter("income_predictions_total", "Predicciones servidas",
                       ["segment", "model_version"])
INGRESO = Histogram(
    "income_predicted_mxn", "Ingreso mensual predicho (mediana), en pesos", ["segment"],
    # La mediana ponderada de la ENIGH 2024 es ~$7,900; la cola llega a cientos de miles.
    buckets=(1_000, 2_500, 5_000, 7_500, 10_000, 15_000, 20_000, 30_000, 50_000,
             100_000, 250_000))

SIN_RUTA = "sin_ruta"     # 404 de rutas inexistentes: una sola serie, no una por URL


def _ruta(request: Request) -> str:
    """La plantilla de la ruta que atiende la petición, no la URL literal.

    Un escáner que pide mil URLs inventadas no debe crear mil series: todas caen en
    `sin_ruta`. PARTIAL = la ruta existe pero no con ese método (405).
    """
    parcial = None
    for r in request.app.router.routes:
        coincide, _ = r.matches(request.scope)
        if coincide == Match.FULL:
            return getattr(r, "path", SIN_RUTA)
        if coincide == Match.PARTIAL and parcial is None:
            parcial = getattr(r, "path", None)
    return parcial or SIN_RUTA


def instrumentar(app: FastAPI) -> None:
    """Agrega el middleware que mide cada petición y el endpoint /metrics."""

    @app.middleware("http")
    async def medir(request: Request, call_next):
        ruta = _ruta(request)
        if ruta == "/metrics":                 # medir al medidor sólo agrega ruido
            return await call_next(request)
        t0 = time.perf_counter()
        codigo = 500                           # si la app lanza, cuenta como 500
        try:
            respuesta = await call_next(request)
            codigo = respuesta.status_code
            return respuesta
        finally:
            DURACION.labels(request.method, ruta).observe(time.perf_counter() - t0)
            PETICIONES.labels(request.method, ruta, str(codigo)).inc()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def registrar_arranque(modelos: dict, version_app: str, commit: str) -> None:
    """Qué código y qué modelos sirve este pod. Se llama una vez, al terminar de cargar."""
    APP.labels(version_app, commit).set(1)
    MODELO.clear()
    for segmento, m in modelos.items():
        MODELO.labels(segmento, m.nombre, m.version, m.alias or "", m.run_id or "").set(1)


def registrar_prediccion(segmento: str, version_modelo: str, mediana: float) -> None:
    PREDICCIONES.labels(segmento, version_modelo).inc()
    INGRESO.labels(segmento).observe(mediana)
