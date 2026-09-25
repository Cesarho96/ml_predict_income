"""API de predicción de ingreso — capa HTTP.

Diseño: esta capa sólo traduce. Valida la entrada contra el contrato que el propio
modelo trae, llama a `Predictor` y formatea la salida. Cero lógica de negocio aquí.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from src.predictor import Predictor, segmento_de

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("api")

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar en el arranque, no en la primera petición: así el primer usuario no paga
    # el costo y `/ready` puede decir la verdad sobre si el pod sirve o no. Si el
    # registry no responde o no hay champion, esto lanza y el proceso termina: el
    # orquestador lo ve y lo reinicia. Ver la decisión 3 en src/predictor.py.
    log.info("resolviendo y cargando modelos…")
    predictor.cargar()
    log.info("modelos listos: %s",
             ", ".join(m.etiqueta for m in predictor.modelos.values()))
    yield
    log.info("apagando")


app = FastAPI(
    title="Predicción de ingreso mensual — México (ENIGH 2024)",
    description=(
        "Devuelve la **mediana condicional** del ingreso mensual de una persona adulta "
        "y un intervalo del 80%. No predice el ingreso exacto de nadie: el error "
        "mediano es ~31% y el intervalo es el entregable."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------- esquemas
class PeticionPrediccion(BaseModel):
    """Los campos se validan contra el contrato que viaja dentro del modelo registrado.

    Una sola fuente de verdad: si el modelo se reentrena con otras categorías, el
    contrato cambia con él y la API deja de aceptar lo que el modelo ya no conoce.
    Nada que mantener en dos lados.
    """

    trabajo_mes_pasado: bool = Field(
        ..., description="¿Trabajaste el mes pasado? Decide qué modelo responde.")

    # --- comunes
    anios_escolaridad: float | None = Field(None, ge=0, le=24)
    es_jefe_hogar: int | None = Field(None, ge=0, le=1)
    cve_entidad: str | None = Field(None, description="Clave INEGI de entidad, '01'–'32'")

    # --- sólo ocupados
    horas_trabajadas: float | None = Field(None, ge=1, le=168)
    formalidad: str | None = None
    sinco_grupo: str | None = None
    posicion_ocupacion: str | None = None
    sector_agrupado: str | None = None
    tam_empresa_grupo: str | None = None
    es_mujer: int | None = Field(None, ge=0, le=1)

    # --- sólo no ocupados
    edad: float | None = Field(None, ge=18, le=110)
    integrantes_hogar: float | None = Field(None, ge=1, le=30)
    situacion_conyugal: str | None = None
    actividad_no_ocupado: str | None = None

    @model_validator(mode="after")
    def validar_contra_contrato(self):
        seg = segmento_de(self.trabajo_mes_pasado)
        if not predictor.listo:
            return self  # el arranque aún no termina; /ready lo reporta
        b = predictor.modelos[seg]

        faltan = [f for f in b.features if getattr(self, f, None) is None]
        if faltan:
            raise ValueError(
                f"para segmento '{seg}' faltan campos obligatorios: {sorted(faltan)}")

        for campo, permitidos in b.categoricas.items():
            v = getattr(self, campo, None)
            if v is not None and v not in permitidos:
                raise ValueError(
                    f"'{campo}'='{v}' no es un valor conocido. Válidos: {permitidos}")
        return self


class RespuestaPrediccion(BaseModel):
    segmento: str
    inferior: float
    mediana: float
    superior: float
    moneda: str
    periodo: str
    cobertura: float
    modelo: str
    version_modelo: str
    mensaje: str


# --------------------------------------------------------------------- endpoints
@app.get("/health", tags=["operación"])
def health():
    """Liveness: ¿el proceso está vivo? No dice nada sobre los modelos.

    Kubernetes reinicia el contenedor si esto falla, así que debe ser barato y no
    depender de nada externo.
    """
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ready", tags=["operación"])
def ready():
    """Readiness: ¿este pod puede atender tráfico?

    Separarlo de /health es lo que evita que el balanceador mande peticiones a un pod
    que todavía está cargando los modelos.
    """
    if not predictor.listo:
        raise HTTPException(status_code=503, detail="modelos no cargados")
    # Qué versión EXACTA sirve este pod. Con varias réplicas y un alias que se mueve,
    # es la única forma de saber quién respondió qué.
    return {
        "status": "ready",
        "modelos": {s: {"nombre": m.nombre, "version": m.version, "alias": m.alias,
                        "uri": m.uri, "run_id": m.run_id, "familia": m.familia,
                        "features": len(m.features)}
                    for s, m in predictor.modelos.items()},
    }


@app.get("/contrato/{segmento}", tags=["contrato"])
def contrato(segmento: str):
    """El contrato de entrada, tal como lo trae el modelo. Útil para construir el formulario."""
    if segmento not in predictor.modelos:
        raise HTTPException(404, f"segmento desconocido: {segmento}")
    m = predictor.modelos[segmento]
    return {"segmento": m.segmento, "modelo": m.etiqueta, "features": m.features,
            "contrato": m.contrato}


@app.post("/predict", response_model=RespuestaPrediccion, tags=["predicción"])
def predict(p: PeticionPrediccion):
    if not predictor.listo:
        raise HTTPException(503, "modelos no cargados")
    try:
        r = predictor.predecir(p.model_dump(), p.trabajo_mes_pasado)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return RespuestaPrediccion(
        segmento=r.segmento, inferior=r.inferior, mediana=r.mediana, superior=r.superior,
        moneda=r.moneda, periodo=r.periodo, cobertura=r.cobertura,
        modelo=r.modelo, version_modelo=r.version,
        mensaje=(f"Entre ${r.inferior:,.0f} y ${r.superior:,.0f} al mes, "
                 f"típicamente ${r.mediana:,.0f}."),
    )


@app.exception_handler(Exception)
async def error_no_manejado(request: Request, exc: Exception):
    # Nunca devolver un traceback al cliente: se registra con detalle y se responde genérico.
    log.exception("error no manejado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "error interno"})
