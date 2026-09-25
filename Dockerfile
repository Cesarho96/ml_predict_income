# Imágenes del proyecto: una para SERVIR (la predeterminada) y una para ENTRENAR.
#
#   docker build -t ml-predict-income .            → etapa `runtime` (la última)
#   docker build --target train -t ...-train .     → etapa `train`
#   (o simplemente `python tasks.py up` / `python tasks.py train`, vía docker compose)
#
# Sin directiva `# syntax=`: no hace falta para nada de lo que usa este archivo y evita
# una descarga extra del frontend de BuildKit en redes restringidas.
#
# Decisiones y por qué:
#
# 1. `python:3.11-slim` y no `alpine`. Alpine usa musl en vez de glibc, así que
#    numpy/scipy/scikit-learn no tienen wheels: pip los compila desde fuente. El build
#    pasa de 40 segundos a 20 minutos y la imagen termina MÁS grande, no más chica.
#
# 2. Multi-stage. Las etapas `builder*` tienen caché de pip; las finales sólo reciben
#    el virtualenv ya construido. Menos peso y menos superficie de ataque.
#
# 3. Usuario no-root con UID fijo (10001). `runAsNonRoot` en Kubernetes valida contra
#    un UID numérico.
#
# 4. Dependencias antes que código: un cambio en `app/main.py` no reinstala sklearn.
#
# 5. (Hito 3) El modelo YA NO se hornea. La imagen es sólo código y el modelo se
#    descarga del registry al arrancar (`src/predictor.py`). El tag de la imagen
#    identifica el código; la versión del modelo la reporta `/ready`. Reentrenar ya no
#    exige reconstruir, a cambio de que arrancar exija que el registry responda.
#
# 6. (Hito 3) Entrenar y servir comparten base. El mismo commit entrenado en Windows y
#    en Linux dio modelos distintos (ver src/train.py). Entrenar en `train`, que es la
#    misma Debian + Python + wheels que `runtime`, elimina esa variable: lo que se
#    entrena es exactamente lo que después se deserializa.

ARG PYTHON_VERSION=3.11

# ----------------------------------------------------------------- builder: servicio
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ------------------------------------------------ builder: entrenamiento (= servicio + extras)
# Parte del venv de servicio y sólo AGREGA. Así las versiones de sklearn/numpy/pandas
# son, por construcción, las mismas con las que se va a servir.
FROM builder AS builder-train
COPY requirements-train.txt ./
RUN pip install -r requirements-train.txt

# ----------------------------------------------------------------- base común
FROM python:${PYTHON_VERSION}-slim AS base

# HOME=/tmp: el usuario no tiene home, y con el sistema de archivos en sólo-lectura
# cualquier librería que quiera escribir en ~ (cachés, configuración) tiene que caer en
# el único lugar escribible, el tmpfs de /tmp.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/tmp \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# ----------------------------------------------------------------- train
# Los datos NO van en la imagen: se montan en /app/data al correr (docker-compose.yml).
# Una imagen de entrenamiento con los datos dentro se vuelve un artefacto de datos
# versionado por accidente, y pesa lo que pesen los microdatos.
FROM base AS train
COPY --from=builder-train /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/
USER app
CMD ["python", "-m", "src.train"]

# ----------------------------------------------------------------- runtime (predeterminada)
FROM base AS runtime

ENV PORT=8000 \
    MODEL_ALIAS=champion \
    MODEL_CACHE_DIR=/tmp/modelos

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app app/ ./app/

USER app
EXPOSE 8000

# Healthcheck con el python de la imagen: agregar curl sólo para esto son ~10 MB.
# start-period más largo que en el Hito 1: ahora el arranque incluye bajar el modelo.
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

# Un worker por contenedor: en Kubernetes se escala con réplicas, no con procesos.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
