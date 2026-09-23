# Imagen de servicio para la API de predicción de ingreso.
#
# Sin directiva `# syntax=`: no hace falta para nada de lo que usa este archivo y evita
# una descarga extra del frontend de BuildKit en redes restringidas. Si más adelante
# quieres builds incrementales más rápidos, agrégala y usa una caché de pip:
#     # syntax=docker/dockerfile:1.7
#     RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
#
# Cinco decisiones y por qué:
#
# 1. `python:3.11-slim` y no `alpine`. Alpine usa musl en vez de glibc, así que
#    numpy/scipy/scikit-learn no tienen wheels: pip los compila desde fuente. El build
#    pasa de 40 segundos a 20 minutos y la imagen termina MÁS grande, no más chica.
#    Slim es la base correcta para el stack científico de Python.
#
# 2. Multi-stage. La etapa `builder` tiene compiladores y caché de pip; la final sólo
#    recibe el virtualenv ya construido. Nada de gcc, headers ni caché en la imagen que
#    se publica: menos peso y menos superficie de ataque.
#
# 3. Usuario no-root con UID fijo. `USER app` (10001) para que un escape del proceso no
#    sea root en el nodo. El UID explícito importa en Kubernetes: `runAsNonRoot` valida
#    contra un UID numérico, y un usuario sin UID fijo puede fallar el admission.
#
# 4. Las dependencias se copian e instalan ANTES que el código. Las capas de Docker se
#    cachean en orden: así un cambio en `app/main.py` no reinstala scikit-learn.
#
# 5. El modelo se hornea en la imagen. La imagen queda inmutable y auto-contenida —
#    el tag identifica código + modelo, y un rollback es un rollback de verdad. El costo
#    es que reentrenar exige reconstruir. En el Hito 3 esto cambia a descargar el modelo
#    del registry de MLflow al arrancar; la discusión de ese trade-off está en el roadmap.

ARG PYTHON_VERSION=3.11

# ----------------------------------------------------------------- etapa 1: builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------- etapa 2: runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}" \
    MODEL_DIR=/app/models \
    PORT=8000

# Usuario sin privilegios, sin home y sin shell de login.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/    ./src/
COPY --chown=app:app app/    ./app/
COPY --chown=app:app models/ ./models/

USER app

EXPOSE 8000

# Healthcheck con el python que ya está en la imagen: agregar curl sólo para esto son
# ~10 MB y un binario más que mantener parcheado.
HEALTHCHECK --interval=30s --timeout=3s --start-period=25s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

# Un worker por contenedor: en Kubernetes se escala con réplicas, no con procesos
# dentro del pod. Así el HPA ve una señal de CPU limpia y cada pod tiene una sola copia
# del modelo en memoria.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
