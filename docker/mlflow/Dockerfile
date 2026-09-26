# Servidor de MLflow: tracking + model registry + proxy de artefactos.
#
# Por qué construirlo y no usar la imagen oficial: la versión queda clavada en el mismo
# archivo que el resto del proyecto (3.16.1, igual que requirements-dev.txt), corre
# como no-root, y en el Hito 6 se publica al mismo registro de imágenes (ECR) que la
# API. La oficial (ghcr.io/mlflow/mlflow) es una alternativa válida si prefieres no
# mantener esto.
#
# Local: SQLite + artefactos en disco, en un volumen de Docker.
# AWS:   las mismas banderas con --backend-store-uri postgresql://… (RDS) y
#        --artifacts-destination s3://… (S3). El código de entrenamiento y de servicio
#        no cambia: sólo habla HTTP con este servidor.

# Misma base, fijada por el mismo digest que el Dockerfile de la API (ver ahí el porqué).
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN pip install mlflow==3.16.1

RUN groupadd --system --gid 10002 mlflow \
 && useradd --system --uid 10002 --gid mlflow --no-create-home --shell /usr/sbin/nologin mlflow \
 && mkdir -p /mlflow \
 && chown mlflow:mlflow /mlflow
# Un volumen nombrado vacío hereda el dueño de /mlflow la primera vez que se monta: así
# el usuario no-root puede escribir la base y los artefactos.

USER mlflow
EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health',timeout=2).status==200 else 1)"

# --workers 1: SQLite admite un solo escritor. Con los 4 workers por omisión, dos
#   registros simultáneos pueden fallar con "database is locked". Con Postgres se sube.
# --allowed-hosts: MLflow 3 rechaza (403) cualquier Host que no sea localhost o una IP
#   privada, como defensa contra DNS rebinding. Dentro de la red de compose la API y el
#   entrenamiento llegan como "mlflow:5000", un nombre, así que hay que permitirlo.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "1", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--serve-artifacts", "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "mlflow:5000,localhost:*,127.0.0.1:*"]
