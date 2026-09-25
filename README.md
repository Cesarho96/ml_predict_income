# ml_predict_income

Predicción del ingreso monetario mensual de una persona adulta en México, a partir de
la ENIGH 2024 del INEGI. Devuelve la **mediana condicional** y un intervalo del 80%.

| | Ocupados | No ocupados |
|---|---|---|
| preguntas | 10 | 7 |
| MdAPE (test) | 30.77% | 41.12% |
| cobertura del intervalo | 80.9% | 81.6% |

Detalle de decisiones y métricas: `models/MODEL_CARD.md`.

## Arranque rápido

Requiere **Python 3.11** y **Docker**. Las versiones están clavadas a las que
serializan los modelos: un pickle de scikit-learn no es portable entre versiones
menores.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements-dev.txt
python tasks.py test                   # no necesita Docker, MLflow ni datos
```

## De cero a una API sirviendo (Hito 3)

El modelo **no está en el repo ni en la imagen**: vive en el registry de MLflow y la
API lo descarga al arrancar, por alias.

```bash
python tasks.py mlflow                 # registry + UI en http://127.0.0.1:5000
python tasks.py train                  # entrena en contenedor Linux y REGISTRA (sin promover)
python tasks.py promover ocupados 1    # champion → v1   (o desde la UI de MLflow)
python tasks.py promover no_ocupados 1
python tasks.py up                     # API en http://127.0.0.1:8000/docs, espera a /ready
python tasks.py smoke
```

Registrar no es desplegar: una versión nueva no tiene alias y nadie la sirve hasta que
alguien la promueve. Los contenedores vivos no cambian al mover el alias —lo leen al
arrancar—, así que para que un cambio llegue: `docker compose restart api`. Rollback es
lo mismo con la versión anterior.

`train` necesita `data/processed/` (lo generan los notebooks 01–02); se monta en sólo
lectura, no se copia a la imagen.

`python tasks.py` sin argumentos lista todas las tareas.

## Estructura

```
src/                código importable: modelo, preprocesamiento, predicción, entrenamiento
app/                capa HTTP (FastAPI). Sólo traduce; la lógica vive en src/
docker/mlflow/      imagen del servidor de MLflow
docker-compose.yml  mlflow + train + api
models/             model card (los modelos viven en el registry)
notebooks/          01 dataset · 02 EDA · 03 selección · 04 modelado
scripts/            promover.py: mueve aliases del registry
tests/              pruebas, con modelos sintéticos: corren sin datos ni registry
```

## Nota sobre los modelos

Cada modelo registrado lleva dentro el contrato de entrada (tipos, valores admitidos y
la pregunta del formulario) y las versiones con las que se serializó. La API valida
contra ese contrato, así que el formulario y el modelo no pueden separarse.

`GET /contrato/{segmento}` lo expone en vivo; `GET /ready` dice qué versión exacta
sirve cada réplica, y cada respuesta de `/predict` trae `version_modelo`.
