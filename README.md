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

Requiere **Python 3.11**. Las versiones están clavadas a las que serializaron los
modelos: un pickle de scikit-learn no es portable entre versiones menores, así que un
`pip install -U` rompe la carga.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements-dev.txt

python tasks.py test      # 14 pruebas
python tasks.py api       # API en http://127.0.0.1:8000/docs
python tasks.py smoke     # en otra terminal
```

`python tasks.py` sin argumentos lista todas las tareas. No hace falta `make`:
el ejecutor es Python puro y corre igual en Windows, Linux y en CI.

## Con Docker

```bash
python tasks.py build
python tasks.py run       # con --read-only, sin capabilities, 1 CPU, 1 GB
python tasks.py smoke
python tasks.py inspect   # tamaño de la imagen y usuario (debe ser uid 10001, no root)
```

## Estructura

```
src/         código importable: preprocesamiento y predicción
app/         capa HTTP (FastAPI). Sólo traduce; la lógica vive en src/
models/      bundles serializados + contrato de features + model card
notebooks/   01 dataset · 02 EDA · 03 selección · 04 modelado
scripts/     utilidades de mantenimiento
tests/       pruebas
```

## Nota sobre los modelos

Los bundles llevan dentro el contrato de entrada (tipos, valores admitidos y la
pregunta que hace el formulario) y las versiones con las que se serializaron. La API
valida contra ese contrato, así que el formulario y el modelo no pueden separarse.

`GET /contrato/{segmento}` lo expone en vivo.
