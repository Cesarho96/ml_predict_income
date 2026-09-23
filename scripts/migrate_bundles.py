"""Migra bundles serializados desde un notebook a la ruta de importación estable.

Por qué hace falta: pickle guarda la RUTA de cada clase. Los bundles que produjo el
notebook 04 apuntan a `__main__.ACategorias`, que sólo existe dentro de aquel kernel.
Este script los abre inyectando la clase real en `__main__`, y los vuelve a guardar —
ya apuntando a `src.preprocessing.ACategorias`.

Es idempotente: correrlo sobre un bundle ya migrado no cambia nada.

    python scripts/migrate_bundles.py models/
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy
import pandas
import sklearn

import __main__
from src.preprocessing import ACategorias

# El puente: pickle busca `__main__.ACategorias` y encuentra la clase real, así que el
# objeto reconstruido ya es `src.preprocessing.ACategorias` y al re-serializar queda bien.
__main__.ACategorias = ACategorias

VERSIONES = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
             "numpy": numpy.__version__, "pandas": pandas.__version__,
             "joblib": joblib.__version__}


def migrar(ruta: Path) -> None:
    bundle = joblib.load(ruta)
    prep = bundle["cuantiles"]["0.5"].named_steps["prep"]
    origen = type(prep).__module__
    # Se anotan las versiones con las que el artefacto se puede abrir: sin esto, un
    # `pip install -U scikit-learn` en la imagen rompe la carga sin decir por qué.
    bundle.setdefault("versiones_entrenamiento", VERSIONES)
    joblib.dump(bundle, ruta, compress=3)
    destino = type(joblib.load(ruta)["cuantiles"]["0.5"].named_steps["prep"]).__module__
    print(f"  {ruta.name:<34} {origen}  →  {destino}")


if __name__ == "__main__":
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    archivos = sorted(carpeta.glob("*.joblib"))
    if not archivos:
        sys.exit(f"sin .joblib en {carpeta}")
    print(f"migrando {len(archivos)} bundles · versiones: {VERSIONES}")
    for a in archivos:
        migrar(a)
    print("listo — ahora `joblib.load` funciona desde cualquier proceso")
