"""Mueve un alias del registry: promover, o hacer rollback, es el mismo gesto.

    python tasks.py promover ocupados 3              # champion → v3
    python tasks.py promover no_ocupados 2 challenger

Registrar un modelo NO lo pone en producción: el entrenamiento deja una versión nueva
sin alias. Ponerla a servir es una decisión explícita —de una persona hoy, de un gate
automático en el Hito 8— y este script es esa decisión, con rastro: imprime qué versión
tenía el alias antes, para que el rollback sea copiar y pegar.

Los pods vivos no cambian: el alias se lee al arrancar. Para que el cambio llegue:
    docker compose restart api        (local)
    kubectl rollout restart ...       (Hito 6)
"""

from __future__ import annotations

import argparse
import os
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

NOMBRES = {"ocupados": "ingreso_ocupados", "no_ocupados": "ingreso_no_ocupados"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segmento", choices=list(NOMBRES))
    ap.add_argument("version")
    ap.add_argument("alias", nargs="?", default="champion")
    a = ap.parse_args()

    nombre = NOMBRES[a.segmento]
    client = MlflowClient()
    try:
        mv = client.get_model_version(nombre, a.version)
    except MlflowException:
        sys.exit(f"{nombre} no tiene versión {a.version} en {os.getenv('MLFLOW_TRACKING_URI')}")

    anterior = client.get_registered_model(nombre).aliases.get(a.alias)
    client.set_registered_model_alias(nombre, a.alias, a.version)

    run = client.get_run(mv.run_id)
    m = run.data.metrics
    print(f"{nombre}@{a.alias}: v{anterior or '—'} → v{a.version}")
    print(f"  run {mv.run_id[:8]}  git {run.data.tags.get('git_sha', '?')}  "
          f"plataforma {run.data.tags.get('plataforma', '?')}")
    print(f"  test MdAPE {m.get('test_MdAPE', float('nan')):.2f}%  "
          f"cobertura {m.get('test_cobertura_80', float('nan')):.1f}%")
    if anterior:
        print(f"  rollback: python tasks.py promover {a.segmento} {anterior} {a.alias}")
    print("  los pods vivos no cambian hasta reiniciar:  docker compose restart api")


if __name__ == "__main__":
    main()
