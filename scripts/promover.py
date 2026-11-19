"""Mueve un alias del registry a mano: promover, o hacer rollback, es el mismo gesto.

    python tasks.py promover ocupados 3              # champion → v3
    python tasks.py promover no_ocupados 2 challenger

Desde el Hito 8 el camino normal es el gate (`src/promocion.py`, última tarea del DAG
`reentrenar`). Esto es la decisión HUMANA, para lo que el gate no hace a propósito:
nombrar el primer champion, hacer rollback, o promover algo que una persona revisó.

Mover @champion deja el MISMO rastro que el gate (src.promocion.mover_champion): el
champion previo queda como @anterior, @challenger se retira si era ésta, y la versión
queda etiquetada `promovida_por=manual`. Imprime el rollback para copiar y pegar.

Los pods vivos no cambian: el alias se lee al arrancar. Para que el cambio llegue:
    kubectl rollout restart deployment/api -n ml-predict-income     (kind, Hito 6)
    docker compose restart api                                      (compose)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.promocion import mover_champion  # noqa: E402

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

    if a.alias == "champion":
        anterior = mover_champion(client, nombre, str(mv.version), "manual",
                                  "decisión humana (tasks.py promover)")
    else:
        anterior = client.get_registered_model(nombre).aliases.get(a.alias)
        client.set_registered_model_alias(nombre, a.alias, a.version)

    run = client.get_run(mv.run_id)
    m = run.data.metrics
    print(f"{nombre}@{a.alias}: v{anterior or '—'} → v{a.version}")
    print(f"  run {mv.run_id[:8]}  git {run.data.tags.get('git_sha', '?')}  "
          f"plataforma {run.data.tags.get('plataforma', '?')}")
    print(f"  test MdAPE {m.get('test_MdAPE', float('nan')):.2f}%  "
          f"cobertura {m.get('test_cobertura_80', float('nan')):.1f}%")
    if anterior and str(anterior) != str(a.version):
        if a.alias == "champion":
            print(f"  @anterior → v{anterior}")
        print(f"  rollback: python tasks.py promover {a.segmento} {anterior} {a.alias}")
    print("  los pods vivos no cambian hasta reiniciar:  "
          "kubectl rollout restart deployment/api -n ml-predict-income")


if __name__ == "__main__":
    main()
