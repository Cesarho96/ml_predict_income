"""Microdato crudo de la ENIGH → dataset listo para validar y entrenar.

    python -m src.pipeline.construir                  # edición de $EDICION (2024 por omisión)
    python -m src.pipeline.construir --edicion 2026

Primera tarea del DAG `reentrenar`. Escribe en la carpeta de datos procesados:
  · enigh{edición}_features_v2.parquet          el dataset (mismo nombre y forma que el notebook)
  · enigh{edición}_features_v2.meta.json        cuántas filas, sha256, de qué commit salió
  · split_upm_3way.csv                          sólo si aparecieron UPM nuevas

El parquet se escribe a un temporal y se renombra al final: una corrida que falle a la
mitad nunca deja un dataset a medias que la siguiente tarea tome por bueno.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from src import datos
from src.pipeline.features import construir_features, features_del_modelo
from src.pipeline.ingesta import construir_personas
from src.pipeline.particion import extender


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def construir(edicion: int, crudos: Path = datos.CRUDOS, proc: Path = datos.PROC) -> dict:
    t0 = time.time()
    personas = construir_personas(crudos, edicion)
    df = construir_features(personas, features_del_modelo(proc))

    split, nuevas = extender(df["upm"].unique(),
                             pd.read_csv(proc / datos.SPLIT, dtype={"upm": str}))
    if nuevas:
        split.to_csv(proc / datos.SPLIT, index=False)

    salida = proc / datos.dataset(edicion)
    tmp = salida.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(salida)                       # renombre atómico

    meta = {
        "edicion": edicion,
        "filas": len(df),
        "columnas": list(df.columns),
        "por_segmento": df["segmento"].value_counts().to_dict(),
        "upm_nuevas_particionadas": nuevas,
        "sha256": sha256(salida),
        "git_sha": os.getenv("GIT_SHA", "desconocido"),
        "segundos": round(time.time() - t0, 1),
    }
    salida.with_name(salida.name.replace(".parquet", ".meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edicion", type=int, default=datos.EDICION)
    a = ap.parse_args()
    meta = construir(a.edicion)
    print(f"dataset {datos.dataset(a.edicion)}: {meta['filas']:,} filas en {meta['segundos']} s")
    print(json.dumps({k: meta[k] for k in ("edicion", "filas", "sha256",
                                            "upm_nuevas_particionadas")}))


if __name__ == "__main__":
    main()
