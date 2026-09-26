"""Compara lo que acaba de entrenar el DAG contra el champion y marca el @challenger.

    python -m src.comparar                    # la corrida viene en $ORQUESTADOR_RUN_ID
    python -m src.comparar --corrida <id>

Última tarea del DAG `reentrenar`. Tres reglas:

1. **Nunca toca @champion.** Promover es una decisión (de una persona hoy, de un gate
   con más evidencia en el Hito 8). Esta tarea sólo deja el candidato listo y el
   veredicto escrito en el registry, al lado de la versión.

2. **Compara sobre los MISMOS datos.** No compara las métricas que cada corrida loggeó:
   si la ENIGH cambió, cada una se midió en un test distinto y la diferencia no dice
   nada. Carga los dos modelos y los evalúa sobre el test de los datos actuales.

3. **Una diferencia menor que la banda de ruido no es una mejora.** Las bandas vienen del
   notebook 04: cuánto se mueve el MdAPE de test sólo por cambiar la semilla. Un "30.5%
   contra 30.8%" dentro de ±0.78 pp es empate, no progreso.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.model import ModeloIngreso
from src.train import PROC, SEGMENTOS, TARGET, cargar_datos, metricas

EXPERIMENTO = "ingreso-enigh2024"
CHAMPION, CHALLENGER = "champion", "challenger"

# Banda de ruido del MdAPE de test, en puntos porcentuales (notebook 04: desviación entre
# semillas con los mismos datos y los mismos hiperparámetros).
BANDA_PP = {"ocupados": 0.78, "no_ocupados": 0.55}
# El intervalo promete 80%. Por debajo de 78% la promesa se rompe, aunque el MdAPE mejore.
COBERTURA_MINIMA = 78.0

MARCAN_CHALLENGER = ("mejor", "equivalente", "sin champion")


@dataclass
class Resultado:
    segmento: str
    nombre: str
    version_nueva: str
    version_champion: str | None
    mdape_nuevo: float
    mdape_champion: float | None
    delta_pp: float | None
    banda_pp: float
    cobertura_nuevo: float
    cobertura_champion: float | None
    veredicto: str
    marcado_challenger: bool


def veredicto(delta_pp: float | None, banda_pp: float, cobertura: float) -> str:
    """delta_pp = MdAPE nuevo − MdAPE champion. Negativo = el nuevo se equivoca menos."""
    if cobertura < COBERTURA_MINIMA:
        return "peor (cobertura)"
    if delta_pp is None:
        return "sin champion"
    if delta_pp < -banda_pp:
        return "mejor"
    if delta_pp > banda_pp:
        return "peor"
    return "equivalente"


# ------------------------------------------------------------------ registry
def version_de_la_corrida(client: MlflowClient, nombre: str, slug: str, corrida: str):
    """La versión que registró ESTA corrida del DAG, encontrada por su etiqueta de linaje.

    No "la versión más nueva": si alguien entrenó a mano mientras el DAG corría, la más
    nueva sería otra. El tag `orquestador_run_id` (lo pone src/train.py) no se equivoca.
    """
    exp = client.get_experiment_by_name(EXPERIMENTO)
    if exp is None:
        raise RuntimeError(f"no existe el experimento {EXPERIMENTO!r}")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.orquestador_run_id = '{corrida}' and tags.segmento = '{slug}'")
    if len(runs) != 1:
        raise RuntimeError(f"esperaba 1 corrida de {slug} para {corrida!r}, hay {len(runs)}")
    versiones = client.search_model_versions(
        f"name = '{nombre}' and run_id = '{runs[0].info.run_id}'")
    if len(versiones) != 1:
        raise RuntimeError(f"la corrida {runs[0].info.run_id} no registró {nombre}")
    return versiones[0]


def champion_de(client: MlflowClient, nombre: str):
    try:
        return client.get_model_version_by_alias(nombre, CHAMPION)
    except MlflowException:
        return None


def cargar(nombre: str, version: str) -> ModeloIngreso:
    return mlflow.pyfunc.load_model(f"models:/{nombre}/{version}").unwrap_python_model()


# ------------------------------------------------------------------ evaluación
def matriz(sub: pd.DataFrame, modelo: ModeloIngreso) -> pd.DataFrame:
    """Las columnas que ESTE modelo espera, con los tipos de su contrato.

    Cada modelo trae su lista de features: si un reentrenamiento cambió las variables,
    el champion se evalúa con las suyas y el nuevo con las suyas, sobre las mismas filas.
    """
    X = sub[modelo.features].copy()
    for c in modelo.contrato:
        col = c["variable"]
        X[col] = (X[col].astype(str) if c["tipo"] == "categórica"
                  else pd.to_numeric(X[col]).astype("float64"))
    return X


def evaluar(modelo: ModeloIngreso, test: pd.DataFrame) -> tuple[float, float]:
    """(MdAPE %, cobertura del intervalo %), ambos ponderados por el factor de expansión."""
    salida = modelo.predict(None, matriz(test, modelo))
    y, w = test[TARGET].to_numpy(), test["factor"].to_numpy()
    mdape = metricas(y, salida["mediana"].to_numpy(), w)["MdAPE"]
    dentro = (y >= salida["inferior"].to_numpy()) & (y <= salida["superior"].to_numpy())
    return mdape, round(float(np.average(dentro, weights=w)) * 100, 2)


# ------------------------------------------------------------------ principal
def comparar(corrida: str, proc: Path = PROC, client: MlflowClient | None = None
             ) -> list[Resultado]:
    client = client or MlflowClient()
    df = cargar_datos(proc)
    resultados = []

    for slug, cfg in SEGMENTOS.items():
        nombre = cfg["nombre_mlflow"]
        test = df[(df["segmento"] == cfg["etiqueta"]) & (df["particion"] == "test")]

        nueva = version_de_la_corrida(client, nombre, slug, corrida)
        mdape_n, cob_n = evaluar(cargar(nombre, nueva.version), test)

        champ = champion_de(client, nombre)
        mdape_c = cob_c = delta = None
        if champ is not None:
            mdape_c, cob_c = evaluar(cargar(nombre, champ.version), test)
            delta = round(mdape_n - mdape_c, 2)

        banda = BANDA_PP[slug]
        v = veredicto(delta, banda, cob_n)
        marcar = v in MARCAN_CHALLENGER

        # El veredicto queda escrito EN la versión: quien abra el registry lo ve sin
        # buscar logs de Airflow.
        for k, val in {"veredicto": v, "delta_MdAPE_pp": delta, "banda_pp": banda,
                       "champion_comparado": champ.version if champ else None,
                       "orquestador_run_id": corrida}.items():
            if val is not None:
                client.set_model_version_tag(nombre, nueva.version, k, str(val))
        if marcar:
            client.set_registered_model_alias(nombre, CHALLENGER, nueva.version)

        resultados.append(Resultado(
            slug, nombre, str(nueva.version), str(champ.version) if champ else None,
            mdape_n, mdape_c, delta, banda, cob_n, cob_c, v, marcar))
    return resultados


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def informe(resultados: list[Resultado]) -> str:
    lineas = ["", f"{'segmento':<12} {'nueva':>6} {'champ':>6} {'MdAPE nuevo':>12} "
                  f"{'MdAPE champ':>12} {'Δ pp':>7} {'banda':>6}  veredicto"]
    for r in resultados:
        lineas.append(
            f"{r.segmento:<12} {'v' + r.version_nueva:>6} "
            f"{'v' + r.version_champion if r.version_champion else '—':>6} "
            f"{_num(r.mdape_nuevo):>12} {_num(r.mdape_champion):>12} {_num(r.delta_pp):>7} "
            f"±{r.banda_pp:<5}  {r.veredicto}"
            + ("  → @challenger" if r.marcado_challenger else "  (sin alias)"))
    lineas.append("@champion NO se tocó. Para promover: python tasks.py promover "
                  "<segmento> <versión>")
    return "\n".join(lineas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrida", default=os.getenv("ORQUESTADOR_RUN_ID", "").strip())
    a = ap.parse_args()
    if not a.corrida:
        sys.exit("falta --corrida (o la variable ORQUESTADOR_RUN_ID)")
    resultados = comparar(a.corrida)
    print(informe(resultados))
    # Última línea = XCom del DockerOperator: un resumen chico, no datos.
    print(json.dumps({r.segmento: {"version": r.version_nueva, "veredicto": r.veredicto}
                      for r in resultados}, ensure_ascii=False))


if __name__ == "__main__":
    main()

