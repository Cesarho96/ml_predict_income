"""Entrena, evalúa y registra los dos modelos. Reproduce el notebook 04 sin exploración.

    python tasks.py train               # la forma canónica: en contenedor Linux
    python -m src.train                 # directo, contra $MLFLOW_TRACKING_URI
    python -m src.train --sin-registrar # sólo loggea la corrida

Dónde se entrena importa. El Hito 3 demostró que el mismo commit, los mismos datos, la
misma semilla y las mismas versiones de librerías dan un modelo DISTINTO en Windows y
en Linux: scikit-learn ordena las categorías con el `qsort` de la librería C del
sistema, y glibc y la de Microsoft desempatan distinto. Por eso el entrenamiento
canónico corre en el contenedor `train` (la misma base Linux que sirve) y cada corrida
queda etiquetada con su plataforma.

Lo que este script NO hace, a propósito:

- **No elige variables.** Las lee de `features_*.json`, congeladas en el notebook 03.
- **No particiona.** Lee `split_upm_3way.csv`. Volver a partir aquí invalidaría en
  silencio la selección de variables, que se hizo con esas filas exactas de train.
- **No busca hiperparámetros.** Están abajo, con su procedencia. El tuning fue una
  actividad de notebook y su resultado es una constante, no algo que se re-derive en
  cada corrida.

Esa disciplina es la que hace que dos corridas del mismo commit den el mismo modelo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.model import ALFAS, ModeloIngreso, derivar_contrato, ejemplo_de_entrada
from src.preprocessing import ACategorias
from src.validacion import validar

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
TARGET = "ingreso_mensual"
COBERTURA_OBJETIVO = 0.80

# Hiperparámetros del notebook 04 §6 (RandomizedSearch sobre validation → grid local).
# Se dejan explícitos y no se re-derivan: el tuning es una decisión tomada, no un paso
# del pipeline. Si cambian, cambian aquí y la corrida lo registra.
#
# `random_state` está aquí por una razón que costó una corrida descubrir: el notebook 04
# guardó en `hiperparametros` el `best_params_` de la búsqueda, y ahí NO aparece la
# semilla —la búsqueda nunca la varió, así que no es uno de "sus" parámetros. Entrenar
# con 42 en vez del 0 del notebook movió el MdAPE de no ocupados 1.21 pp, por encima de
# su banda de ruido de 0.55 pp. El artefacto no bastaba para reproducirse a sí mismo.
# Lección: los parámetros de una corrida son TODOS los que el estimador recibió, no sólo
# los que alguien buscó.
SEGMENTOS = {
    "ocupados": {
        "nombre_mlflow": "ingreso_ocupados",
        "contrato": "features_ocupados.json",
        "etiqueta": "Ocupados",
        "params": {"min_samples_leaf": 40, "max_leaf_nodes": 127, "max_iter": 300,
                   "learning_rate": 0.08, "l2_regularization": 0, "random_state": 0},
    },
    "no_ocupados": {
        "nombre_mlflow": "ingreso_no_ocupados",
        "contrato": "features_no_ocupados.json",
        "etiqueta": "No ocupados",
        "params": {"min_samples_leaf": 80, "max_leaf_nodes": 63, "max_iter": 300,
                   "learning_rate": 0.12, "l2_regularization": 0, "random_state": 0},
    },
}


# ---------------------------------------------------------------- métricas
def wq(v, q, w) -> float:
    v, w = np.asarray(v, float), np.asarray(w, float)
    o = np.argsort(v)
    return float(np.interp(q, np.cumsum(w[o]) / w.sum(), v[o]))


def metricas(y, pred, w) -> dict[str, float]:
    pred = np.clip(pred, 1, None)
    err = np.abs(y - pred)
    ape = err / y
    res_log = np.log(y) - np.log(pred)
    return {
        "MdAPE": round(wq(ape, .5, w) * 100, 2),
        "pct_20": round(float(np.average(ape <= .20, weights=w)) * 100, 2),
        "MdAE": round(wq(err, .5, w), 0),
        "MAE": round(float(np.average(err, weights=w)), 0),
        "RMSE": round(float(np.sqrt(np.average((y - pred) ** 2, weights=w))), 0),
        "RMSE_log": round(float(np.sqrt(np.average(res_log ** 2, weights=w))), 4),
        "R2_log": round(float(1 - np.average(res_log ** 2, weights=w)
                              / np.average((np.log(y) - np.average(np.log(y), weights=w)) ** 2,
                                           weights=w)), 4),
    }


def sha_git() -> str:
    """El commit que produjo el modelo. En el contenedor no hay git: llega por GIT_SHA."""
    if sha := os.getenv("GIT_SHA", "").strip():
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "desconocido"


# ---------------------------------------------------------------- datos
def cargar_datos() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "enigh2024_features_v2.parquet")
    split = pd.read_csv(PROC / "split_upm_3way.csv", dtype={"upm": str})
    df["upm"] = df["upm"].astype(str)
    df = df.merge(split, on="upm", how="left", validate="many_to_one")
    if df["particion"].isna().any():
        raise ValueError("hay upm sin partición en split_upm_3way.csv")
    return df


def preparar_segmento(df: pd.DataFrame, cfg: dict) -> dict:
    spec = json.loads((PROC / cfg["contrato"]).read_text(encoding="utf-8"))
    feats = spec["features"]
    sub = df[df.segmento == cfg["etiqueta"]].reset_index(drop=True)

    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(sub[c])]
    X = sub[feats].copy()
    for c in feats:
        X[c] = X[c].astype(str) if c in cat else pd.to_numeric(X[c]).astype("float64")

    p = sub["particion"].to_numpy()
    d = {"X": X, "cat": cat, "spec": spec, "features": feats,
            "y": np.log(sub[TARGET].clip(lower=1)).to_numpy(),
            "y_pesos": sub[TARGET].to_numpy(), "w": sub["factor"].to_numpy(),
            "tr": p == "train", "va": p == "validation", "te": p == "test"}
    d["contrato"] = construir_contrato(d)
    return d


def construir_contrato(d: dict) -> list[dict]:
    """El contrato sale de las filas de TRAIN (ver `src.model.derivar_contrato`)."""
    preguntas = {f["variable"]: f.get("pregunta", "") for f in d["spec"]["formulario"]}
    return derivar_contrato(d["X"][d["tr"]], d["cat"], preguntas)


# ---------------------------------------------------------------- entrenamiento
def entrenar_segmento(d: dict, params: dict) -> tuple[dict, float]:
    """Ajusta los tres cuantiles en train y calibra el conformal en validation."""
    cuantiles = {}
    for alfa in ALFAS:
        pipe = Pipeline([
            ("prep", ACategorias(d["cat"])),
            ("modelo", HistGradientBoostingRegressor(
                loss="quantile", quantile=float(alfa),
                categorical_features="from_dtype", **params)),
        ])
        pipe.fit(d["X"][d["tr"]], d["y"][d["tr"]])
        cuantiles[alfa] = pipe

    # Conformal sobre validation: el modelo nunca vio esas filas, así que la cobertura
    # que se mide ahí es la que se puede prometer.
    Xv, yv, wv = d["X"][d["va"]], d["y"][d["va"]], d["w"][d["va"]]
    lo, mid, hi = (cuantiles[a].predict(Xv) for a in ALFAS)

    def cobertura(f: float) -> float:
        return float(np.average((yv >= mid - f * (mid - lo)) & (yv <= mid + f * (hi - mid)),
                                weights=wv))

    rejilla = np.arange(1.0, 2.51, 0.02)
    cobs = np.array([cobertura(f) for f in rejilla])
    factor = float(rejilla[np.argmax(cobs >= COBERTURA_OBJETIVO)]) if (
        cobs >= COBERTURA_OBJETIVO).any() else 2.5
    log.info("  cobertura cruda %.3f → factor %.2f → calibrada %.3f",
             cobertura(1.0), factor, cobertura(factor))
    return cuantiles, factor


def evaluar(modelo: ModeloIngreso, d: dict, parte: str) -> dict:
    m = d[parte]
    salida = modelo.predict(None, d["X"][m])
    met = metricas(d["y_pesos"][m], salida["mediana"].to_numpy(), d["w"][m])
    dentro = ((d["y_pesos"][m] >= salida["inferior"]) & (d["y_pesos"][m] <= salida["superior"]))
    met["cobertura_80"] = round(float(np.average(dentro, weights=d["w"][m])) * 100, 2)
    met["ancho_mediano"] = round(float(np.median(salida["superior"] - salida["inferior"])), 0)
    return met


# ---------------------------------------------------------------- main
def configurar_mlflow(experimento: str) -> None:
    """Hacia dónde se registra: `MLFLOW_TRACKING_URI`, o un SQLite local si no hay.

    Hito 2: SQLite + carpeta local. Funcionó, pero guardó cada artefacto con una ruta
    ABSOLUTA de la máquina que entrenó (`file:///D:/repositories/...`). Un contenedor
    Linux no puede abrir eso: el registry existía, pero sólo para una computadora.

    Hito 3: un servidor de MLflow con `--serve-artifacts`. Los artefactos se suben y se
    bajan POR HTTP a través del servidor (`mlflow-artifacts:/...`), así que cualquier
    cliente que alcance la URL puede usarlos, sin montar discos ni compartir rutas.
    Cambiar a Postgres + S3 en AWS será cambiar las banderas del servidor, no este código.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip() or f"sqlite:///{RAIZ / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experimento) is None:
        # Con servidor, el servidor decide dónde van los artefactos; con SQLite local,
        # una carpeta junto al repo.
        destino = None if uri.startswith("http") else (RAIZ / "mlartifacts").as_uri()
        mlflow.create_experiment(experimento, artifact_location=destino)
    mlflow.set_experiment(experimento)
    log.info("MLflow: %s (experimento %s)", uri, experimento)


def main(registrar: bool = True, experimento: str = "ingreso-enigh2024") -> None:
    configurar_mlflow(experimento)

    df = cargar_datos()
    resumen = {}

    for slug, cfg in SEGMENTOS.items():
        log.info("=== %s ===", cfg["etiqueta"])
        d = preparar_segmento(df, cfg)

        with mlflow.start_run(run_name=f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"):
            mlflow.set_tags({"segmento": slug, "familia": "HistGradientBoosting",
                             "git_sha": sha_git(), "objetivo": "quantile",
                             "target": "log(ingreso_mensual)",
                             # Mismo commit ≠ mismo modelo si cambia la plataforma.
                             "plataforma": platform.system(),
                             "python": platform.python_version()})
            mlflow.log_params({**cfg["params"], "n_features": len(d["features"]),
                               "cobertura_objetivo": COBERTURA_OBJETIVO,
                               "entrenado_ponderado": False})
            mlflow.log_metrics({"n_train": int(d["tr"].sum()),
                                "n_validation": int(d["va"].sum()),
                                "n_test": int(d["te"].sum())})

            t0 = time.time()
            cuantiles, factor = entrenar_segmento(d, cfg["params"])
            mlflow.log_metric("segundos_entrenamiento", round(time.time() - t0, 1))
            mlflow.log_param("factor_conformal", factor)

            versiones = {"scikit-learn": __import__("sklearn").__version__,
                         "pandas": pd.__version__, "numpy": np.__version__}
            modelo = ModeloIngreso(cuantiles, factor, d["features"], d["contrato"],
                                   meta={"segmento": cfg["etiqueta"],
                                         "familia": "HistGradientBoosting",
                                         "cobertura_objetivo": COBERTURA_OBJETIVO,
                                         "versiones_entrenamiento": versiones})

            for parte, etiqueta in (("va", "validation"), ("te", "test")):
                met = evaluar(modelo, d, parte)
                mlflow.log_metrics({f"{etiqueta}_{k}": v for k, v in met.items()})
                if parte == "te":
                    resumen[slug] = met
                    log.info("  test: %s", met)

            # El contrato va en DOS lugares, y no es redundancia:
            #  · artefacto del MODELO → viaja con el paquete de despliegue; quien carga
            #    `models:/...@champion` lo tiene a la mano sin buscar la corrida.
            #  · artefacto de la CORRIDA → permite diffear contratos entre corridas en
            #    la UI, que es como se detecta que un reentrenamiento cambió las
            #    categorías admitidas.
            tmp = Path(tempfile.mkdtemp()) / "contrato.json"
            tmp.write_text(json.dumps(d["contrato"], indent=2, ensure_ascii=False),
                           encoding="utf-8")

            # Antes de registrar: un modelo roto no debe llegar al registry, donde
            # alguien podría promoverlo. Si falla, la excepción corta aquí y la corrida
            # queda marcada como FAILED en MLflow, sin versión registrada.
            for ok in validar(modelo, d["X"][d["te"]]):
                log.info("  ✓ %s", ok)
            mlflow.set_tag("validacion", "ok")

            ejemplo = ejemplo_de_entrada(d["contrato"], d["features"])
            firma = infer_signature(ejemplo, modelo.predict(None, ejemplo))

            # `code_paths` mete src/ dentro del artefacto: es la solución de MLflow al
            # problema que nos mordió en el Hito 1 (una clase que sólo existía en el
            # notebook). El modelo viaja con el código que necesita para deserializarse.
            info = mlflow.pyfunc.log_model(
                name="modelo",
                python_model=modelo,
                signature=firma,
                input_example=ejemplo,
                # Se copia src/ completo, no módulos sueltos: los pipelines están
                # serializados contra la ruta `src.preprocessing.ACategorias`, así que
                # el paquete tiene que existir con ese nombre al deserializar. El costo
                # es que train.py también viaja dentro del artefacto de servicio.
                code_paths=[str(RAIZ / "src")],
                artifacts={"contrato": str(tmp)},
                pip_requirements=[f"{p}=={v}" for p, v in versiones.items()],
                registered_model_name=cfg["nombre_mlflow"] if registrar else None,
            )
            mlflow.log_dict(d["contrato"], "contrato.json")
            version = getattr(info, "registered_model_version", None)
            log.info("  registrado: %s%s", cfg["nombre_mlflow"] if registrar else info.model_uri,
                     f" v{version}" if version else "")
            # Ya no se escribe ningún .joblib: el registry es la única copia del modelo.
            # Registrar NO es promover: la versión nueva no tiene alias y nadie la sirve
            # hasta que una persona (o, en el Hito 8, un gate) la marque como champion.

    print("\n" + "=" * 70)
    for slug, met in resumen.items():
        print(f"{slug:<14} MdAPE {met['MdAPE']:.2f}%  RMSE ${met['RMSE']:,.0f}  "
              f"cobertura {met['cobertura_80']:.1f}%")
    print("=" * 70)
    print(f"Revisa y promueve en {mlflow.get_tracking_uri()}  "
          "(o: python tasks.py promover <segmento> <versión>)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sin-registrar", action="store_true",
                    help="loggea la corrida pero no registra el modelo")
    ap.add_argument("--experimento", default="ingreso-enigh2024")
    a = ap.parse_args()
    main(registrar=not a.sin_registrar, experimento=a.experimento)
