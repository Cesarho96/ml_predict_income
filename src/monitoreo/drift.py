"""Drift de entrada con PSI (Hito 7c): ¿se parecen las personas que piden predicciones a
la población con la que se entrenó el modelo?

    python -m src.monitoreo.drift --horas 24        # lo que corre el DAG `monitor`

Sin ground truth en producción (nadie nos dice su ingreso real), esto es lo único medible
sobre la calidad del modelo en uso: si la entrada se aleja de lo que el modelo conoce,
sus intervalos dejan de estar respaldados por datos.

PSI (Population Stability Index), por feature:

    PSI = Σ  (p_actual − p_referencia) · ln(p_actual / p_referencia)       sobre los bins

  · Categóricas: un bin por categoría.
  · Numéricas: bins por DECILES de la referencia (10 % cada uno), así cada bin pesa lo
    mismo y los extremos no quedan vacíos.
  · Lectura estándar (riesgo de crédito): < 0.10 estable · 0.10–0.25 vigilar · > 0.25 actuar.

Dos decisiones que importan:

1. La referencia es la POBLACIÓN, no la muestra. La ENIGH sobre-muestrea algunos estados
   y lo corrige con el factor de expansión. Comparar contra la muestra cruda "detectaría"
   el diseño de la encuesta. Por eso las proporciones de referencia se ponderan por
   `factor`, igual que las métricas del modelo.

2. El PSI tiene sesgo con muestras chicas: aun sin drift, con n peticiones y k bins su
   valor esperado ronda (k − 1) / n. Con 32 estados y 100 peticiones ≈ 0.31: "actuar" por
   puro azar. Por eso cada feature pide n ≥ max(300, 20·(k − 1)) para opinar con calma.
   Con menos, sólo opina si la señal está muy por encima del azar (umbral + 3 veces el
   ruido esperado): 300 peticiones todas de la CDMX no son ruido, aunque sean pocas. Si
   no alcanza, el resultado se guarda como `insuficiente`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import datos

EPS = 1e-4                    # una proporción 0 haría ln(0); se sustituye por esto
UMBRAL_VIGILAR, UMBRAL_ACTUAR = 0.10, 0.25
DECILES = np.linspace(0.1, 0.9, 9)
FALTANTE = "(faltante)"

# El API nombra los segmentos en minúsculas; el dataset, como en el notebook.
SEGMENTOS = {"ocupados": "Ocupados", "no_ocupados": "No ocupados"}


# ===================================================================== matemática
def psi(p_ref: np.ndarray, p_act: np.ndarray) -> float:
    q = np.clip(np.asarray(p_ref, dtype=float), EPS, None)
    p = np.clip(np.asarray(p_act, dtype=float), EPS, None)
    return float(np.sum((p - q) * np.log(p / q)))


def n_minimo(k_bins: int) -> int:
    """Peticiones necesarias para que el ruido del PSI (≈ (k−1)/n) quede bajo ~0.05."""
    return max(300, 20 * (k_bins - 1))


def nivel(valor: float, n: int, minimo: int, k_bins: int) -> str:
    ruido = (k_bins - 1) / max(n, 1)          # PSI esperado SIN drift, sólo por azar
    if n == 0 or np.isnan(valor):
        return "insuficiente"
    if n < minimo and valor < UMBRAL_ACTUAR + 3 * ruido:
        return "insuficiente"
    if valor < UMBRAL_VIGILAR:
        return "estable"
    return "vigilar" if valor < UMBRAL_ACTUAR else "actuar"


def cuantiles_ponderados(x: np.ndarray, w: np.ndarray, qs: np.ndarray) -> np.ndarray:
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    acumulado = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, acumulado, x)


# ===================================================================== referencia
@dataclass
class Feature:
    """Cómo se discretiza una feature, aprendido de la referencia."""
    nombre: str
    tipo: str                               # "categorica" | "numerica"
    bins: list[str]                         # etiquetas, en orden
    p_ref: np.ndarray                       # proporción ponderada de la población por bin
    cortes: np.ndarray | None = None        # sólo numéricas: límites superiores

    def asignar(self, valores: pd.Series) -> pd.Series:
        """Etiqueta de bin para cada valor (de la referencia o de producción)."""
        if self.tipo == "categorica":
            return normalizar_categoria(valores)
        x = pd.to_numeric(valores, errors="coerce")
        idx = np.searchsorted(self.cortes, x.to_numpy(dtype=float), side="left")
        etiquetas = np.asarray(self.bins[: len(self.cortes) + 1], dtype=object)[idx]
        return pd.Series(np.where(x.isna(), FALTANTE, etiquetas), index=valores.index)


def normalizar_categoria(valores: pd.Series) -> pd.Series:
    """Una categoría se compara como texto. Las binarias llegan como 0/1, 1.0 o True: se
    llevan al mismo texto para que la referencia y producción hablen igual."""
    def uno(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return FALTANTE
        if isinstance(v, bool | np.bool_):
            return str(int(v))
        if isinstance(v, int | float | np.integer | np.floating) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return valores.map(uno)


def es_categorica(serie: pd.Series) -> bool:
    """Texto, o un número con pocos valores (0/1): no tiene sentido partirlo en deciles."""
    return not pd.api.types.is_numeric_dtype(serie) or serie.nunique() <= 10


def aprender_feature(nombre: str, valores: pd.Series, pesos: pd.Series) -> Feature:
    w = pesos.to_numpy(dtype=float)
    if es_categorica(valores):
        cats = normalizar_categoria(valores)
        p = pd.Series(w, index=cats.index).groupby(cats).sum()
        p = p / p.sum()
        return Feature(nombre, "categorica", list(p.index), p.to_numpy())
    x = pd.to_numeric(valores).to_numpy(dtype=float)
    cortes = np.unique(cuantiles_ponderados(x, w, DECILES))
    bins = ([f"≤ {cortes[0]:g}"]
            + [f"({a:g}, {b:g}]" for a, b in zip(cortes[:-1], cortes[1:], strict=True)]
            + [f"> {cortes[-1]:g}"])
    f = Feature(nombre, "numerica", bins, np.array([]), cortes)
    etiquetas = f.asignar(valores)
    p = pd.Series(w, index=etiquetas.index).groupby(etiquetas).sum().reindex(bins,
                                                                             fill_value=0)
    f.p_ref = (p / p.sum()).to_numpy()
    return f


def referencia(proc: Path = datos.PROC, particion: str = "train") -> dict[str, list[Feature]]:
    """Por segmento, cómo se distribuye cada feature EN LA POBLACIÓN con la que se
    entrenó: la partición de entrenamiento, ponderada por el factor de expansión."""
    df = pd.read_parquet(proc / datos.dataset(datos.EDICION))
    split = pd.read_csv(proc / datos.SPLIT, dtype={"upm": str})
    df = df.merge(split, on="upm", how="left", validate="many_to_one")
    df = df[df["particion"] == particion]
    salida = {}
    for seg, etiqueta in SEGMENTOS.items():
        feats = json.loads((proc / f"features_{seg}.json").read_text(encoding="utf-8"))
        sub = df[df["segmento"] == etiqueta]
        salida[seg] = [aprender_feature(f, sub[f], sub["factor"]) for f in feats["features"]]
    return salida


# ===================================================================== comparación
@dataclass
class Resultado:
    segmento: str
    feature: str
    tipo: str
    psi: float
    n_actual: int
    n_minimo: int
    nivel: str
    detalle: list[dict] = field(default_factory=list)   # por bin: p_ref, p_actual, aporte


def comparar(feature: Feature, valores: pd.Series, segmento: str) -> Resultado:
    etiquetas = feature.asignar(valores)
    conteo = etiquetas.value_counts()
    # Bins que sólo aparecen en producción (una categoría nueva, faltantes) también
    # cuentan: con referencia ≈ 0, aportan mucho PSI, y deben.
    bins = feature.bins + [b for b in conteo.index if b not in feature.bins]
    p_act = (conteo.reindex(bins, fill_value=0) / max(len(valores), 1)).to_numpy()
    p_ref = np.concatenate([feature.p_ref, np.zeros(len(bins) - len(feature.bins))])
    valor = psi(p_ref, p_act) if len(valores) else float("nan")
    q, p = np.clip(p_ref, EPS, None), np.clip(p_act, EPS, None)
    aportes = (p - q) * np.log(p / q)
    detalle = [{"bin": b, "p_ref": round(float(r), 4), "p_actual": round(float(a), 4),
                "aporte": round(float(c), 4)}
               for b, r, a, c in zip(bins, p_ref, p_act, aportes, strict=True)]
    minimo = n_minimo(len(feature.bins))
    return Resultado(segmento, feature.nombre, feature.tipo, round(valor, 4), len(valores),
                     minimo, nivel(valor, len(valores), minimo, len(feature.bins)), detalle)


def calcular(ref: dict[str, list[Feature]], actual: pd.DataFrame) -> list[Resultado]:
    """`actual`: una fila por predicción, con `segmento` y las features como columnas."""
    resultados = []
    for seg, features in ref.items():
        sub = actual[actual["segmento"] == seg] if len(actual) else actual
        for f in features:
            valores = sub[f.nombre] if f.nombre in sub else pd.Series([None] * len(sub))
            resultados.append(comparar(f, valores, seg))
    return resultados


# ===================================================================== base de datos
SQL_LEER = """
    SELECT segmento, features
    FROM predicciones
    WHERE ts >= %s AND ts < %s
    ORDER BY random()          -- con mucho tráfico basta una muestra: el PSI no mejora
    LIMIT %s                   -- con millones de filas, y la memoria del job sí sufre
"""

SQL_GUARDAR = """
    INSERT INTO drift (corrida, calculado, ventana_inicio, ventana_fin, segmento, feature,
                       tipo, psi, n_actual, n_minimo, nivel, detalle)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (corrida, segmento, feature) DO UPDATE SET
        calculado = EXCLUDED.calculado, psi = EXCLUDED.psi, n_actual = EXCLUDED.n_actual,
        n_minimo = EXCLUDED.n_minimo, nivel = EXCLUDED.nivel, detalle = EXCLUDED.detalle
"""


def leer_predicciones(con, desde: datetime, hasta: datetime, maximo: int) -> pd.DataFrame:
    filas = con.execute(SQL_LEER, (desde, hasta, maximo)).fetchall()
    if not filas:
        return pd.DataFrame(columns=["segmento"])
    return pd.DataFrame([{"segmento": s, **f} for s, f in filas])


def guardar(con, corrida: str, desde: datetime, hasta: datetime,
            resultados: list[Resultado]) -> None:
    """Idempotente: si Airflow reintenta la MISMA corrida, se sobrescribe, no se duplica."""
    from psycopg.types.json import Jsonb

    ahora = datetime.now(UTC)
    with con.transaction(), con.cursor() as cur:
        cur.executemany(SQL_GUARDAR, [
            (corrida, ahora, desde, hasta, r.segmento, r.feature, r.tipo,
             None if np.isnan(r.psi) else r.psi, r.n_actual, r.n_minimo, r.nivel,
             Jsonb(r.detalle)) for r in resultados])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PSI de las predicciones recientes contra la "
                                             "población de entrenamiento")
    ap.add_argument("--horas", type=float, default=24, help="ventana hacia atrás (h)")
    ap.add_argument("--maximo", type=int, default=100_000, help="muestra máxima de filas")
    a = ap.parse_args(argv)

    import psycopg

    dsn = os.environ["MONITOR_DSN"]           # sin default: sin base no hay nada que medir
    corrida = os.getenv("ORQUESTADOR_RUN_ID") or f"manual-{uuid.uuid4().hex[:8]}"
    hasta = datetime.now(UTC)
    desde = hasta - timedelta(hours=a.horas)

    ref = referencia()
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10,
                         application_name="monitor-drift") as con:
        actual = leer_predicciones(con, desde, hasta, a.maximo)
        resultados = calcular(ref, actual)
        guardar(con, corrida, desde, hasta, resultados)

    print(f"{len(actual):,} predicciones entre {desde:%Y-%m-%d %H:%M} y {hasta:%H:%M} UTC")
    if actual.empty:
        print("  SIN PREDICCIONES en la ventana: no hay nada que comparar. ¿Llegó el tráfico "
              "a la API? ¿La ventana (--horas) cubre el momento en que se envió?")
    for r in sorted(resultados, key=lambda r: (r.segmento, -np.nan_to_num(r.psi))):
        print(f"  {r.segmento:<12} {r.feature:<22} PSI {r.psi:6.3f}  n={r.n_actual:<6} "
              f"{r.nivel}")
    # Última línea = JSON chico: Airflow la guarda como XCom (visible en la UI).
    peor = max(resultados, key=lambda r: np.nan_to_num(r.psi, nan=-1))
    print(json.dumps({"corrida": corrida, "predicciones": len(actual),
                      "peor": {k: v for k, v in asdict(peor).items() if k != "detalle"},
                      "actuar": sum(r.nivel == "actuar" for r in resultados)},
                     ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
