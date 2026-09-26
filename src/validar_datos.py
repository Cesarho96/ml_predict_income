"""Valida los datos procesados ANTES de entrenar. Primera tarea del DAG `reentrenar`.

    python -m src.validar_datos          # exit 0 = datos aptos; exit 1 = no se entrena

Por qué existe. `src/train.py` confía en que el parquet trae lo que el notebook 02 dejó:
columnas, tipos, rangos. Cuando INEGI publique la ENIGH 2026 y alguien regenere el
dataset, cualquier cambio silencioso —una columna renombrada, un ingreso trimestral que
nadie dividió entre 3, estados sin cero a la izquierda— produciría un modelo que entrena
"bien", pasa las métricas y está mal. Validar antes es la diferencia entre un DAG que
falla en rojo en el paso 1 y un modelo roto que llega al registry.

Qué valida, en tres niveles:
  1. Esquema por columna (pandera): tipo, nulos, rangos, valores permitidos.
  2. Reglas del dominio sobre el conjunto: tamaño, población expandida, mediana del
     ingreso — lo que un analista de la ENIGH vería "raro" en cinco segundos.
  3. Contratos entre archivos: las features congeladas en el notebook 03 existen, y cada
     UPM tiene partición en split_upm_3way.csv.

No valida que las categorías sean las mismas de 2024: un reentrenamiento con categorías
nuevas es legítimo, y el contrato de la API se deriva del entrenamiento (Hito 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
DATASET = "enigh2024_features_v2.parquet"

SEGMENTOS = ("Ocupados", "No ocupados")
ENTIDADES = [f"{i:02d}" for i in range(1, 33)]

# --- Umbrales del dominio, con su porqué ------------------------------------------------
# ENIGH 2024: 179,557 adultos con ingreso. Una ENIGH nueva tendrá un tamaño parecido; un
# archivo con la mitad de filas es un archivo truncado, no una muestra distinta.
MIN_FILAS = 100_000
# Suma de `factor` = población adulta con ingreso que representa la muestra: 77.1 millones
# en 2024. Fuera de 60–100 M los factores de expansión están mal o faltan hogares.
POBLACION_MILLONES = (60, 100)
# Mediana ponderada del ingreso mensual: $7,908 en 2024. El rango admite que se duplique
# (inflación + aumentos al salario mínimo en dos años), pero atrapa el error clásico:
# ingreso TRIMESTRAL sin dividir entre 3 (~$24,000) o dividido dos veces (~$2,600).
MEDIANA_MENSUAL = (4_000, 16_000)


# Enteros de cualquier ancho. Pandera con `int` exige int64 exacto, y el notebook 02 guarda
# las binarias como int8 (8 veces menos memoria). El ancho es un detalle de
# almacenamiento; lo que importa es que sea entero. (Lo descubrió la primera corrida.)
ENTERO = Check(lambda s: pd.api.types.is_integer_dtype(s), error="debe ser de tipo entero")


def _no_nulos_en(segmento: str, columna: str) -> Check:
    """En su segmento la columna es obligatoria, aunque en el otro sea NaN."""
    return Check(lambda df: df.loc[df["segmento"] == segmento, columna].notna().all(),
                 error=f"'{columna}' tiene nulos en el segmento {segmento}")


ESQUEMA = DataFrameSchema(
    {
        # --- identificación, peso y objetivo
        "upm": Column(str, Check.str_matches(r"^\d{7}$"), nullable=False),
        "factor": Column(None, [ENTERO, Check.ge(1)], nullable=False),
        "ingreso_mensual": Column(float, Check.gt(0), nullable=False),
        "segmento": Column(str, Check.isin(SEGMENTOS), nullable=False),
        # --- comunes
        "cve_entidad": Column(str, Check.isin(ENTIDADES), nullable=False),
        "anios_escolaridad": Column(float, Check.in_range(0, 24), nullable=False),
        "es_jefe_hogar": Column(None, [ENTERO, Check.isin([0, 1])], nullable=False),
        # --- ocupados
        "horas_trabajadas": Column(float, Check.in_range(1, 168), nullable=True),
        "formalidad": Column(str, nullable=False),
        "sinco_grupo": Column(str, nullable=False),
        "posicion_ocupacion": Column(str, nullable=False),
        "sector_agrupado": Column(str, nullable=False),
        "tam_empresa_grupo": Column(str, nullable=False),
        "es_mujer": Column(None, [ENTERO, Check.isin([0, 1])], nullable=False),
        # --- no ocupados
        "edad": Column(None, [ENTERO, Check.in_range(18, 120)], nullable=False),
        "integrantes_hogar": Column(None, [ENTERO, Check.in_range(1, 40)], nullable=False),
        "situacion_conyugal": Column(str, nullable=False),
        "actividad_no_ocupado": Column(str, nullable=True),
    },
    checks=[
        _no_nulos_en("Ocupados", "horas_trabajadas"),
        _no_nulos_en("No ocupados", "actividad_no_ocupado"),
        Check(lambda df: len(df) >= MIN_FILAS,
              error=f"menos de {MIN_FILAS:,} filas: ¿archivo truncado?"),
        Check(lambda df: df["segmento"].value_counts().reindex(SEGMENTOS).fillna(0).min() > 0,
              error="falta uno de los dos segmentos"),
    ],
    coerce=False,     # validar lo que HAY, no convertirlo hasta que pase
    strict=False,     # columnas extra (las 76 del parquet) no son un error
)


def mediana_ponderada(v: pd.Series, w: pd.Series) -> float:
    o = np.argsort(v.to_numpy())
    vv, ww = v.to_numpy()[o], w.to_numpy()[o]
    return float(np.interp(0.5, np.cumsum(ww) / ww.sum(), vv))


def reglas_de_dominio(df: pd.DataFrame) -> list[str]:
    """Reglas sobre el conjunto que un esquema por columna no puede expresar."""
    errores = []
    poblacion = df["factor"].sum() / 1e6
    if not POBLACION_MILLONES[0] <= poblacion <= POBLACION_MILLONES[1]:
        errores.append(f"población expandida {poblacion:.1f} M fuera de {POBLACION_MILLONES} M")
    mediana = mediana_ponderada(df["ingreso_mensual"], df["factor"])
    if not MEDIANA_MENSUAL[0] <= mediana <= MEDIANA_MENSUAL[1]:
        errores.append(f"mediana ponderada ${mediana:,.0f} fuera de {MEDIANA_MENSUAL}: "
                       "¿ingreso trimestral sin dividir entre 3?")
    return errores


def contratos_entre_archivos(df: pd.DataFrame, proc: Path) -> list[str]:
    errores = []
    for archivo in ("features_ocupados.json", "features_no_ocupados.json"):
        feats = json.loads((proc / archivo).read_text(encoding="utf-8"))["features"]
        if faltan := sorted(set(feats) - set(df.columns)):
            errores.append(f"{archivo} pide columnas que el dataset no tiene: {faltan}")
    split = pd.read_csv(proc / "split_upm_3way.csv", dtype={"upm": str})
    if sin_particion := len(set(df["upm"]) - set(split["upm"])):
        errores.append(f"{sin_particion} UPM del dataset no tienen partición en split_upm_3way.csv")
    if faltan := {"train", "validation", "test"} - set(split["particion"]):
        errores.append(f"split_upm_3way.csv no tiene las particiones {sorted(faltan)}")
    return errores


def validar(proc: Path = PROC, dataset: str = DATASET) -> dict:
    """Lanza `DatosInvalidos` con TODOS los problemas encontrados, no sólo el primero."""
    df = pd.read_parquet(proc / dataset)
    errores: list[str] = []
    try:
        ESQUEMA.validate(df, lazy=True)       # lazy: junta todas las fallas
    except pa.errors.SchemaErrors as e:
        casos = e.failure_cases
        for (col, chk), g in casos.groupby(["column", "check"], dropna=False):
            donde = col if isinstance(col, str) else "conjunto"
            ejemplos = [x for x in g["failure_case"].dropna().astype(str).unique()[:3]
                        if x not in ("False", "True")]
            detalle = f" · {len(g)} casos · ej. {ejemplos}" if ejemplos else ""
            errores.append(f"[esquema] {donde} · {chk}{detalle}")
    errores += [f"[dominio] {x}" for x in reglas_de_dominio(df)]
    errores += [f"[contrato] {x}" for x in contratos_entre_archivos(df, proc)]
    if errores:
        raise DatosInvalidos(errores)
    return {"filas": len(df),
            "poblacion_millones": round(df["factor"].sum() / 1e6, 1),
            "mediana_mensual": round(mediana_ponderada(df["ingreso_mensual"], df["factor"])),
            "por_segmento": df["segmento"].value_counts().to_dict()}


class DatosInvalidos(Exception):
    def __init__(self, errores: list[str]):
        self.errores = errores
        super().__init__(f"{len(errores)} problema(s) en los datos")


def main() -> None:
    try:
        resumen = validar()
    except DatosInvalidos as e:
        print(f"DATOS INVÁLIDOS — no se entrena. {len(e.errores)} problema(s):")
        for x in e.errores:
            print(f"  ✗ {x}")
        sys.exit(1)
    print("datos válidos:", json.dumps(resumen, ensure_ascii=False))


if __name__ == "__main__":
    main()
