"""Lectura de tablas de la ENIGH. Todo se lee como TEXTO.

Por qué texto: las claves de la ENIGH tienen ceros a la izquierda que importan
(entidad "09", UPM "0000123", nivel escolar "02"). Si pandas infiere números, "09" se
vuelve 9 y los mapeos dejan de coincidir en silencio. Se convierte a número sólo lo que
es número, y explícitamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_PERSONA = ["folioviv", "foliohog", "numren"]
ID_HOGAR = ["folioviv", "foliohog"]
ID_VIVIENDA = ["folioviv"]

# INEGI: "&" = no especificado; vacío = no aplica. Para el modelo, ambos son faltante.
NA_ENIGH = {"": np.nan, "&": np.nan, "NA": np.nan}


class TablaNoEncontrada(FileNotFoundError):
    pass


def carpeta_edicion(crudos: Path, edicion: int) -> Path:
    return Path(crudos) / f"conjunto_de_datos_enigh{edicion}_ns_csv"


def ruta_tabla(crudos: Path, edicion: int, tabla: str) -> Path:
    """Convención de nombres de INEGI (2018–2024). Si una edición la cambia, falla aquí."""
    base = carpeta_edicion(crudos, edicion) / f"conjunto_de_datos_{tabla}_enigh{edicion}_ns"
    ruta = base / "conjunto_de_datos" / f"conjunto_de_datos_{tabla}_enigh{edicion}_ns.csv"
    if not ruta.exists():
        raise TablaNoEncontrada(
            f"no existe {ruta}. ¿Cambió la convención de nombres de INEGI en {edicion}?")
    return ruta


def leer(crudos: Path, edicion: int, tabla: str, columnas: list[str]) -> pd.DataFrame:
    """Lee sólo `columnas` de una tabla, como texto, sin espacios y con NA normalizados."""
    df = pd.read_csv(ruta_tabla(crudos, edicion, tabla), usecols=columnas, dtype=str,
                     low_memory=False)
    for col in df.columns:
        df[col] = df[col].str.strip().replace(NA_ENIGH)
    return df


def num(serie: pd.Series) -> pd.Series:
    """Texto → número, y FALLA si algún valor presente no es un número.

    El notebook usaba `errors="coerce"` a secas, que convierte en NaN cualquier cosa que
    no parezca número. Con la ENIGH 2024 no pasa nada. Con una edición nueva que traiga
    "12.5a" o "N/D", el valor desaparecería en silencio y el modelo aprendería de un
    faltante que no existe. (Mismo error que invalidó resultados en el notebook 03.)
    """
    res = pd.to_numeric(serie, errors="coerce")
    perdidos = res.isna() & serie.notna()
    if perdidos.any():
        raise ValueError(f"'{serie.name}': {int(perdidos.sum())} valores no numéricos, "
                         f"ej. {serie[perdidos].unique()[:5].tolist()}")
    return res
