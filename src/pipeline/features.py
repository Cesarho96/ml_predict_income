"""Persona → las variables que usa el modelo. Port del notebook 02 (§ ingeniería).

Sólo las 4 derivadas que sobrevivieron a la selección del notebook 03. Las demás que el
notebook construyó (jornada, experiencia potencial, razón de dependencia) se evaluaron y
no entraron: no se calculan aquí.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

IDS = ["folioviv", "foliohog", "numren"]
DISENIO = ["factor", "upm", "est_dis"]
TARGET = "ingreso_mensual"

SECTOR_8 = {
    "11 Agricultura, cría, forestal, pesca": "1 Agropecuario",
    "21 Minería": "2 Industria", "22 Electricidad, agua y gas": "2 Industria",
    "31-33 Industrias manufactureras": "2 Industria",
    "23 Construcción": "3 Construcción",
    "43 Comercio al por mayor": "4 Comercio", "46 Comercio al por menor": "4 Comercio",
    "48-49 Transportes y almacenamiento": "5 Transporte y logística",
    "51 Información en medios masivos": "6 Servicios profesionales y financieros",
    "52 Servicios financieros y de seguros": "6 Servicios profesionales y financieros",
    "53 Servicios inmobiliarios y de alquiler": "6 Servicios profesionales y financieros",
    "54 Servicios profesionales, científicos y técnicos":
        "6 Servicios profesionales y financieros",
    "55 Corporativos": "6 Servicios profesionales y financieros",
    "56 Apoyo a negocios y manejo de residuos": "6 Servicios profesionales y financieros",
    "61 Servicios educativos": "7 Educación, salud y gobierno",
    "62 Servicios de salud y asistencia social": "7 Educación, salud y gobierno",
    "93 Gobierno y organismos internacionales": "7 Educación, salud y gobierno",
    "71 Esparcimiento y recreación": "8 Otros servicios",
    "72 Alojamiento temporal y alimentos": "8 Otros servicios",
    "81 Otros servicios excepto gobierno": "8 Otros servicios",
}


def features_del_modelo(proc: Path) -> list[str]:
    """Las features congeladas en el notebook 03 (features_*.json), en orden estable."""
    todas: list[str] = []
    for archivo in ("features_ocupados.json", "features_no_ocupados.json"):
        todas += json.loads((proc / archivo).read_text(encoding="utf-8"))["features"]
    return list(dict.fromkeys(todas))


def construir_features(p: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = p.copy()
    trabaja = df["tiene_trabajo_principal"] == 1

    # El ruteo del producto: una pregunta, "¿trabajaste el mes pasado?".
    df["segmento"] = np.where(trabaja, "Ocupados", "No ocupados")

    # Formalidad en tres niveles que una persona sabe contestar.
    cotiza = df["cotiza_seguridad_social"].fillna(0).astype(float)
    formal = df["empleo_formal"].fillna(0).astype(float)
    prest = df["n_prestaciones"].fillna(0).astype(float)
    df["formalidad"] = np.select(
        [~trabaja, (cotiza == 1) | (formal == 1), prest > 0],
        ["0 No trabaja", "3 Formal (con seguridad social)",
         "2 Informal con algunas prestaciones"],
        default="1 Informal sin prestaciones")

    # Tamaño de empresa: los 11 códigos de INEGI en 4 rangos preguntables.
    df["tam_empresa_grupo"] = pd.cut(
        df["tam_empresa"].astype(float), bins=[0, 2, 7, 9, 11],
        labels=["1 Micro (1-5)", "2 Pequeña (6-50)", "3 Mediana (51-250)",
                "4 Grande (251+)"]).astype(object)
    df["tam_empresa_grupo"] = df["tam_empresa_grupo"].fillna("0 No aplica / no sabe")

    # Sector SCIAN en 8 grupos.
    respaldo = pd.Series(np.where(trabaja, "9 No especificado", "0 No trabaja"),
                         index=df.index)
    df["sector_agrupado"] = df["scian_sector"].map(SECTOR_8).fillna(respaldo)

    if faltan := [f for f in features if f not in df.columns]:
        raise ValueError(f"el modelo pide features que el pipeline no calcula: {faltan}")
    columnas = IDS + DISENIO + [TARGET, "segmento", "tiene_trabajo_principal"] + features
    return df[list(dict.fromkeys(columnas))]
