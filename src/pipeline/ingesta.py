"""Tablas crudas de la ENIGH → una fila por persona adulta con ingreso monetario > 0.

Port del notebook 01, recortado a las columnas que el modelo usa. Cada regla conserva el
razonamiento del notebook; donde el notebook sólo exploraba, aquí no hay código.

Universo: personas de 18+ con ingreso monetario mensual > 0 (decisión del dataset,
ver claude/dataset-enigh2024-decisiones.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.enigh import ID_HOGAR, ID_PERSONA, ID_VIVIENDA, leer, num

EDAD_MINIMA = 18

# --------------------------------------------------------------------- ingreso
# Claves del catálogo ingresos_cat. P050–P066 son percepciones financieras y de capital
# (retiros de ahorro, préstamos, venta de bienes): NO son ingreso corriente.
GRUPOS_CLAVE = {
    "sueldos_salarios": list(range(1, 23)),                          # P001–P022
    "negocios": list(range(68, 82)),                                 # P068–P081
    "trabajo_menores": [67],                                         # P067
    "rentas": list(range(23, 32)),                                   # P023–P031
    "transferencias": list(range(32, 49)) + list(range(101, 109)),   # P032–P048, P101–P108
    "otros": [49],                                                   # P049
}
CLAVES_NO_INGRESO = list(range(50, 67))
CLAVE_A_GRUPO = {c: g for g, claves in GRUPOS_CLAVE.items() for c in claves}


def ingreso_mensual(crudos: Path, edicion: int) -> pd.DataFrame:
    """Ingreso monetario mensual por persona = Σ ing_tri de las claves de ingreso / 3."""
    ing = leer(crudos, edicion, "ingresos", ID_PERSONA + ["clave", "ing_tri"])
    ing["ing_tri"] = num(ing["ing_tri"]).fillna(0.0)
    ing["clave_num"] = ing["clave"].str.extract(r"P(\d+)")[0].astype(int)
    ing["grupo"] = ing["clave_num"].map(CLAVE_A_GRUPO)

    # La amenaza principal de una edición nueva: una clave que INEGI agregó y nadie
    # clasificó. Sin esto, su ingreso desaparecería del total sin avisar.
    sin_clasificar = ing.loc[ing["grupo"].isna() & ~ing["clave_num"].isin(CLAVES_NO_INGRESO),
                             "clave"].unique()
    if len(sin_clasificar):
        raise ValueError(f"claves de ingreso sin clasificar en {edicion}: {sorted(sin_clasificar)}")

    t = (ing.dropna(subset=["grupo"])
         .pivot_table(index=ID_PERSONA, columns="grupo", values="ing_tri", aggfunc="sum")
         .fillna(0.0))
    t.columns.name = None
    t = t.reindex(columns=list(GRUPOS_CLAVE), fill_value=0.0)
    t = (t / 3).round(2)                              # trimestral → mensual
    return t.sum(axis=1).round(2).rename("ingreso_mensual").reset_index()


# --------------------------------------------------------------------- personas
# Años de escolaridad (convención CONEVAL): años acumulados antes de iniciar cada nivel.
BASE_NIVEL = {"00": 0, "01": 0, "02": 0, "03": 6, "04": 9, "07": 12, "08": 16, "09": 16,
              "10": 18}
# Normal (05) y técnica (06) dependen del antecedente escolar.
BASE_ANTECEDENTE = {"1": 6, "2": 9, "3": 12, "4": 16, "5": 18}

SITUACION_CONYUGAL = {
    "1": "Unión libre", "2": "Casado(a)", "3": "Casado(a)", "4": "Casado(a)",
    "5": "Separado(a)", "6": "Divorciado(a)", "7": "Viudo(a)", "8": "Soltero(a)",
}
ACT_PNEA = {
    "1": "Buscó trabajo", "2": "Pensionado(a)/jubilado(a)", "3": "Quehaceres del hogar",
    "4": "Estudia", "5": "Limitación permanente", "6": "Otra situación",
}


def anios_escolaridad(nivel, grado, antecedente):
    if pd.isna(nivel):
        return np.nan
    g = 0 if pd.isna(grado) else int(grado)
    if nivel in ("00", "01"):
        return 0
    if nivel in BASE_NIVEL:
        return min(BASE_NIVEL[nivel] + g, 24)
    if nivel in ("05", "06"):
        return min(BASE_ANTECEDENTE.get(antecedente, 9) + g, 24)
    return np.nan


def personas(crudos: Path, edicion: int) -> pd.DataFrame:
    pob = leer(crudos, edicion, "poblacion", ID_PERSONA + [
        "parentesco", "sexo", "edad", "nivelaprob", "gradoaprob", "antec_esc",
        "edo_conyug", "segsoc", "act_pnea1", "est_dis", "upm", "factor"])
    if pob.duplicated(ID_PERSONA).any():
        raise ValueError("hay personas duplicadas en la tabla poblacion")

    p = pd.DataFrame(index=pob.index)
    p[ID_PERSONA] = pob[ID_PERSONA]
    p["est_dis"] = pob["est_dis"]
    p["upm"] = pob["upm"]
    p["factor"] = num(pob["factor"])
    p["edad"] = num(pob["edad"])
    p["es_mujer"] = (pob["sexo"] == "2").astype("int8")
    p["es_jefe_hogar"] = (pob["parentesco"] == "101").astype("int8")
    p["situacion_conyugal"] = pob["edo_conyug"].map(SITUACION_CONYUGAL)
    p["anios_escolaridad"] = [
        anios_escolaridad(n, g, a) for n, g, a in
        zip(pob["nivelaprob"], pob["gradoaprob"], pob["antec_esc"], strict=True)]
    p["actividad_no_ocupado"] = pob["act_pnea1"].map(ACT_PNEA)
    p["cotiza_seguridad_social"] = pob["segsoc"].map({"1": 1, "2": 0}).astype("Int8")
    return p


# --------------------------------------------------------------------- empleo
SINCO_GRUPO = {
    "0": "0 Casos especiales",
    "1": "1 Funcionarios, directores y jefes",
    "2": "2 Profesionistas y técnicos",
    "3": "3 Auxiliares en actividades administrativas",
    "4": "4 Comerciantes, empleados en ventas y agentes",
    "5": "5 Servicios personales y vigilancia",
    "6": "6 Actividades agrícolas, ganaderas, forestales, caza y pesca",
    "7": "7 Artesanales, construcción y otros oficios",
    "8": "8 Operadores de maquinaria, ensambladores y conductores",
    "9": "9 Actividades elementales y de apoyo",
}
SCIAN_SECTOR = {
    "11": "11 Agricultura, cría, forestal, pesca", "21": "21 Minería",
    "22": "22 Electricidad, agua y gas", "23": "23 Construcción",
    "31": "31-33 Industrias manufactureras", "32": "31-33 Industrias manufactureras",
    "33": "31-33 Industrias manufactureras",
    "43": "43 Comercio al por mayor", "46": "46 Comercio al por menor",
    "48": "48-49 Transportes y almacenamiento", "49": "48-49 Transportes y almacenamiento",
    "51": "51 Información en medios masivos", "52": "52 Servicios financieros y de seguros",
    "53": "53 Servicios inmobiliarios y de alquiler",
    "54": "54 Servicios profesionales, científicos y técnicos", "55": "55 Corporativos",
    "56": "56 Apoyo a negocios y manejo de residuos", "61": "61 Servicios educativos",
    "62": "62 Servicios de salud y asistencia social", "71": "71 Esparcimiento y recreación",
    "72": "72 Alojamiento temporal y alimentos", "81": "81 Otros servicios excepto gobierno",
    "93": "93 Gobierno y organismos internacionales",
}
PRES_COLS = [f"pres_{i}" for i in range(1, 20)]
MED_PUBLICO = [f"medtrab_{i}" for i in range(1, 6)]  # IMSS, ISSSTE, ISSSTE est., PEMEX, univ.


def posicion(subor, indep, personal, pago):
    if pago in ("2", "3"):
        return "Trabajador sin pago"
    if subor == "1":
        return "Subordinado remunerado"
    if indep == "1":
        return "Empleador" if personal == "1" else "Trabajador por cuenta propia"
    return np.nan


def empleo(crudos: Path, edicion: int) -> pd.DataFrame:
    """Atributos del trabajo PRINCIPAL (id_trabajo == 1)."""
    trab = leer(crudos, edicion, "trabajos", ID_PERSONA + [
        "id_trabajo", "subor", "indep", "personal", "pago", "htrab", "sinco", "scian",
        "tam_emp"] + PRES_COLS + MED_PUBLICO)
    pr = trab[trab["id_trabajo"] == "1"].copy()
    if pr.duplicated(ID_PERSONA).any():
        raise ValueError("hay personas con más de un trabajo principal")

    e = pd.DataFrame(index=pr.index)
    e[ID_PERSONA] = pr[ID_PERSONA]
    e["tiene_trabajo_principal"] = np.int8(1)
    e["posicion_ocupacion"] = pd.Series(
        [posicion(s, i, p, g) for s, i, p, g in
         zip(pr["subor"], pr["indep"], pr["personal"], pr["pago"], strict=True)],
        index=pr.index).fillna("No especificado")
    e["sinco_grupo"] = pr["sinco"].str[0].map(SINCO_GRUPO)
    e["scian_sector"] = pr["scian"].str[:2].map(SCIAN_SECTOR).fillna("99 No especificado")
    horas = num(pr["htrab"])
    e["horas_trabajadas"] = horas.where(horas.between(1, 168))
    e["tam_empresa"] = num(pr["tam_emp"]).where(lambda s: s <= 11)   # 12 = "no sabe"
    e["n_prestaciones"] = pr[PRES_COLS].notna().sum(axis=1).astype("int8")
    e["empleo_formal"] = pr[MED_PUBLICO].notna().any(axis=1).astype("int8")
    return e


# --------------------------------------------------------------------- hogar y vivienda
def hogar(crudos: Path, edicion: int) -> pd.DataFrame:
    con = leer(crudos, edicion, "concentradohogar", ID_HOGAR + ["tot_integ"])
    h = con[ID_HOGAR].copy()
    h["integrantes_hogar"] = num(con["tot_integ"])
    return h


def geografia(crudos: Path, edicion: int) -> pd.DataFrame:
    viv = leer(crudos, edicion, "viviendas", ID_VIVIENDA + ["ubica_geo"])
    g = viv[ID_VIVIENDA].copy()
    g["cve_entidad"] = viv["ubica_geo"].str[:2]
    return g


# --------------------------------------------------------------------- ensamblado
def construir_personas(crudos: Path, edicion: int) -> pd.DataFrame:
    """Una fila por persona adulta con ingreso monetario mensual > 0."""
    per = personas(crudos, edicion)
    df = (per
          .merge(ingreso_mensual(crudos, edicion), on=ID_PERSONA, how="left")
          .merge(empleo(crudos, edicion), on=ID_PERSONA, how="left")
          .merge(hogar(crudos, edicion), on=ID_HOGAR, how="left")
          .merge(geografia(crudos, edicion), on=ID_VIVIENDA, how="left"))
    if len(df) != len(per):
        raise ValueError("un join cambió el número de personas: llaves duplicadas")

    # Sin registro en `ingresos` = sin ingreso monetario (no es un faltante).
    df["ingreso_mensual"] = df["ingreso_mensual"].fillna(0.0)
    # Bloque laboral: NaN estructural (no trabaja) → categoría explícita.
    df["tiene_trabajo_principal"] = df["tiene_trabajo_principal"].fillna(0).astype("int8")
    for col in ("posicion_ocupacion", "sinco_grupo", "scian_sector"):
        df[col] = df[col].fillna("NO_APLICA")
    # El left join vuelve float a los enteros del bloque laboral; se restauran nulables.
    for col in ("n_prestaciones", "empleo_formal"):
        df[col] = df[col].astype("Int8")

    universo = (df["edad"] >= EDAD_MINIMA) & (df["ingreso_mensual"] > 0)
    return df.loc[universo].reset_index(drop=True)
