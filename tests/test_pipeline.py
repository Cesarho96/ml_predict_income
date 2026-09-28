"""El pipeline crudo → dataset, sobre una ENIGH en miniatura escrita a mano.

Cuatro personas con casos elegidos, en el mismo formato de carpetas y columnas que publica
INEGI. Corre en CI sin microdatos reales. La paridad con los notebooks sobre la ENIGH 2024
completa está en tests/test_paridad.py (sólo corre donde están los datos crudos).
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import datos
from src.pipeline.construir import construir
from src.pipeline.features import construir_features, features_del_modelo
from src.pipeline.ingesta import construir_personas
from src.pipeline.particion import extender, particion_de

EDICION = 2024

# A: adulto que trabaja, con IMSS  · B: adulta pensionada  · C: menor con ingreso (fuera)
# D: adulto sin ingreso (fuera)
POBLACION = [
    dict(folioviv="0100001", foliohog="1", numren="01", parentesco="101", sexo="1", edad="40",
         nivelaprob="07", gradoaprob="4", antec_esc="", edo_conyug="2", segsoc="1",
         act_pnea1="", est_dis="001", upm="0000001", factor="120"),
    dict(folioviv="0100001", foliohog="1", numren="02", parentesco="201", sexo="2", edad="67",
         nivelaprob="02", gradoaprob="6", antec_esc="", edo_conyug="2", segsoc="2",
         act_pnea1="2", est_dis="001", upm="0000001", factor="120"),
    dict(folioviv="0100001", foliohog="1", numren="03", parentesco="301", sexo="1", edad="15",
         nivelaprob="03", gradoaprob="2", antec_esc="", edo_conyug="8", segsoc="",
         act_pnea1="4", est_dis="001", upm="0000001", factor="120"),
    dict(folioviv="0200002", foliohog="1", numren="01", parentesco="101", sexo="2", edad="30",
         nivelaprob="04", gradoaprob="3", antec_esc="", edo_conyug="8", segsoc="2",
         act_pnea1="3", est_dis="002", upm="0000002", factor="95"),
]
INGRESOS = [
    dict(folioviv="0100001", foliohog="1", numren="01", clave="P001", ing_tri="30000"),
    dict(folioviv="0100001", foliohog="1", numren="01", clave="P050", ing_tri="99999"),  # retiro
    dict(folioviv="0100001", foliohog="1", numren="02", clave="P032", ing_tri="9000"),
    dict(folioviv="0100001", foliohog="1", numren="03", clave="P001", ing_tri="3000"),
]
TRABAJOS = [
    dict(folioviv="0100001", foliohog="1", numren="01", id_trabajo="1", subor="1", indep="",
         personal="", pago="1", htrab="45", sinco="2111", scian="5411", tam_emp="8",
         pres_1="1", medtrab_1="1"),
]
CONCENTRADO = [dict(folioviv="0100001", foliohog="1", tot_integ="3"),
               dict(folioviv="0200002", foliohog="1", tot_integ="1")]
VIVIENDAS = [dict(folioviv="0100001", ubica_geo="09010"), dict(folioviv="0200002", ubica_geo="15033")]

COLUMNAS_TRABAJOS = (["folioviv", "foliohog", "numren", "id_trabajo", "subor", "indep",
                      "personal", "pago", "htrab", "sinco", "scian", "tam_emp"]
                     + [f"pres_{i}" for i in range(1, 20)] + [f"medtrab_{i}" for i in range(1, 6)])


def _escribir(crudos: Path, tabla: str, filas: list[dict], columnas=None):
    base = (crudos / f"conjunto_de_datos_enigh{EDICION}_ns_csv"
            / f"conjunto_de_datos_{tabla}_enigh{EDICION}_ns" / "conjunto_de_datos")
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas, columns=columnas).fillna("").to_csv(
        base / f"conjunto_de_datos_{tabla}_enigh{EDICION}_ns.csv", index=False)


@pytest.fixture
def crudos(tmp_path):
    c = tmp_path / "raw"
    _escribir(c, "poblacion", POBLACION)
    _escribir(c, "ingresos", INGRESOS)
    _escribir(c, "trabajos", TRABAJOS, COLUMNAS_TRABAJOS)
    _escribir(c, "concentradohogar", CONCENTRADO)
    _escribir(c, "viviendas", VIVIENDAS)
    return c


def _persona(df, numren):
    return df[(df.folioviv == "0100001") & (df.numren == numren)].iloc[0]


# ------------------------------------------------------------------ ingesta
def test_universo_e_ingreso(crudos):
    p = construir_personas(crudos, EDICION)
    # Fuera: el menor (C) y el adulto sin ingreso (D).
    assert sorted(p["numren"]) == ["01", "02"]
    a, b = _persona(p, "01"), _persona(p, "02")
    assert a.ingreso_mensual == 10_000.0      # P001 / 3; el retiro P050 NO cuenta
    assert b.ingreso_mensual == 3_000.0       # transferencia P032 / 3


def test_atributos_de_persona_y_trabajo(crudos):
    p = construir_personas(crudos, EDICION)
    a, b = _persona(p, "01"), _persona(p, "02")
    assert (a.es_jefe_hogar, a.es_mujer, a.anios_escolaridad) == (1, 0, 16)   # licenciatura 4
    assert (a.tiene_trabajo_principal, a.horas_trabajadas) == (1, 45)
    assert a.sinco_grupo == "2 Profesionistas y técnicos"
    assert a.posicion_ocupacion == "Subordinado remunerado"
    assert (a.cve_entidad, a.integrantes_hogar) == ("09", 3)
    assert (b.tiene_trabajo_principal, b.posicion_ocupacion) == (0, "NO_APLICA")
    assert b.actividad_no_ocupado == "Pensionado(a)/jubilado(a)"


def test_una_clave_de_ingreso_nueva_detiene_todo(crudos):
    """El riesgo #1 de una edición nueva: INEGI agrega una clave que nadie clasificó."""
    _escribir(crudos, "ingresos", INGRESOS + [dict(folioviv="0100001", foliohog="1",
                                                    numren="01", clave="P999", ing_tri="1")])
    with pytest.raises(ValueError, match="sin clasificar.*P999"):
        construir_personas(crudos, EDICION)


def test_un_numero_ilegible_no_se_vuelve_nan_en_silencio(crudos):
    filas = [dict(f) for f in POBLACION]
    filas[0]["factor"] = "12O"                # letra O en vez de cero
    _escribir(crudos, "poblacion", filas)
    with pytest.raises(ValueError, match="factor.*no numéricos"):
        construir_personas(crudos, EDICION)


def test_edicion_sin_datos_dice_que_falta(crudos):
    with pytest.raises(FileNotFoundError, match="2026"):
        construir_personas(crudos, 2026)


# ------------------------------------------------------------------ features
def test_features_derivadas(crudos):
    df = construir_features(construir_personas(crudos, EDICION),
                            features_del_modelo(datos.RAIZ / "data" / "processed"))
    a = df[df.numren == "01"].iloc[0]
    b = df[df.numren == "02"].iloc[0]
    assert a.segmento == "Ocupados" and b.segmento == "No ocupados"
    assert a.formalidad == "3 Formal (con seguridad social)"
    assert a.tam_empresa_grupo == "3 Mediana (51-250)"          # código 8
    assert a.sector_agrupado == "6 Servicios profesionales y financieros"   # SCIAN 54
    assert (b.formalidad, b.sector_agrupado, b.tam_empresa_grupo) == (
        "0 No trabaja", "0 No trabaja", "0 No aplica / no sabe")


def test_una_feature_que_el_pipeline_no_calcula_es_error(crudos):
    with pytest.raises(ValueError, match="no calcula.*inventada"):
        construir_features(construir_personas(crudos, EDICION), ["edad", "inventada"])


# ------------------------------------------------------------------ partición
def test_particion_conserva_lo_existente_y_es_estable():
    existente = pd.DataFrame({"upm": ["0000001", "0000002"], "particion": ["test", "train"]})
    completa, n = extender(["0000001", "0000002", "0000003"], existente)
    assert n == 1
    assert dict(zip(completa.upm, completa.particion, strict=True))["0000001"] == "test"
    assert particion_de("0000003") == particion_de("0000003")     # sin semilla, sin estado


def test_particion_nueva_respeta_proporciones():
    nuevas = [f"{i:07d}" for i in range(20_000)]
    reparto = pd.Series([particion_de(u) for u in nuevas]).value_counts(normalize=True)
    assert abs(reparto["train"] - 0.64) < 0.02
    assert abs(reparto["validation"] - 0.16) < 0.02
    assert abs(reparto["test"] - 0.20) < 0.02


# ------------------------------------------------------------------ de punta a punta
def test_construir_escribe_dataset_metadatos_y_particion(crudos, tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    for f in ("features_ocupados.json", "features_no_ocupados.json"):
        shutil.copy(datos.RAIZ / "data" / "processed" / f, proc / f)
    pd.DataFrame({"upm": ["0000001"], "particion": ["train"]}).to_csv(
        proc / datos.SPLIT, index=False)

    meta = construir(EDICION, crudos=crudos, proc=proc)

    assert meta["filas"] == 2 and meta["upm_nuevas_particionadas"] == 0  # D no entra
    df = pd.read_parquet(proc / datos.dataset(EDICION))
    assert len(df) == 2 and not list(proc.glob("*.tmp"))
    guardada = json.loads((proc / "enigh2024_features_v2.meta.json").read_text())
    assert guardada["sha256"] == meta["sha256"]
