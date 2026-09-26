"""La validación de datos atrapa los errores que dice atrapar, y deja pasar datos sanos.

Datos sintéticos con la forma del parquet real: en CI no hay microdatos de la ENIGH.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.validar_datos import DATASET, DatosInvalidos, validar

N = 120_000


@pytest.fixture
def sano(tmp_path):
    rng = np.random.default_rng(0)
    ocupado = rng.random(N) < 0.78
    df = pd.DataFrame({
        "upm": [f"{i % 9000:07d}" for i in range(N)],
        "factor": np.full(N, 640, dtype="int64"),               # ~77 M de población
        "ingreso_mensual": rng.lognormal(np.log(8000), 0.9, N),  # mediana ~$8,000
        "segmento": np.where(ocupado, "Ocupados", "No ocupados"),
        "cve_entidad": rng.choice([f"{i:02d}" for i in range(1, 33)], N),
        "anios_escolaridad": rng.integers(0, 22, N).astype(float),
        "es_jefe_hogar": rng.integers(0, 2, N).astype("int8"),
        "horas_trabajadas": np.where(ocupado, rng.integers(1, 80, N), np.nan),
        "formalidad": "3 Formal (con seguridad social)",
        "sinco_grupo": "2 Profesionistas y técnicos",
        "posicion_ocupacion": "Subordinado remunerado",
        "sector_agrupado": "4 Comercio",
        "tam_empresa_grupo": "1 Micro (1-5)",
        "es_mujer": rng.integers(0, 2, N).astype("int8"),
        "edad": rng.integers(18, 90, N),
        "integrantes_hogar": rng.integers(1, 10, N),
        "situacion_conyugal": "Casado(a)",
        "actividad_no_ocupado": np.where(ocupado, None, "Quehaceres del hogar"),
    })
    for seg, feats in (("ocupados", ["horas_trabajadas", "formalidad"]),
                       ("no_ocupados", ["edad", "actividad_no_ocupado"])):
        (tmp_path / f"features_{seg}.json").write_text(json.dumps({"features": feats}))
    upms = sorted(df["upm"].unique())
    pd.DataFrame({"upm": upms,
                  "particion": (["train", "validation", "test"] * len(upms))[:len(upms)]}
                 ).to_csv(tmp_path / "split_upm_3way.csv", index=False)
    return tmp_path, df


def _con(tmp_path, df):
    df.to_parquet(tmp_path / DATASET)
    return tmp_path


def test_datos_sanos_pasan(sano):
    tmp, df = sano
    r = validar(_con(tmp, df))
    assert r["filas"] == N and 60 <= r["poblacion_millones"] <= 100


@pytest.mark.parametrize("nombre,romper,espera", [
    ("ingreso trimestral sin dividir entre 3",
     lambda d: d.assign(ingreso_mensual=d.ingreso_mensual * 3), "trimestral"),
    ("estados sin cero a la izquierda",
     lambda d: d.assign(cve_entidad=d.cve_entidad.str.lstrip("0")), "cve_entidad"),
    ("horas nulas en ocupados",
     lambda d: d.assign(horas_trabajadas=np.nan), "horas_trabajadas"),
    ("archivo truncado", lambda d: d.head(20_000), "truncado"),
    ("columna renombrada", lambda d: d.rename(columns={"formalidad": "formal"}), "formalidad"),
    ("binaria guardada como float", lambda d: d.assign(es_mujer=d.es_mujer.astype(float)), "entero"),
])
def test_atrapa_errores_reales(sano, nombre, romper, espera):
    tmp, df = sano
    with pytest.raises(DatosInvalidos) as e:
        validar(_con(tmp, romper(df)))
    assert any(espera in x for x in e.value.errores), (nombre, e.value.errores)


def test_upm_sin_particion(sano):
    tmp, df = sano
    df.loc[0, "upm"] = "9999999"
    with pytest.raises(DatosInvalidos, match="problema"):
        validar(_con(tmp, df))
