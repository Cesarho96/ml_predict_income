"""Pruebas de la API. Son el contrato ejecutable: si estas pasan, la imagen sirve."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

OCUPADO = {
    "trabajo_mes_pasado": True,
    "horas_trabajadas": 48, "formalidad": "3 Formal (con seguridad social)",
    "cve_entidad": "09", "anios_escolaridad": 16,
    "sinco_grupo": "2 Profesionistas y técnicos",
    "posicion_ocupacion": "Subordinado remunerado", "es_jefe_hogar": 1,
    "sector_agrupado": "6 Servicios profesionales y financieros",
    "es_mujer": 0, "tam_empresa_grupo": "3 Mediana (51-250)",
}
NO_OCUPADO = {
    "trabajo_mes_pasado": False,
    "actividad_no_ocupado": "Pensionado(a)/jubilado(a)", "es_jefe_hogar": 1,
    "anios_escolaridad": 12, "edad": 70, "cve_entidad": "19",
    "integrantes_hogar": 2, "situacion_conyugal": "Casado(a)",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # el `with` dispara el lifespan y carga los modelos
        yield c


def test_health_no_depende_de_los_modelos(client):
    assert client.get("/health").status_code == 200


def test_ready_reporta_los_dos_modelos_y_su_version(client):
    r = client.get("/ready")
    assert r.status_code == 200
    modelos = r.json()["modelos"]
    assert set(modelos) == {"ocupados", "no_ocupados"}
    assert all("version" in m and "uri" in m for m in modelos.values())


@pytest.mark.parametrize("payload,segmento", [(OCUPADO, "ocupados"),
                                              (NO_OCUPADO, "no_ocupados")])
def test_predict_devuelve_intervalo_ordenado(client, payload, segmento):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["segmento"] == segmento
    assert d["inferior"] < d["mediana"] < d["superior"]
    assert d["inferior"] > 0
    assert d["moneda"] == "MXN"
    # Cada respuesta dice qué modelo la produjo: es lo que hará atribuible el drift.
    assert d["version_modelo"].startswith(f"ingreso_{segmento}/")


def test_categoria_desconocida_es_422_no_500(client):
    """Una categoría inválida es error del cliente. El modelo no debe adivinar."""
    malo = {**OCUPADO, "formalidad": "no existe"}
    assert client.post("/predict", json=malo).status_code == 422


def test_campo_faltante_es_422(client):
    incompleto = {k: v for k, v in OCUPADO.items() if k != "horas_trabajadas"}
    assert client.post("/predict", json=incompleto).status_code == 422


def test_router_usa_los_campos_del_segmento_correcto(client):
    """Mandar campos de ocupado con trabajo_mes_pasado=False debe fallar, no colarse."""
    mezcla = {**OCUPADO, "trabajo_mes_pasado": False}
    assert client.post("/predict", json=mezcla).status_code == 422


def test_contrato_expone_las_features(client):
    d = client.get("/contrato/ocupados").json()
    assert len(d["features"]) == 10
