"""Las métricas que lee Prometheus (Hito 7): que existan, que cuenten y que no exploten."""

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.main import app
from tests.test_api import OCUPADO


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def total(nombre: str, **etiquetas) -> float:
    """Suma de las muestras de una métrica que tienen esas etiquetas."""
    return sum(s.value for m in REGISTRY.collect() for s in m.samples
               if s.name == nombre
               and all(s.labels.get(k) == v for k, v in etiquetas.items()))


def test_metrics_habla_el_formato_de_prometheus(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in r.text and "income_model_info" in r.text


def test_una_prediccion_mueve_los_contadores(client):
    antes_pred = total("income_predictions_total", segment="ocupados")
    antes_http = total("http_requests_total", route="/predict", status="200")
    antes_hist = total("income_predicted_mxn_count", segment="ocupados")

    assert client.post("/predict", json=OCUPADO).status_code == 200

    assert total("income_predictions_total", segment="ocupados") == antes_pred + 1
    assert total("http_requests_total", route="/predict", status="200") == antes_http + 1
    assert total("income_predicted_mxn_count", segment="ocupados") == antes_hist + 1


def test_un_422_se_cuenta_como_422(client):
    antes = total("http_requests_total", route="/predict", status="422")
    client.post("/predict", json={**OCUPADO, "formalidad": "no existe"})
    assert total("http_requests_total", route="/predict", status="422") == antes + 1


def test_las_urls_inventadas_no_crean_series(client):
    """Un escáner pidiendo rutas al azar no debe crear una serie por URL."""
    antes = total("http_requests_total", route="sin_ruta", status="404")
    for i in range(3):
        assert client.get(f"/wp-admin/{i}.php").status_code == 404
    assert total("http_requests_total", route="sin_ruta", status="404") == antes + 3
    rutas = {s.labels["route"] for m in REGISTRY.collect() for s in m.samples
             if s.name == "http_requests_total"}
    assert not any(r.startswith("/wp-admin") for r in rutas)


def test_la_ruta_se_registra_como_plantilla(client):
    client.get("/contrato/ocupados")
    rutas = {s.labels["route"] for m in REGISTRY.collect() for s in m.samples
             if s.name == "http_requests_total"}
    assert "/contrato/{segmento}" in rutas and "/contrato/ocupados" not in rutas


def test_cada_pod_dice_que_modelo_sirve(client):
    segmentos = {s.labels["segment"] for m in REGISTRY.collect() for s in m.samples
                 if s.name == "income_model_info" and s.value == 1}
    assert segmentos == {"ocupados", "no_ocupados"}
