"""El reconciliador (Hito 8c): lleva @champion al Deployment, sólo cuando hace falta y nunca a
medias. Y con permisos mínimos.

Sin cluster: el API de Kubernetes se sustituye por un Deployment en memoria (FakeKube) y,
para el cliente HTTP, por un servidor local que registra lo que recibe. El registry es un
MLflow real sobre SQLite.
"""

from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from mlflow import MlflowClient

from src import predictor, reconciliador
from src.reconciliador import Kube, declarado, parche, reconciliar, variable

K8S = Path(__file__).resolve().parent.parent / "k8s" / "api"


# ------------------------------------------------------------------ dobles de prueba
def deployment(**fijadas: str) -> dict:
    """Un Deployment `api` mínimo, con la forma que devuelve el API de Kubernetes."""
    env = [{"name": "PREDICCIONES_DSN", "valueFrom": {"secretKeyRef": {"name": "x"}}}]
    env += [{"name": variable(s), "value": reconciliador.uri(s, v)} for s, v in fijadas.items()]
    return {"metadata": {"name": "api", "annotations": {}},
            "spec": {"template": {"spec": {"containers": [{"name": "api", "env": env}]}}}}


class FakeKube:
    """Aplica el strategic merge patch como lo haría Kubernetes, para lo que usamos:
    anotaciones del Deployment y `env` del contenedor, combinados por `name`."""

    def __init__(self, dep: dict):
        self.dep, self.parches = dep, []

    def deployment(self, nombre):
        assert nombre == "api"
        return copy.deepcopy(self.dep)

    def parchar(self, nombre, cuerpo):
        self.parches.append(cuerpo)
        self.dep["metadata"]["annotations"].update(cuerpo["metadata"]["annotations"])
        for pc in cuerpo["spec"]["template"]["spec"]["containers"]:
            c = next(c for c in self.dep["spec"]["template"]["spec"]["containers"]
                     if c["name"] == pc["name"])
            por_nombre = {e["name"]: e for e in c["env"]}
            por_nombre.update({e["name"]: e for e in pc["env"]})
            c["env"] = list(por_nombre.values())
        return self.dep


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Dos modelos con 3 versiones cada uno (sin artefactos: aquí sólo importan aliases)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    for var in ("MODEL_URI_OCUPADOS", "MODEL_URI_NO_OCUPADOS"):
        monkeypatch.delenv(var, raising=False)
    client = MlflowClient()
    for nombre in reconciliador.NOMBRES.values():
        client.create_registered_model(nombre)
        for _ in range(3):
            client.create_model_version(nombre, source=(tmp_path / "x").as_uri())
    return client


def champion(client, version: str, segmentos=("ocupados", "no_ocupados")):
    for s in segmentos:
        client.set_registered_model_alias(reconciliador.NOMBRES[s], "champion", version)


# ------------------------------------------------------------------ la decisión
def test_primera_vez_fija_la_version_y_despues_queda_al_dia(registry):
    champion(registry, "2")
    kube = FakeKube(deployment())                 # recién desplegado: nada fijado

    r = reconciliar(registry, kube)
    assert r.parchado and r.declarado == {"ocupados": None, "no_ocupados": None}
    assert declarado(kube.dep) == {"ocupados": "2", "no_ocupados": "2"}

    r = reconciliar(registry, kube)               # el minuto siguiente
    assert r.al_dia and not r.parchado and len(kube.parches) == 1


def test_si_el_gate_mueve_el_champion_el_deployment_lo_sigue(registry):
    champion(registry, "3")
    kube = FakeKube(deployment(ocupados="2", no_ocupados="2"))

    assert reconciliar(registry, kube).parchado
    assert declarado(kube.dep) == {"ocupados": "3", "no_ocupados": "3"}
    assert "ingreso_ocupados v3" in kube.dep["metadata"]["annotations"][reconciliador.CAMBIO]


def test_el_parche_solo_toca_las_versiones(registry):
    """Strategic merge por `name`: el Secret de la base y lo demás no se pierden."""
    p = parche({"ocupados": "3", "no_ocupados": "2"})
    (contenedor,) = p["spec"]["template"]["spec"]["containers"]
    assert contenedor["name"] == "api"
    assert {e["name"] for e in contenedor["env"]} == {"MODEL_URI_OCUPADOS",
                                                     "MODEL_URI_NO_OCUPADOS"}
    champion(registry, "3")
    kube = FakeKube(deployment(ocupados="2", no_ocupados="2"))
    reconciliar(registry, kube)
    env = kube.dep["spec"]["template"]["spec"]["containers"][0]["env"]
    assert any(e["name"] == "PREDICCIONES_DSN" for e in env)


def test_sin_champion_en_un_segmento_no_toca_nada(registry):
    champion(registry, "2", segmentos=("ocupados",))       # no_ocupados sin champion
    kube = FakeKube(deployment(ocupados="1", no_ocupados="1"))
    with pytest.raises(RuntimeError, match="ingreso_no_ocupados@champion"):
        reconciliar(registry, kube)
    assert kube.parches == []


def test_simular_no_parcha(registry):
    champion(registry, "3")
    kube = FakeKube(deployment(ocupados="2", no_ocupados="2"))
    r = reconciliar(registry, kube, simular=True)
    assert not r.al_dia and not r.parchado and kube.parches == []


def test_un_rollout_undo_se_corrige_al_minuto(registry):
    """La fuente de verdad del MODELO es el registry: si alguien regresa el Deployment a
    versiones viejas, el reconciliador lo devuelve a @champion. Se opera moviendo el alias."""
    champion(registry, "3")
    kube = FakeKube(deployment(ocupados="3", no_ocupados="3"))
    kube.dep = deployment(ocupados="2", no_ocupados="2")      # `kubectl rollout undo`
    assert reconciliar(registry, kube).parchado
    assert declarado(kube.dep) == {"ocupados": "3", "no_ocupados": "3"}


# ------------------------------------------------------------------ contrato con la API
def test_la_api_carga_exactamente_la_version_que_fija_el_reconciliador(registry, monkeypatch):
    """Los mismos nombres y la misma variable que src/predictor.resolver honra."""
    assert reconciliador.NOMBRES == predictor.NOMBRES
    champion(registry, "3")
    for seg, v in {"ocupados": "2", "no_ocupados": "1"}.items():
        monkeypatch.setenv(variable(seg), reconciliador.uri(seg, v))
        uri, _, version, alias, _ = predictor.resolver(seg)
        # Fijada: carga v2 aunque @champion sea v3, y no dice "alias champion".
        assert (uri, version, alias) == (reconciliador.uri(seg, v), v, None)


# ------------------------------------------------------------------ el cliente HTTP
class _Grabadora(BaseHTTPRequestHandler):
    recibidas: list = []
    codigo = 200

    def _responder(self):
        largo = int(self.headers.get("Content-Length") or 0)
        self.recibidas.append({"metodo": self.command, "ruta": self.path,
                               "cabeceras": dict(self.headers),
                               "cuerpo": json.loads(self.rfile.read(largo)) if largo else None})
        self.send_response(self.codigo)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(deployment(ocupados="1", no_ocupados="1")).encode())

    do_GET = do_PATCH = _responder

    def log_message(self, *args):
        pass


@pytest.fixture
def servidor():
    _Grabadora.recibidas, _Grabadora.codigo = [], 200
    httpd = HTTPServer(("127.0.0.1", 0), _Grabadora)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_el_cliente_habla_el_dialecto_de_kubernetes(servidor):
    kube = Kube(base=servidor, token="tok", namespace="ml-predict-income")
    assert declarado(kube.deployment("api")) == {"ocupados": "1", "no_ocupados": "1"}
    kube.parchar("api", parche({"ocupados": "2", "no_ocupados": "2"}))

    get, patch = _Grabadora.recibidas
    assert get["ruta"] == "/apis/apps/v1/namespaces/ml-predict-income/deployments/api"
    assert get["cabeceras"]["Authorization"] == "Bearer tok"
    assert patch["metodo"] == "PATCH"
    # Sin este Content-Type, Kubernetes interpreta el cuerpo como merge patch "plano" y
    # REEMPLAZA la lista `env` completa: se perdería el Secret de la base.
    assert patch["cabeceras"]["Content-Type"] == "application/strategic-merge-patch+json"
    assert patch["ruta"].endswith("?fieldManager=reconciliador")


def test_un_403_dice_que_falta_permiso(servidor):
    _Grabadora.codigo = 403
    kube = Kube(base=servidor, token="tok", namespace="ml-predict-income")
    with pytest.raises(RuntimeError, match="Role `reconciliador`"):
        kube.deployment("api")


# ------------------------------------------------------------------ permisos mínimos
def test_el_role_solo_permite_leer_y_parchar_el_deployment_api():
    """Si alguien amplía el Role ("*" para que funcione), esta prueba lo detiene en el PR."""
    yaml = pytest.importorskip("yaml")
    docs = {d["kind"]: d for d in yaml.safe_load_all((K8S / "reconciliador.yaml").read_text(
        encoding="utf-8"))}
    assert docs["Role"]["rules"] == [{"apiGroups": ["apps"], "resources": ["deployments"],
                                      "resourceNames": ["api"], "verbs": ["get", "patch"]}]
    pod = docs["CronJob"]["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "reconciliador"
    (c,) = pod["containers"]
    assert c["image"] == "ml-predict-income"          # misma imagen (y tag) que la API
    assert c["command"] == ["python", "-m", "src.reconciliador"]
    assert docs["CronJob"]["spec"]["concurrencyPolicy"] == "Forbid"
    # La API sigue sin credenciales del cluster: sólo el reconciliador las tiene.
    api = yaml.safe_load((K8S / "deployment.yaml").read_text(encoding="utf-8"))
    assert api["spec"]["template"]["spec"]["automountServiceAccountToken"] is False
    assert "reconciliador.yaml" in (K8S / "kustomization.yaml").read_text(encoding="utf-8")
