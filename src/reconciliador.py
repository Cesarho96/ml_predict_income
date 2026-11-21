"""Reconciliador (Hito 8c): lleva el @champion del registry al Deployment `api`.

    python -m src.reconciliador              # lo corre el CronJob `reconciliador`, cada minuto
    python -m src.reconciliador --simular    # decide e imprime; no toca el cluster

El gate (src/promocion.py) mueve @champion en MLflow, pero eso no cambia lo que sirven
los pods: cada uno carga su modelo al arrancar. Este programa cierra esa brecha con el
patrón de los controladores de Kubernetes y de GitOps (Argo CD, Flux): comparar el
estado DESEADO con el DECLARADO y corregir la diferencia. Nadie empuja el cambio al
cluster: el cluster lo jala.

    deseado    = la versión que tiene @champion en el registry, por segmento
    declarado  = la versión FIJADA en el Deployment:
                 MODEL_URI_<SEGMENTO>=models:/<nombre>/<versión>   (src/predictor.py la honra)
    si difieren → PATCH al Deployment → Kubernetes hace un rolling update (Hito 6:
                  maxUnavailable 0, readiness, preStop): sin caída

Por qué fijar la VERSIÓN en el Deployment y no sólo reiniciar los pods:
  · Durante el rollout todos los pods nuevos cargan la MISMA versión, aunque @champion
    se mueva a la mitad.
  · El Deployment dice qué modelo corre (`kubectl describe`), y el historial de rollouts
    lo registra (`kubectl rollout history` → CHANGE-CAUSE).

Lo que NO hace:
  · No decide qué modelo es mejor: eso es del gate o de una persona. Obedece a @champion.
  · No actúa a medias: si no puede leer el champion de UN segmento, no toca nada.

Hay que saber operarlo: `kubectl rollout undo` regresa el Deployment a versiones viejas
y, al minuto, el reconciliador lo vuelve a poner en @champion. Está haciendo su trabajo:
la fuente de verdad del MODELO es el registry. Para regresar un modelo se mueve el alias
(`python tasks.py promover <segmento> <versión>`). Para detener la reconciliación en un
incidente, se suspende el CronJob (k8s/README.md).

Permisos: su ServiceAccount sólo puede leer y parchar el Deployment `api` de su propio
namespace (k8s/api/reconciliador.yaml). Nada más.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# Los mismos nombres que src/predictor.py (lo verifica tests/test_reconciliador.py). No se
# importa de ahí a propósito: predictor trae pandas y scikit-learn, y este proceso corre
# cada minuto sólo para leer dos aliases.
NOMBRES = {"ocupados": "ingreso_ocupados", "no_ocupados": "ingreso_no_ocupados"}
CHAMPION = "champion"
CONTENEDOR = "api"
CAMBIO = "kubernetes.io/change-cause"      # lo muestra `kubectl rollout history`
GESTOR = "reconciliador"                   # fieldManager: dueño de los campos que parcha
CUENTA = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def variable(segmento: str) -> str:
    """La variable que src/predictor.resolver lee para fijar la versión de un segmento."""
    return f"MODEL_URI_{segmento.upper()}"


def uri(segmento: str, version: str) -> str:
    return f"models:/{NOMBRES[segmento]}/{version}"


# ------------------------------------------------------------------ deseado y declarado
def deseado(client: MlflowClient) -> dict[str, str]:
    """{segmento: versión} de @champion. Si falta UNO, error: nunca se actúa a medias."""
    out = {}
    for seg, nombre in NOMBRES.items():
        try:
            mv = client.get_model_version_by_alias(nombre, CHAMPION)
        except MlflowException as e:
            raise RuntimeError(f"no se pudo leer {nombre}@{CHAMPION}: {e}") from e
        if mv.status != "READY":
            raise RuntimeError(f"{nombre} v{mv.version} está en estado {mv.status}, no READY")
        out[seg] = str(mv.version)
    return out


def declarado(deployment: dict) -> dict[str, str | None]:
    """{segmento: versión fijada en el contenedor `api`}; None si no hay versión fijada."""
    contenedores = deployment["spec"]["template"]["spec"]["containers"]
    api = next(c for c in contenedores if c["name"] == CONTENEDOR)
    env = {e["name"]: e.get("value") or "" for e in api.get("env", [])}
    out = {}
    for seg, nombre in NOMBRES.items():
        valor, prefijo = env.get(variable(seg), ""), f"models:/{nombre}/"
        out[seg] = valor.removeprefix(prefijo) if valor.startswith(prefijo) else None
    return out


def describir(versiones: dict[str, str | None]) -> str:
    return ", ".join(f"{NOMBRES[s]} v{v or '—'}" for s, v in versiones.items())


def parche(objetivo: dict[str, str]) -> dict:
    """Strategic merge patch: en `containers` y en `env` Kubernetes combina por `name`,
    así que sólo cambian estas dos variables; el resto del contenedor queda igual."""
    return {
        "metadata": {"annotations": {CAMBIO: f"reconciliador: @champion = "
                                              f"{describir(objetivo)}"}},
        "spec": {"template": {"spec": {"containers": [{
            "name": CONTENEDOR,
            "env": [{"name": variable(s), "value": uri(s, v)} for s, v in objetivo.items()],
        }]}}},
    }


# ------------------------------------------------------------------ API de Kubernetes
class Kube:
    """Lo mínimo del API de Kubernetes con la biblioteca estándar: leer y parchar un
    Deployment. Dentro de un pod, Kubernetes monta el token de la ServiceAccount, el CA
    del cluster y el namespace en CUENTA; la dirección del API llega por variables."""

    def __init__(self, base: str | None = None, token: str | None = None,
                 namespace: str | None = None, ca: str | None = None):
        if base is None:
            host = os.environ["KUBERNETES_SERVICE_HOST"]
            base = f"https://{host}:{os.getenv('KUBERNETES_SERVICE_PORT', '443')}"
        self.base = base
        self.token = token if token is not None else (CUENTA / "token").read_text().strip()
        self.namespace = namespace or (CUENTA / "namespace").read_text().strip()
        self.contexto = (ssl.create_default_context(cafile=ca or str(CUENTA / "ca.crt"))
                         if base.startswith("https") else None)

    def _ruta(self, nombre: str) -> str:
        return f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{nombre}"

    def _pedir(self, metodo: str, ruta: str, cuerpo: dict | None = None,
               tipo: str = "application/json") -> dict:
        cabeceras = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        datos = None
        if cuerpo is not None:
            datos, cabeceras["Content-Type"] = json.dumps(cuerpo).encode(), tipo
        req = urllib.request.Request(self.base + ruta, data=datos, method=metodo,
                                     headers=cabeceras)
        try:
            with urllib.request.urlopen(req, context=self.contexto, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            pista = {401: "el token no es válido", 403: "el Role `reconciliador` no lo "
                     "permite", 404: "no existe ese Deployment en este namespace"}
            raise RuntimeError(f"{metodo} {ruta} → HTTP {e.code}: "
                               f"{pista.get(e.code, e.reason)}") from e

    def deployment(self, nombre: str) -> dict:
        return self._pedir("GET", self._ruta(nombre))

    def parchar(self, nombre: str, cuerpo: dict) -> dict:
        consulta = urllib.parse.urlencode({"fieldManager": GESTOR})
        return self._pedir("PATCH", f"{self._ruta(nombre)}?{consulta}", cuerpo,
                           "application/strategic-merge-patch+json")


# ------------------------------------------------------------------ principal
@dataclass
class Resultado:
    deseado: dict[str, str]
    declarado: dict[str, str | None]
    parchado: bool

    @property
    def al_dia(self) -> bool:
        return self.deseado == self.declarado


def reconciliar(client: MlflowClient, kube, nombre: str = "api",
                simular: bool = False) -> Resultado:
    objetivo = deseado(client)                 # primero el registry: si falla, nada se toca
    actual = declarado(kube.deployment(nombre))
    r = Resultado(objetivo, actual, parchado=False)
    if not r.al_dia and not simular:
        kube.parchar(nombre, parche(objetivo))
        r.parchado = True
    return r


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deployment", default=os.getenv("RECONCILIAR_DEPLOYMENT", "api"))
    ap.add_argument("--simular", action="store_true", help="decide e imprime; no parcha")
    a = ap.parse_args(argv)
    try:
        r = reconciliar(MlflowClient(), Kube(), a.deployment, a.simular)
    except (RuntimeError, KeyError, OSError) as e:      # el Job queda Failed: visible
        sys.exit(f"no se reconcilió: {e}")

    print(f"@champion en el registry:      {describir(r.deseado)}")
    print(f"fijado en deployment/{a.deployment}:  {describir(r.declarado)}")
    if r.al_dia:
        print("→ al día: nada que hacer")
    elif r.parchado:
        print("→ PATCH aplicado: Kubernetes hace el rolling update "
              f"(kubectl rollout status deploy/{a.deployment})")
    else:
        print("→ difieren; --simular: no se parchó")
    print(json.dumps({"deseado": r.deseado, "declarado": r.declarado,
                      "parchado": r.parchado}, ensure_ascii=False))


if __name__ == "__main__":
    main()
