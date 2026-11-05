# Kubernetes local (Hito 6)

La API servida en un cluster de [kind](https://kind.sigs.k8s.io/), con la misma imagen que
publica CI. MLflow **no** vive en el cluster: es el servicio de `docker-compose.yml`
(el registry compartido con Airflow); el cluster lo alcanza por un alias de DNS.

```
tu máquina ──▶ localhost:<puerto> ──▶ Envoy (Gateway `entrada`) ──▶ HTTPRoute ──▶ Service `api`
                                                                                    │
                    MLflow (compose, :5000) ◀── Service `mlflow` (ExternalName) ◀── pods de la API (2–4, HPA)
```

| Ruta | Qué es | Quién la administra en una empresa |
|---|---|---|
| `kind.yaml` | el cluster (1 nodo) | plataforma |
| `cloud-provider-kind.compose.yaml` | LoadBalancer + controlador de Gateway API para kind | (en la nube: el proveedor) |
| `cluster/metrics-server/` | métricas de CPU/memoria (`kubectl top`, HPA) | plataforma |
| `cluster/gateway/` | la puerta de entrada y qué namespaces pueden publicar rutas | plataforma |
| `cluster/monitoreo/` | Prometheus + Grafana (Helm: kube-prometheus-stack) | plataforma |
| `api/` | la app: Deployment, Service, HPA, PDB, ruta, config, alias de MLflow, ServiceMonitor, alertas, dashboard | equipo de la app |
| `pruebas/` | tráfico y carga desde dentro del cluster (no son parte de la app) | — |

## Requisitos (Windows)

- Docker Desktop con **cgroups v2**: `docker info --format "{{.CgroupVersion}}"` → `2`.
  Si dice `1`, en `%USERPROFILE%\.wslconfig`: `[wsl2]` / `kernelCommandLine = cgroup_no_v1=all`,
  luego `wsl --shutdown`. Kubernetes ≥ 1.35 no arranca sobre cgroups v1.
- `winget install Kubernetes.kind`, `winget install Kubernetes.kubectl` y `winget install Helm.Helm`.
- **kubectl ≥ 1.36** (el cluster es 1.37). Docker Desktop trae un kubectl viejo que gana en
  el PATH: `where.exe kubectl` debe listar primero el de WinGet. Un cliente viejo no falla:
  descarta en silencio los campos que no conoce (así se rompió `rollout undo` con `preStop.sleep`).

## Desde cero

```powershell
docker compose up -d mlflow                                   # el cluster depende de él
python tasks.py predicciones                                  # registro de predicciones (Hito 7b)
kind create cluster --config k8s/kind.yaml

# La imagen que dice k8s/api/kustomization.yaml (newTag = commit del código).
# Si HEAD no es ese commit: git switch --detach <tag>, construir, git switch -
$tag = git rev-parse --short HEAD
docker build --build-arg GIT_SHA=$tag -t ml-predict-income:$tag .
kind load docker-image ml-predict-income:$tag --name ml-predict-income

docker compose -f k8s/cloud-provider-kind.compose.yaml up -d  # antes de api/: instala las CRDs
kubectl apply -k k8s/cluster/metrics-server
kubectl apply -k k8s/cluster/gateway

# Monitoreo, antes de api/: instala las CRDs ServiceMonitor y PrometheusRule
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
kubectl apply -f k8s/cluster/monitoreo/namespace.yaml
helm upgrade --install monitoreo prometheus-community/kube-prometheus-stack `
  --version 92.14.1 -n monitoreo -f k8s/cluster/monitoreo/values.yaml --wait --timeout 10m

python tasks.py secreto-k8s                                   # Secret con la conexión (no va en git)
kubectl apply -k k8s/api

kubectl get pods -n ml-predict-income -w                      # 2 pods 1/1
docker ps --filter name=kindccm                               # puerto de entrada: 0.0.0.0:<P>->80
$env:PORT = "<P>"; python tasks.py smoke
```

## Desplegar una versión nueva y revertirla

```powershell
$sha = git rev-parse --short HEAD                             # código ya commiteado
docker build --build-arg GIT_SHA=$sha -t ml-predict-income:$sha .
kind load docker-image ml-predict-income:$sha --name ml-predict-income
# editar newTag: "<sha>" en k8s/api/kustomization.yaml (con comillas)
kubectl apply -k k8s/api
kubectl rollout status deploy/api -n ml-predict-income

kubectl rollout undo deploy/api -n ml-predict-income          # palanca de emergencia
# …y después alinear git (revert del newTag) o volver con `kubectl apply -k k8s/api`
```

Para comprobar que no se cae ni una petición: `kubectl apply -f k8s/pruebas/trafico.yaml`
y `kubectl logs -f -n ml-predict-income trafico` durante el despliegue.

## Monitoreo

```powershell
kubectl port-forward -n monitoreo svc/monitoreo-kube-prometheus-prometheus 9090:9090   # Prometheus
kubectl port-forward -n monitoreo svc/monitoreo-grafana 3000:80                        # Grafana
# contraseña de admin de Grafana al portapapeles (sin imprimirla):
$p = kubectl get secret -n monitoreo monitoreo-grafana -o jsonpath="{.data.admin-password}"
[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p)) | Set-Clipboard
```

- Prometheus → Status → Target health: `serviceMonitor/ml-predict-income/api/0`, un endpoint por pod.
- Grafana → dashboard **API de ingreso — servicio y modelo** (`api/dashboards/api.json`).
  Se edita en git, no en la UI: lo hecho en la UI se pierde al recrear el pod.
- Alertas: `api/alertas.yaml`, probadas con promtool (`tests/alertas/`, en CI).

## Trampas que ya aparecieron

| Síntoma | Causa |
|---|---|
| `kubeadm init` falla al crear el cluster | cgroups v1 en WSL2 |
| `RESTARTS` crece, `CrashLoopBackOff`, `Connection refused` a `mlflow:5000` | MLflow de compose apagado: la API falla rápido al arrancar sin registry (a propósito) |
| `rollout undo` → `preStop: Required value` | kubectl viejo (el de Docker Desktop) |
| `ErrImagePull` | la imagen no se cargó con `kind load` (el nodo no ve las imágenes de Docker Desktop) |
| un pod se reinicia al quitar `replicas` | el Deployment y el HPA eran dueños del mismo campo |
| la ruta da 404 | el namespace perdió la etiqueta `entrada/rutas: permitidas` (`describe httproute` → `NotAllowedByListeners`) |
| `port-forward` se cae en un rollout | es un túnel a UN pod; para probar, tráfico al Service desde dentro |
| `CreateContainerConfigError` en los pods de la API | falta el Secret `api-predicciones`: `python tasks.py secreto-k8s` |
| la API no guarda predicciones (`prediction_log_dropped_total{reason="error_base"}` sube) | `predicciones-db` apagada, o el DSN apunta a :5432 (el ExternalName no traduce puertos: es :5433) |
| `predicciones-db` arranca pero faltan los roles `api_escritura`/`lector` | el esquema sólo corre con el volumen VACÍO; si el primer arranque falló a la mitad, nunca vuelve a correr. Con la base vacía: `docker compose rm -sf predicciones-db`, `docker volume rm ml-predict-income_predicciones-db`, `python tasks.py predicciones` |
| Prometheus no lee el ServiceMonitor | el chart sólo lee los que llevan `release: monitoreo`; `values.yaml` lo desactiva (`*SelectorNilUsesHelmValues: false`) |

## Apagar

```powershell
kind delete cluster --name ml-predict-income
docker compose -f k8s/cloud-provider-kind.compose.yaml down
```

CI valida estos manifiestos en cada PR (job `Manifiestos de Kubernetes`: kustomize +
kubeconform `-strict` contra Kubernetes 1.37 y Gateway API).
