#!/usr/bin/env python
"""Ejecutor de tareas del proyecto. Reemplaza al Makefile.

    python tasks.py            # lista las tareas
    python tasks.py test
    python tasks.py up
    python tasks.py promover ocupados 3     # las tareas pueden recibir argumentos

Por qué no un Makefile: `make` no viene con Windows, y un Makefile con `grep`/`awk`
dentro sólo corre en Unix. Este archivo usa el intérprete que ya tienes activado
(`sys.executable`), así que funciona igual en Windows, macOS, Linux y en el runner de
GitHub Actions — y eso significa que el comando que corres en tu máquina es literalmente
el mismo que corre CI. Cero dependencias: sólo la librería estándar.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _env(nombre: str, defecto: str) -> str:
    """Lee una variable de entorno SIN espacios a los lados.

    En cmd de Windows, `set PORT=8010 && python tasks.py smoke` guarda "8010 " —el
    espacio antes de `&&` forma parte del valor— y docker lo rechaza con
    `Invalid hostPort: 8010 .`. Como `set` dura toda la sesión, el error aparece
    después y en otro comando. Nunca confíes en que una variable de entorno llega limpia.
    """
    return os.getenv(nombre, defecto).strip()


RAIZ = Path(__file__).resolve().parent
IMAGEN = _env("IMAGE", "ml-predict-income")
TAG = _env("TAG", "dev")          # el mismo que usa docker-compose.yml
PUERTO = _env("PORT", "8000")
# Desde tu máquina, MLflow es el contenedor de compose publicado en 127.0.0.1:5000.
TRACKING = _env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PY = sys.executable
COMPOSE = ("docker", "compose")


def _validar_puerto() -> None:
    """Sólo para las tareas que usan el puerto: un PORT raro no debe romper `test`."""
    if not PUERTO.isdigit() or not 1 <= int(PUERTO) <= 65535:
        sys.exit(f"PORT inválido: {PUERTO!r}. Debe ser un número entre 1 y 65535.\n"
                 f"En cmd de Windows usa comillas:  set \"PORT=8010\"")

TAREAS: dict[str, str] = {}


def tarea(ayuda: str):
    def deco(fn):
        TAREAS[fn.__name__.replace("_", "-")] = ayuda
        return fn
    return deco


def _git_sha() -> str:
    """El commit que se entrena, marcado `-dirty` si hay cambios sin commitear.

    Un modelo entrenado con código que no está en ningún commit no se puede reproducir:
    la marca queda en la corrida de MLflow para que eso sea visible al promover.
    """
    def git(*a):
        return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        sucio = git("status", "--porcelain", "--untracked-files=no")
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if sucio else "")
    except Exception:  # noqa: BLE001
        return "desconocido"


def _get(url: str, timeout: float = 5) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def corre(*cmd: str, env: dict | None = None, check: bool = True) -> int:
    """Ejecuta sin shell: así los argumentos con espacios no se rompen en Windows."""
    print(f"$ {' '.join(cmd)}", flush=True)
    entorno = {**os.environ, "PYTHONPATH": str(RAIZ), **(env or {})}
    r = subprocess.run(cmd, cwd=RAIZ, env=entorno)
    if check and r.returncode:
        sys.exit(r.returncode)
    return r.returncode


# --------------------------------------------------------------------- tareas
@tarea("instala las dependencias de desarrollo")
def instalar():
    corre(PY, "-m", "pip", "install", "-r", "requirements-dev.txt")


@tarea("corre las pruebas (no necesitan Docker, ni MLflow, ni datos)")
def test():
    corre(PY, "-m", "pytest", "-q")


@tarea("revisa estilo y errores")
def lint():
    corre(PY, "-m", "ruff", "check", "src", "app", "tests", "scripts")


@tarea("corrige lo que ruff pueda corregir solo")
def fmt():
    corre(PY, "-m", "ruff", "check", "--fix", "src", "app", "tests", "scripts")


# ------------------------------------------------------------- MLflow y modelos
@tarea("levanta el servidor de MLflow (UI + registry) en http://127.0.0.1:5000")
def mlflow():
    corre(*COMPOSE, "up", "-d", "--build", "--wait", "mlflow")
    print("MLflow listo → http://127.0.0.1:5000")


@tarea("entrena EN CONTENEDOR Linux y registra (la forma canónica)")
def train():
    sha = _git_sha()
    if sha.endswith("-dirty"):
        print(f"⚠  hay cambios sin commitear: la corrida quedará marcada como {sha}")
    env = {"GIT_SHA": sha}
    corre(*COMPOSE, "--profile", "train", "build", "train", env=env)
    corre(*COMPOSE, "--profile", "train", "run", "--rm", "train", env=env)


@tarea("entrena con el Python de tu máquina (rápido, pero NO canónico)")
def train_local():
    print("⚠  entrenar en Windows da un modelo distinto al de Linux con el mismo commit.\n"
          "   Úsalo para iterar; para registrar lo que se va a servir: python tasks.py train")
    corre(PY, "-m", "src.train", env={"MLFLOW_TRACKING_URI": TRACKING, "GIT_SHA": _git_sha()})


@tarea("mueve un alias: promover <segmento> <versión> [alias]")
def promover(*args):
    corre(PY, "scripts/promover.py", *args, env={"MLFLOW_TRACKING_URI": TRACKING})


# ------------------------------------------------------------- API
@tarea("levanta la API en local, sin Docker, con recarga automática (usa MLflow)")
def api():
    _validar_puerto()
    corre(PY, "-m", "uvicorn", "app.main:app", "--reload", "--port", PUERTO,
          env={"MLFLOW_TRACKING_URI": TRACKING})


@tarea("levanta MLflow + la API en contenedores y espera a /ready")
def up():
    _validar_puerto()
    corre(*COMPOSE, "up", "-d", "--build", "api", env={"PORT": PUERTO, "TAG": TAG})
    # `up -d` regresa en cuanto el contenedor ARRANCA, no cuando está listo. Aquí se
    # espera a /ready, que es lo que haría un balanceador.
    url = f"http://127.0.0.1:{PUERTO}/ready"
    for _ in range(45):
        try:
            _, cuerpo = _get(url, timeout=2)
            for seg, m in cuerpo["modelos"].items():
                print(f"  {seg:<12} {m['nombre']} v{m['version']}  (alias {m['alias']})")
            print(f"API lista → http://127.0.0.1:{PUERTO}/docs")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(2)
    print("la API no llegó a /ready. Últimos logs:")
    corre(*COMPOSE, "logs", "--tail", "25", "api", check=False)
    sys.exit(1)


@tarea("apaga los contenedores (el volumen de MLflow se conserva)")
def down():
    corre(*COMPOSE, "down")


@tarea("muestra los logs de la API")
def logs():
    corre(*COMPOSE, "logs", "--tail", "60", "api")


@tarea("construye sólo la imagen de servicio, con tag (para inspect/scan)")
def build():
    corre("docker", "build", "-t", f"{IMAGEN}:{TAG}", ".")


@tarea("prueba de humo contra una API ya corriendo (local o contenedor)")
def smoke():
    _validar_puerto()
    base = f"http://127.0.0.1:{PUERTO}"
    caso = {"trabajo_mes_pasado": True, "horas_trabajadas": 48,
            "formalidad": "3 Formal (con seguridad social)", "cve_entidad": "09",
            "anios_escolaridad": 16, "sinco_grupo": "2 Profesionistas y técnicos",
            "posicion_ocupacion": "Subordinado remunerado", "es_jefe_hogar": 1,
            "sector_agrupado": "6 Servicios profesionales y financieros",
            "es_mujer": 0, "tam_empresa_grupo": "3 Mediana (51-250)"}

    def pide(ruta, cuerpo=None):
        datos = json.dumps(cuerpo).encode() if cuerpo else None
        cab = {"Content-Type": "application/json"} if cuerpo else {}
        req = urllib.request.Request(base + ruta, data=datos, headers=cab)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())

    try:
        for ruta in ("/health", "/ready"):
            code, cuerpo = pide(ruta)
            print(f"  {ruta:<9} {code}  {json.dumps(cuerpo, ensure_ascii=False)[:110]}")
        code, cuerpo = pide("/predict", caso)
        print(f"  /predict  {code}  {cuerpo['mensaje']}  [{cuerpo['version_modelo']}]")
    except urllib.error.URLError as e:
        sys.exit(f"no hay nadie escuchando en {base} — ¿corriste `python tasks.py up` "
                 f"o `python tasks.py api`?  ({e})")

    # El contrato debe rechazar lo que no conoce: un 422, no un 500 ni una adivinanza.
    try:
        pide("/predict", {**caso, "formalidad": "no existe"})
        sys.exit("FALLO: una categoría inválida fue aceptada")
    except urllib.error.HTTPError as e:
        print(f"  categoría inválida → HTTP {e.code} {'OK' if e.code == 422 else 'INESPERADO'}")


@tarea("revisa la imagen: tamaño, usuario y capas")
def inspect():
    corre("docker", "image", "ls", f"{IMAGEN}:{TAG}")
    corre("docker", "run", "--rm", "--entrypoint", "id", f"{IMAGEN}:{TAG}")


@tarea("busca vulnerabilidades conocidas en la imagen")
def scan():
    if corre("docker", "scout", "cves", f"{IMAGEN}:{TAG}", check=False):
        print("docker scout no disponible; alternativa: trivy image "
              f"{IMAGEN}:{TAG}")


@tarea("borra cachés locales")
def limpiar():
    import shutil
    for patron in ("__pycache__", ".pytest_cache", ".ruff_cache"):
        for p in RAIZ.rglob(patron):
            if ".venv" not in p.parts:
                shutil.rmtree(p, ignore_errors=True)
                print(f"  borrado {p.relative_to(RAIZ)}")


def ayuda():
    print(__doc__.split("\n\n")[0])
    print("\nTareas disponibles:\n")
    for nombre, texto in TAREAS.items():
        print(f"  {nombre:<12} {texto}")
    print(f"\nVariables: IMAGE={IMAGEN}  TAG={TAG}  PORT={PUERTO}  "
          f"MLFLOW_TRACKING_URI={TRACKING}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "ayuda", "help"):
        ayuda()
        sys.exit(0)
    nombre = sys.argv[1]
    fn = globals().get(nombre.replace("-", "_"))
    if not callable(fn) or nombre not in TAREAS:
        sys.exit(f"tarea desconocida: {nombre}\n(corre `python tasks.py` para ver la lista)")
    fn(*sys.argv[2:])
