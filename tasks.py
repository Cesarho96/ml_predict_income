#!/usr/bin/env python
"""Ejecutor de tareas del proyecto. Reemplaza al Makefile.

    python tasks.py            # lista las tareas
    python tasks.py test
    python tasks.py build

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
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
IMAGEN = os.getenv("IMAGE", "ml-predict-income")
TAG = os.getenv("TAG", "0.1.0")
PUERTO = os.getenv("PORT", "8000")
PY = sys.executable

TAREAS: dict[str, str] = {}


def tarea(ayuda: str):
    def deco(fn):
        TAREAS[fn.__name__.replace("_", "-")] = ayuda
        return fn
    return deco


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


@tarea("re-serializa los bundles a la ruta de importación estable (una sola vez)")
def migrar():
    corre(PY, "scripts/migrate_bundles.py", "models")


@tarea("corre las pruebas")
def test():
    corre(PY, "-m", "pytest", "tests", "-q")


@tarea("revisa estilo y errores")
def lint():
    corre(PY, "-m", "ruff", "check", "src", "app", "tests", "scripts")


@tarea("corrige lo que ruff pueda corregir solo")
def fmt():
    corre(PY, "-m", "ruff", "check", "--fix", "src", "app", "tests", "scripts")


@tarea("levanta la API en local, sin Docker, con recarga automática")
def api():
    corre(PY, "-m", "uvicorn", "app.main:app", "--reload", "--port", PUERTO)


@tarea("entrena y registra los modelos en MLflow")
def train():
    corre(PY, "-m", "src.train")


@tarea("abre la UI de MLflow en http://127.0.0.1:5000")
def mlflow_ui():
    corre(PY, "-m", "mlflow", "ui", "--backend-store-uri",
          f"sqlite:///{RAIZ / 'mlflow.db'}", "--port", "5000")


@tarea("construye la imagen de Docker")
def build():
    corre("docker", "build", "-t", f"{IMAGEN}:{TAG}", ".")


@tarea("corre el contenedor con las restricciones que tendrá en producción")
def run():
    corre("docker", "run", "--rm", "-p", f"{PUERTO}:8000",
          "--read-only", "--tmpfs", "/tmp",
          "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
          "--memory", "1g", "--cpus", "1",
          f"{IMAGEN}:{TAG}")


@tarea("prueba de humo contra una API ya corriendo (local o contenedor)")
def smoke():
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
        print(f"  /predict  {code}  {cuerpo['mensaje']}")
    except urllib.error.URLError as e:
        sys.exit(f"no hay nadie escuchando en {base} — ¿corriste `python tasks.py api` "
                 f"o `python tasks.py run`?  ({e})")

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
        print(f"  {nombre:<10} {texto}")
    print(f"\nVariables: IMAGE={IMAGEN}  TAG={TAG}  PORT={PUERTO}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "ayuda", "help"):
        ayuda()
        sys.exit(0)
    nombre = sys.argv[1]
    fn = globals().get(nombre.replace("-", "_"))
    if not callable(fn) or nombre not in TAREAS:
        sys.exit(f"tarea desconocida: {nombre}\n(corre `python tasks.py` para ver la lista)")
    fn()
