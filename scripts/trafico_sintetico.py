"""Tráfico sintético para probar el monitoreo de drift (Hito 7c).

    python scripts/trafico_sintetico.py --perfil poblacion --puerto 64012
    python scripts/trafico_sintetico.py --perfil sesgado   --puerto 64012

Dos perfiles, y cada uno debe dar un resultado distinto. Si no, el monitoreo está mal:

  poblacion   personas muestreadas de la ENIGH con probabilidad ∝ factor de expansión:
              la población de México. Contra la referencia, PSI ≈ 0 en todo ("estable").
              Es el CONTROL: si esto marca drift, la medición está rota.

  sesgado     las mismas personas, pero como sería el tráfico de una web: todos en la
              Ciudad de México y con 18 años de escolaridad (posgrado). Debe marcar
              "actuar" en cve_entidad y anios_escolaridad, y en nada más.

Usa el dataset de data/processed (el mismo del que sale la referencia) y le pega a la API
por el Gateway, como un usuario. Sólo la biblioteca estándar + pandas.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import datos  # noqa: E402

ENTEROS = {"es_jefe_hogar", "es_mujer"}
SEGMENTO_API = {"Ocupados": True, "No ocupados": False}


def muestra(n: int, semilla: int) -> pd.DataFrame:
    df = pd.read_parquet(datos.PROC / datos.dataset(datos.EDICION))
    # Con reemplazo y ∝ factor: cada fila representa a `factor` personas reales.
    return df.sample(n=n, replace=True, weights="factor", random_state=semilla)


def sesgar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cve_entidad"] = "09"
    df["anios_escolaridad"] = 18.0
    return df


def peticiones(df: pd.DataFrame) -> list[dict]:
    feats = {seg: json.loads((datos.PROC / f"features_{seg}.json").read_text(
        encoding="utf-8"))["features"] for seg in ("ocupados", "no_ocupados")}
    salida = []
    for fila in df.to_dict("records"):
        ocupado = SEGMENTO_API[fila["segmento"]]
        cuerpo = {"trabajo_mes_pasado": ocupado}
        for f in feats["ocupados" if ocupado else "no_ocupados"]:
            v = fila[f]
            cuerpo[f] = int(v) if f in ENTEROS else (float(v) if not isinstance(v, str) else v)
        salida.append(cuerpo)
    return salida


def comprobar(base: str) -> None:
    """Una petición a /health antes de mandar miles: si la puerta de entrada no responde,
    mejor saberlo aquí, con qué hacer, que con 4,000 errores iguales."""
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            if r.status == 200:
                return
            problema = f"respondió HTTP {r.status}"
    except urllib.error.HTTPError as e:
        problema = f"respondió HTTP {e.code}"
    except (OSError, http.client.HTTPException) as e:
        problema = f"no respondió ({type(e).__name__})"
    sys.exit(f"{base} {problema}. No se envió nada.\n"
             "Corre `python tasks.py gateway`: revisa la puerta de entrada del cluster, la "
             "repara si hace falta\ny te dice el puerto actual (cambia cada vez que se "
             "reinicia Docker Desktop).")


def enviar(url: str, cuerpos: list[dict], hilos: int) -> Counter:
    codigos: Counter = Counter()
    candado = threading.Lock()
    pendientes = iter(cuerpos)

    def trabajador():
        while True:
            with candado:
                cuerpo = next(pendientes, None)
            if cuerpo is None:
                return
            req = urllib.request.Request(url, data=json.dumps(cuerpo).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    codigo = r.status
            except urllib.error.HTTPError as e:
                codigo = e.code
            except (OSError, http.client.HTTPException):
                codigo = "sin respuesta"
            with candado:
                codigos[codigo] += 1

    t0 = time.time()
    hs = [threading.Thread(target=trabajador) for _ in range(hilos)]
    for h in hs:
        h.start()
    for h in hs:
        h.join()
    print(f"{len(cuerpos):,} peticiones en {time.time() - t0:.0f} s → {dict(codigos)}")
    return codigos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perfil", choices=["poblacion", "sesgado"], required=True)
    ap.add_argument("--puerto", type=int, required=True, help="el del Gateway (docker ps)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--hilos", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.puerto}"
    comprobar(base)
    df = muestra(a.n, a.semilla)
    if a.perfil == "sesgado":
        df = sesgar(df)
    print(f"perfil {a.perfil}: {df['segmento'].value_counts().to_dict()}")
    codigos = enviar(f"{base}/predict", peticiones(df), a.hilos)
    if set(codigos) != {200}:
        sys.exit("hubo respuestas distintas de 200: el drift de esta ventana no es fiable. "
                 "`python tasks.py gateway` dice qué capa falla.")


if __name__ == "__main__":
    main()
