"""El gate de promoción: ¿el challenger que acaba de entrenar el DAG pasa a @champion? (Hito 8)

    python -m src.promocion                         # lo corre el DAG `reentrenar`
    python -m src.promocion --corrida <id> --simular    # a mano: decide, pero no toca nada

    exit 0   al menos un segmento se promovió (o ya estaba promovido: reintento)
    exit 99  nada se promovió → Airflow marca la tarea "skipped"

Última tarea de `reentrenar`, después de `comparar_con_champion`. `comparar` midió y dejó
la evidencia escrita en la versión; este módulo la LEE y decide. Promueve sólo si se
cumple TODO, y siempre deja escrito por qué (tags `gate*` en la versión):

  1. veredicto "mejor": supera al champion por MÁS que la banda de ruido. "equivalente"
     no promueve: cambiar de modelo tiene costo (un despliegue, un riesgo) y un empate
     no lo paga. Ésta es la regla que rechaza el reentrenamiento por drift con los mismos
     datos.
  2. cobertura ≥ 78 %: el intervalo promete 80 %. Un MdAPE mejor no compensa un
     intervalo que miente.
  3. comparado contra el champion ACTUAL: si alguien movió @champion entre `comparar` y
     este paso, el veredicto es viejo. Y si no hay champion, no hay contra qué ganar:
     el primero lo nombra una persona (`tasks.py promover`).
  4. commit limpio: un modelo `-dirty` no se puede reproducir desde git.
  5. entrenado por el orquestador (origen airflow): sólo lo que pasó por el DAG completo
     (dataset → validación → entrenamiento → comparación) tiene toda la evidencia.
  6. edición de la ENIGH ≥ la del champion: nunca se regresa a datos más viejos.
  7. no marcado como `degradado` (modelos hechos peores a propósito para demostraciones).

Cada segmento se decide por separado: son modelos independientes.

Promover = mover @champion a la versión nueva, dejar la anterior como @anterior (el
rollback es una línea, la imprime al final) y retirar @challenger. Los pods vivos no
cambian solos: leen @champion al arrancar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.comparar import CHALLENGER, CHAMPION, COBERTURA_MINIMA, champion_de, version_de_la_corrida
from src.train import SEGMENTOS

ANTERIOR = "anterior"
SALTAR = 99   # el DAG `reentrenar` lo declara en skip_on_exit_code (tests/test_promocion.py)


@dataclass(frozen=True)
class Evidencia:
    """Lo que se sabe de una versión, leído del registry (tags y params)."""
    version: str
    veredicto: str | None          # de `comparar`
    cobertura: float | None        # de `comparar`, % sobre el test actual
    champion_comparado: str | None  # de `comparar`
    git_sha: str                   # de la corrida de entrenamiento
    origen: str                    # airflow | manual
    edicion: int | None            # edición de la ENIGH con la que se entrenó
    degradado: bool


@dataclass(frozen=True)
class Criterio:
    nombre: str
    cumple: bool
    detalle: str


@dataclass
class Decision:
    segmento: str
    nombre: str
    version: str
    champion_antes: str | None
    promovida: bool
    criterios: list[Criterio]
    nota: str = ""

    @property
    def motivo(self) -> str:
        if self.nota:
            return self.nota
        fallas = [c.detalle for c in self.criterios if not c.cumple]
        return "; ".join(fallas) if fallas else "cumple todos los criterios"


# ------------------------------------------------------------------ la regla (pura)
def criterios(ev: Evidencia, champion_actual: str | None,
              edicion_champion: int | None) -> list[Criterio]:
    """Evalúa TODOS (no se detiene en el primero): la auditoría dice cada razón."""
    sha = ev.git_sha.strip()
    if champion_actual is None:
        comparado = Criterio("champion actual", False,
                             "no hay champion: el primero lo nombra una persona")
    else:
        comparado = Criterio(
            "champion actual", ev.champion_comparado == champion_actual,
            f"se comparó contra v{ev.champion_comparado or '—'} y el champion actual es "
            f"v{champion_actual}")
    return [
        Criterio("veredicto", ev.veredicto == "mejor",
                 f"veredicto {ev.veredicto or 'sin veredicto'!r}: sólo 'mejor' promueve"),
        Criterio("cobertura", ev.cobertura is not None and ev.cobertura >= COBERTURA_MINIMA,
                 f"cobertura {'—' if ev.cobertura is None else f'{ev.cobertura:.1f}'} % "
                 f"(mínimo {COBERTURA_MINIMA:.0f} %)"),
        comparado,
        Criterio("commit limpio",
                 bool(sha) and sha != "desconocido" and not sha.endswith("-dirty"),
                 f"commit {sha or '—'!r}"),
        Criterio("orquestador", ev.origen == "airflow",
                 f"origen {ev.origen!r}: sólo se promueve lo que entrenó el DAG"),
        Criterio("edición",
                 ev.edicion is not None
                 and (edicion_champion is None or ev.edicion >= edicion_champion),
                 f"edición {ev.edicion or '—'} vs {edicion_champion or '—'} del champion"),
        Criterio("no degradado", not ev.degradado,
                 "marcado 'degradado': hecho peor a propósito, nunca se promueve"
                 if ev.degradado else "sin la marca 'degradado'"),
    ]


# ------------------------------------------------------------------ registry
def _edicion(run) -> int | None:
    try:
        return int(run.data.params["edicion_enigh"])
    except (KeyError, ValueError):
        return None      # versiones anteriores al Hito 5b no la registraban


def _float(x: str | None) -> float | None:
    try:
        return float(x) if x is not None else None
    except ValueError:
        return None


def evidencia(client: MlflowClient, mv) -> Evidencia:
    run = client.get_run(mv.run_id)
    t = mv.tags
    return Evidencia(
        version=str(mv.version), veredicto=t.get("veredicto"),
        cobertura=_float(t.get("cobertura_test")),
        champion_comparado=t.get("champion_comparado"),
        git_sha=run.data.tags.get("git_sha", ""),
        origen=run.data.tags.get("origen", "manual"),
        edicion=_edicion(run),
        degradado=run.data.tags.get("degradado", "").lower() in ("si", "sí", "true", "1"))


def mover_champion(client: MlflowClient, nombre: str, nueva: str, quien: str,
                   motivo: str) -> str | None:
    """@champion → `nueva`; el que estaba queda como @anterior; @challenger se retira.

    Lo usan el gate y `scripts/promover.py`: promover a mano deja el mismo rastro.
    El orden importa: @anterior primero (inofensivo), luego @champion (el paso que
    cuenta), luego la limpieza. Si algo se cae a la mitad, nunca queda sin champion.
    """
    aliases = {k: str(v) for k, v in client.get_registered_model(nombre).aliases.items()}
    previo = aliases.get(CHAMPION)
    if previo == nueva:
        return previo
    ahora = datetime.now(UTC).isoformat(timespec="seconds")
    if previo:
        client.set_registered_model_alias(nombre, ANTERIOR, previo)
        client.set_model_version_tag(nombre, previo, "reemplazada_por", nueva)
    client.set_registered_model_alias(nombre, CHAMPION, nueva)
    if aliases.get(CHALLENGER) == nueva:
        client.delete_registered_model_alias(nombre, CHALLENGER)
    for k, v in {"promovida": ahora, "promovida_por": quien, "promocion_motivo": motivo,
                 "reemplaza": previo}.items():
        if v:
            client.set_model_version_tag(nombre, nueva, k, v)
    return previo


def decidir(corrida: str, client: MlflowClient | None = None,
            simular: bool = False) -> list[Decision]:
    client = client or MlflowClient()
    decisiones = []
    for slug, cfg in SEGMENTOS.items():
        nombre = cfg["nombre_mlflow"]
        mv = version_de_la_corrida(client, nombre, slug, corrida)
        version = str(mv.version)
        champ = champion_de(client, nombre)
        champ_v = str(champ.version) if champ else None

        if champ_v == version:
            # Reintento de una corrida que ya promovió: no hay nada que hacer.
            decisiones.append(Decision(slug, nombre, version, champ_v, True, [],
                                       nota="ya es @champion (reintento)"))
            continue

        ed_champ = _edicion(client.get_run(champ.run_id)) if champ else None
        checks = criterios(evidencia(client, mv), champ_v, ed_champ)
        d = Decision(slug, nombre, version, champ_v, all(c.cumple for c in checks), checks)
        if not simular:
            for k, v in {"gate": "promovida" if d.promovida else "rechazada",
                         "gate_motivo": d.motivo, "gate_corrida": corrida,
                         "gate_fecha": datetime.now(UTC).isoformat(timespec="seconds")}.items():
                client.set_model_version_tag(nombre, version, k, v[:5000])
            if d.promovida:
                mover_champion(client, nombre, version, "gate", d.motivo)
        decisiones.append(d)
    return decisiones


# ------------------------------------------------------------------ informe
def informe(decisiones: list[Decision], simular: bool) -> str:
    lineas = []
    for d in decisiones:
        lineas.append(f"\n{d.nombre}  v{d.version}  (champion: "
                      f"v{d.champion_antes or '—'})")
        for c in d.criterios:
            lineas.append(f"  {'✓' if c.cumple else '✗'} {c.nombre:<16} {c.detalle}")
        if d.promovida and not d.nota:
            lineas.append(f"  → PROMOVIDA: @champion v{d.champion_antes or '—'} → "
                          f"v{d.version}" + (" (simulado)" if simular else ""))
            if d.champion_antes and not simular:
                lineas.append(f"    rollback: python tasks.py promover {d.segmento} "
                              f"{d.champion_antes}")
        elif d.nota:
            lineas.append(f"  → {d.nota}")
        else:
            lineas.append("  → RECHAZADA: @champion no cambia"
                          + (" (simulado)" if simular else ""))
    if simular:
        lineas.append("\n--simular: no se escribió nada en el registry.")
    elif any(d.promovida and not d.nota for d in decisiones):
        lineas.append("\nLos pods vivos siguen con el modelo anterior hasta reiniciar: "
                      "kubectl rollout restart deployment/api -n ml-predict-income")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corrida", default=os.getenv("ORQUESTADOR_RUN_ID", "").strip(),
                    help="run_id de Airflow que entrenó (por omisión $ORQUESTADOR_RUN_ID)")
    ap.add_argument("--simular", action="store_true",
                    help="decide e imprime, pero no escribe nada en el registry")
    a = ap.parse_args(argv)
    if not a.corrida:
        sys.exit("falta --corrida (o la variable ORQUESTADOR_RUN_ID)")
    try:
        decisiones = decidir(a.corrida, simular=a.simular)
    except (RuntimeError, MlflowException) as e:    # sin su versión: la tarea FALLA
        sys.exit(f"no se pudo decidir: {e}")

    print(informe(decisiones, a.simular))
    # Última línea = XCom: visible en la UI de Airflow.
    print(json.dumps({d.segmento: {"version": d.version, "promovida": d.promovida,
                                   "motivo": d.motivo} for d in decisiones},
                     ensure_ascii=False))
    sys.stdout.flush()
    sys.exit(0 if any(d.promovida for d in decisiones) else SALTAR)


if __name__ == "__main__":
    main()
