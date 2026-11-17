"""¿El drift es SOSTENIDO? Decide si se dispara el reentrenamiento (Hito 8).

    python -m src.monitoreo.persistencia --mediciones 3      # lo corre el DAG `monitor`

    exit 0   empezó un episodio de drift sostenido → Airflow emite el evento
             `drift_sostenido` y el DAG `reentrenar` arranca solo
    exit 99  no hay nada que disparar → Airflow marca la tarea "skipped" y no emite nada

Tres reglas:

1. **Sostenido, no puntual.** Un día raro (una campaña en redes, un bot) no justifica
   reentrenar. Se exige drift —al menos una feature en "actuar"— en las últimas N
   mediciones seguidas. Con la corrida nocturna, N mediciones = N noches.

2. **Se dispara al EMPEZAR, no mientras dure.** Si el drift sigue un mes, reentrenar cada
   noche con los mismos datos produciría 30 modelos iguales. El disparo es por FLANCO,
   como una alarma que suena al cruzar el umbral y no todo el tiempo que sigue arriba:
   sostenido en esta medición Y no sostenido en la medición previa a la racha. Si el
   drift desaparece y vuelve, es un episodio nuevo y vuelve a disparar.

3. **Una medición sin peticiones no cuenta**, ni a favor ni en contra: una noche sin
   tráfico no "cura" el drift ni lo confirma.

La decisión se toma respecto a la corrida del DAG que la pide (--corrida), no a "lo más
reciente": reintentar la misma corrida de Airflow da la misma decisión.

Lo que este módulo NO promete: que reentrenar arregle el drift. Con la misma ENIGH, el
modelo nuevo saldrá igual al actual y el gate lo rechazará (Hito 8b). Reentrenar sólo
puede ayudar si cambió algo más: una edición nueva de la encuesta, o el código.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

SALTAR = 99   # el DAG `monitor` lo declara en skip_on_exit_code (tests/test_persistencia.py)


@dataclass(frozen=True)
class Medicion:
    corrida: str
    fin: datetime          # fin de la ventana que midió
    actuar: int            # cuántas features quedaron en "actuar"
    con_datos: bool        # si hubo peticiones en la ventana


@dataclass(frozen=True)
class Decision:
    disparar: bool
    motivo: str
    racha: list[str] = field(default_factory=list)   # corridas consideradas, de vieja a nueva


def decidir(historia: list[Medicion], corrida: str, n: int) -> Decision:
    """`historia`: todas las mediciones, en cualquier orden. `corrida`: la que decide."""
    propia = next((m for m in historia if m.corrida == corrida), None)
    if propia is None:
        raise ValueError(f"no hay medición de la corrida {corrida!r}: ¿corrió calcular_drift?")
    if not propia.con_datos:
        return Decision(False, "esta medición no tuvo peticiones: no confirma ni descarta nada")

    # Sólo lo medido HASTA esta corrida, y sólo mediciones con datos (regla 3).
    hasta_aqui = sorted((m for m in historia
                         if m.con_datos and (m.fin, m.corrida) <= (propia.fin, propia.corrida)),
                        key=lambda m: (m.fin, m.corrida))
    racha = hasta_aqui[-n:]
    nombres = [m.corrida for m in racha]
    if len(racha) < n:
        return Decision(False, f"hay {len(racha)} medición(es) con datos; se necesitan {n}",
                        nombres)
    if not all(m.actuar > 0 for m in racha):
        limpias = sum(m.actuar == 0 for m in racha)
        return Decision(False, f"no sostenido: {limpias} de las últimas {n} mediciones sin "
                               "drift", nombres)
    previa = hasta_aqui[-n - 1] if len(hasta_aqui) > n else None
    if previa is not None and previa.actuar > 0:
        return Decision(False, "el episodio ya venía de antes de esta racha y ya disparó; "
                               "no se reentrena cada noche con los mismos datos", nombres)
    return Decision(True, f"drift en {n} medición(es) seguida(s): empieza un episodio", nombres)


SQL_HISTORIA = """
    SELECT corrida,
           max(ventana_fin)                          AS fin,
           count(*) FILTER (WHERE nivel = 'actuar')  AS actuar,
           max(n_actual) > 0                         AS con_datos
    FROM drift
    GROUP BY corrida
"""


def leer_historia(con) -> list[Medicion]:
    return [Medicion(c, f, int(a), bool(d)) for c, f, a, d in con.execute(SQL_HISTORIA)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corrida", default=os.getenv("ORQUESTADOR_RUN_ID", "").strip())
    ap.add_argument("--mediciones", type=int, default=3,
                    help="cuántas mediciones seguidas con drift hacen un episodio")
    a = ap.parse_args(argv)
    if not a.corrida:
        sys.exit("falta --corrida (o la variable ORQUESTADOR_RUN_ID)")

    import psycopg

    with psycopg.connect(os.environ["MONITOR_DSN"], connect_timeout=10,
                         application_name="monitor-persistencia") as con:
        historia = leer_historia(con)
    try:
        d = decidir(historia, a.corrida, a.mediciones)
    except ValueError as e:          # sin su medición, la tarea FALLA (exit 1): no es un skip
        sys.exit(str(e))

    print(("DISPARAR reentrenamiento: " if d.disparar else "no se dispara: ") + d.motivo)
    for c in d.racha:
        print(f"  · {c}")
    # Última línea = XCom: visible en la UI de Airflow.
    print(json.dumps({"disparar": d.disparar, "motivo": d.motivo, "racha": d.racha},
                     ensure_ascii=False))
    sys.stdout.flush()
    sys.exit(0 if d.disparar else SALTAR)


if __name__ == "__main__":
    main()
