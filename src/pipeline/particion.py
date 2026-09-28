"""Partición train/validation/test por UPM, estable entre ediciones de la ENIGH.

Dos reglas:
1. **Una UPM que ya tiene partición la conserva.** La de 2024 está versionada en git
   (split_upm_3way.csv) y es una decisión: la selección de features del notebook 03 se
   hizo con esas filas de train. Recalcularla invalidaría esa selección en silencio.
2. **Una UPM nueva se asigna por hash de su clave**, no al azar. El mismo UPM cae
   siempre en la misma partición, en cualquier máquina y en cualquier corrida, sin
   guardar semilla ni estado. Y como la regla 1 manda sobre ésta, una UPM que fue train
   en 2024 no puede aparecer en el test de 2026: no hay fuga entre ediciones.

Proporciones: las del split original (6,764 / 1,691 / 2,114 UPM ≈ 64 / 16 / 20 %).
"""

from __future__ import annotations

import hashlib

import pandas as pd

CORTES = (("train", 0.64), ("validation", 0.80), ("test", 1.0))


def particion_de(upm: str) -> str:
    u = int(hashlib.sha256(upm.encode("utf-8")).hexdigest()[:12], 16) / 16**12
    return next(nombre for nombre, corte in CORTES if u < corte)


def extender(upms, existente: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Devuelve (partición completa, cuántas UPM nuevas se asignaron)."""
    conocidas = set(existente["upm"])
    nuevas = sorted(set(upms) - conocidas)
    if not nuevas:
        return existente, 0
    extra = pd.DataFrame({"upm": nuevas, "particion": [particion_de(u) for u in nuevas]})
    return pd.concat([existente, extra], ignore_index=True), len(nuevas)
