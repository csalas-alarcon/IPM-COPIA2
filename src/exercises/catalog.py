"""
exercises/catalog.py  ·  Catálogo de ejercicios.

Para añadir un ejercicio nuevo, crea su archivo en exercises/ y regístralo
en EXERCISES.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from exercises.base import Exercise
from exercises.bulgarian_squat import BulgarianSquat
from exercises.pullup import PullUp
from exercises.pushup import PushUp


# =============================================================================
# Constantes
# =============================================================================
EXERCISES: dict[str, type[Exercise]] = {
    PullUp.key: PullUp,
    PushUp.key: PushUp,
    BulgarianSquat.key: BulgarianSquat,
}


# =============================================================================
# Funciones
# =============================================================================
def build_exercises(keys) -> list[Exercise]:
    unknown = [k for k in keys if k not in EXERCISES]
    if unknown:
        raise ValueError(f"Ejercicios desconocidos: {unknown}. Opciones: {list(EXERCISES)}")
    return [EXERCISES[k]() for k in keys]
