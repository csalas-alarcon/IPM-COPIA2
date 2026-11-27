"""
Catálogo de ejercicios. Para añadir uno, crea su archivo y regístralo en EXERCISES.
"""

from __future__ import annotations

from exercises.base import Exercise
from exercises.bulgarian_squat import BulgarianSquat
from exercises.pullup import PullUp
from exercises.pushup import PushUp


EXERCISES: dict[str, type[Exercise]] = {
    PullUp.key: PullUp,
    PushUp.key: PushUp,
    BulgarianSquat.key: BulgarianSquat,
}


def build_exercises(keys) -> list[Exercise]:
    unknown = [k for k in keys if k not in EXERCISES]
    if unknown:
        raise ValueError(f"Ejercicios desconocidos: {unknown}. Opciones: {list(EXERCISES)}")
    return [EXERCISES[k]() for k in keys]
