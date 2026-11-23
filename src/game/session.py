"""
game/session.py  ·  Resultados de la sesión.

ExerciseSession guarda las repeticiones de un ejercicio y calcula sus
estadísticas; save_sessions() escribe la sesión completa en un JSON.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from analysis.scoring import RepScore
from exercises.base import Exercise


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass
class ExerciseSession:
    exercise: Exercise
    reps: list[RepScore] = field(default_factory=list)
    partial_reps: int = 0
    skipped: bool = False
    duration: float = 0.0
    calibrated: bool = False                 # ¿Se usaron las referencias de la repetición de prueba?
    reference: dict | None = None            # {"rest": {feature: grados}, "peak": {...}}

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def points(self) -> int:
        """Puntos del ejercicio = suma de la puntuación de cada rep."""
        return int(round(sum(r.total for r in self.reps)))

    @property
    def average(self) -> float:
        return float(np.mean([r.total for r in self.reps])) if self.reps else 0.0

    @property
    def best(self) -> float:
        return max((r.total for r in self.reps), default=0.0)

    def component_average(self, name: str) -> float:
        return float(np.mean([getattr(r, name) for r in self.reps])) if self.reps else 0.0

    @property
    def best_rep(self):
        return max(self.reps, key=lambda r: r.total) if self.reps else None

    def top_feedback(self, n: int = 2) -> list[str]:
        c = Counter(msg for r in self.reps for msg in r.feedback)
        return [msg for msg, _ in c.most_common(n)]

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise.key,
            "name": self.exercise.name,
            "skipped": self.skipped,
            "reps": self.count,
            "partial_reps": self.partial_reps,
            "points": self.points,
            "average_score": round(self.average, 2),
            "best_score": round(self.best, 2),
            "duration_s": round(self.duration, 2),
            "calibrated": self.calibrated,
            "reference": self.reference,
            "repetitions": [r.to_dict() for r in self.reps],
        }


# =============================================================================
# Funciones
# =============================================================================
def save_sessions(sessions: list[ExerciseSession], results_dir: Path) -> Path:
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"session_{stamp}.json"
    data = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "total_points": sum(s.points for s in sessions),
        "exercises": [s.to_dict() for s in sessions],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
