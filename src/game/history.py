"""
Lectura de las partidas guardadas en results/ para el historial y para comparar
con la partida anterior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class HistoryEntry:
    path: Path
    date: datetime
    total_points: int
    reps: int
    average_score: float                 # Media de las repeticiones de la partida
    exercises: list[str] = field(default_factory=list)   # Nombres de los ejercicios hechos


def load_history(results_dir: Path) -> list[HistoryEntry]:
    """Partidas guardadas, de la más antigua a la más reciente."""
    entries: list[HistoryEntry] = []
    for path in sorted(Path(results_dir).glob("session_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            date = datetime.fromisoformat(data["date"])
            points = int(data.get("total_points", 0))
            scores = []
            names = []
            for exercise in data.get("exercises", []):
                if exercise.get("skipped") and not exercise.get("reps"):
                    continue
                names.append(exercise.get("name", "?"))
                for rep in exercise.get("repetitions", []):
                    scores.append(rep["total"])
        except (OSError, ValueError, KeyError, TypeError):
            continue    # Archivo corrupto o de otra versión: se ignora
        average = sum(scores) / len(scores) if scores else 0.0
        entries.append(HistoryEntry(path, date, points, len(scores), average, names))
    entries.sort(key=lambda e: e.date)
    return entries
