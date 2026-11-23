"""
game/history.py  ·  Historial de partidas.

Lee las sesiones guardadas en results/*.json (ver session.py) para la
pantalla de historial y para comparar la partida actual con la anterior.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass
class HistoryEntry:
    path: Path
    date: datetime
    total_points: int
    reps: int
    average_score: float                 # Media de las repeticiones de la partida
    exercises: list[str] = field(default_factory=list)   # Nombres de los ejercicios hechos


# =============================================================================
# Funciones
# =============================================================================
def load_history(results_dir: Path, limit: int | None = None) -> list[HistoryEntry]:
    """Partidas guardadas, de la más antigua a la más reciente (las `limit` últimas)."""
    entries: list[HistoryEntry] = []
    for path in sorted(Path(results_dir).glob("session_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            date = datetime.fromisoformat(data["date"])
            done = [e for e in data.get("exercises", []) if not e.get("skipped") or e.get("reps")]
            scores = [r["total"] for e in done for r in e.get("repetitions", [])]
            entries.append(HistoryEntry(
                path, date, int(data.get("total_points", 0)), len(scores),
                sum(scores) / len(scores) if scores else 0.0, [e.get("name", "?") for e in done]))
        except (OSError, ValueError, KeyError, TypeError):
            continue    # Archivo corrupto o de otra versión: se ignora
    entries.sort(key=lambda e: e.date)
    return entries[-limit:] if limit else entries
