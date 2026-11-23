"""
analysis/counter.py  ·  Conteo de repeticiones.

RepCounter: máquina de estados que cuenta repeticiones a partir del progreso (0..1).

    REPOSO --(p > rest)--> MOVIENDO --(p >= peak)--> PICO --(p <= rest)--> REP ✔
                              |
                              +--(p <= rest sin llegar al pico)--> rep incompleta

ExerciseTracker: une medición + suavizado + comprobación de postura + contador.
Es lo que llama el juego en cada frame.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from analysis.scoring import RepScore, score_repetition
from config import GameConfig, ScoringConfig
from exercises.base import Exercise, Measurement
from vision.pose import PoseFrame


# =============================================================================
# Tipos de datos
# =============================================================================
class RepState(Enum):
    REST = auto()
    MOVING = auto()
    PEAK = auto()


@dataclass
class _Sample:
    t: float
    features: np.ndarray
    progress: float
    penalties: dict[str, float]


@dataclass
class RepEvent:
    kind: str                    # "rep" | "partial"
    max_progress: float
    score: RepScore | None = None


@dataclass
class TrackerState:
    detected: bool = False               # Hay una persona en la imagen
    measurable: bool = False             # Se ven las articulaciones necesarias
    posture_ok: bool = False             # Está en la posición del ejercicio
    progress: float | None = None
    measurement: Measurement | None = None
    penalties: dict[str, float] = field(default_factory=dict)
    issue: str | None = None             # Aviso a mostrar en vivo
    event: RepEvent | None = None


# =============================================================================
# RepCounter
# =============================================================================
class RepCounter:
    def __init__(self, exercise: Exercise, cfg: ScoringConfig):
        self.exercise = exercise
        self.cfg = cfg
        self.reset()

    # --- Estado --------------------------------------------------------------
    def reset(self) -> None:
        self.state = RepState.REST
        self._samples: list[_Sample] = []
        self._last_rest: _Sample | None = None
        self._max_p = 0.0

    def _restart(self, last: _Sample | None) -> None:
        self.state = RepState.REST
        self._samples = []
        self._max_p = 0.0
        self._last_rest = last

    # --- Actualización por frame ---------------------------------------------
    def update(self, t: float, features: np.ndarray, progress: float,
               penalties: dict[str, float]) -> RepEvent | None:
        s = _Sample(t, np.array(features, dtype=float), float(progress), dict(penalties))
        cfg = self.cfg

        if self.state is RepState.REST:
            if s.progress > cfg.rest_threshold:
                # Empieza la repetición: incluimos el último frame en reposo
                self._samples = [self._last_rest] if self._last_rest else []
                self._samples.append(s)
                self._max_p = s.progress
                self.state = RepState.PEAK if s.progress >= cfg.peak_threshold else RepState.MOVING
            else:
                self._last_rest = s
            return None

        self._samples.append(s)
        self._max_p = max(self._max_p, s.progress)
        if self.state is RepState.MOVING and s.progress >= cfg.peak_threshold:
            self.state = RepState.PEAK

        if t - self._samples[0].t > cfg.abandon_rep_seconds:
            self._restart(None)
            return None

        if s.progress <= cfg.rest_threshold:
            event = None
            if self.state is RepState.PEAK:
                score = score_repetition(
                    self.exercise,
                    np.array([x.t for x in self._samples]),
                    np.stack([x.features for x in self._samples]),
                    [x.penalties for x in self._samples],
                    cfg,
                )
                event = RepEvent("rep", self._max_p, score)
            elif self._max_p >= cfg.partial_threshold:
                event = RepEvent("partial", self._max_p)
            self._restart(s)
            return event
        return None


# =============================================================================
# ExerciseTracker
# =============================================================================
class ExerciseTracker:
    def __init__(self, exercise: Exercise, game_cfg: GameConfig, scoring_cfg: ScoringConfig):
        self.exercise = exercise
        self.game_cfg = game_cfg
        self.scoring_cfg = scoring_cfg
        self.counter = RepCounter(exercise, scoring_cfg)
        self._ema: np.ndarray | None = None
        self._last_seen = -1e9

    # --- Estado --------------------------------------------------------------
    def reset(self) -> None:
        self.counter.reset()
        self.exercise.reset()
        self._ema = None

    # --- Actualización por frame ---------------------------------------------
    def update(self, pose: PoseFrame | None, now: float, counting: bool = True) -> TrackerState:
        if pose is None:
            if now - self._last_seen > self.game_cfg.pose_lost_reset:
                self._ema = None
            return TrackerState(issue="No te detecto: colócate frente a la cámara")
        self._last_seen = now

        m = self.exercise.measure(pose, self.game_cfg.min_visibility)
        if m is None:
            return TrackerState(detected=True, issue=self.exercise.visibility_hint)

        # Suavizado exponencial de los ángulos (reduce el ruido del detector)
        a = self.game_cfg.smoothing_alpha
        self._ema = m.features.copy() if self._ema is None else a * m.features + (1 - a) * self._ema
        progress = self.exercise.progress(self._ema)

        ok, posture_msg = self.exercise.check_posture(pose, m.sides)
        penalties = self.exercise.form_penalties(pose, m.sides, progress) if ok else {}
        state = TrackerState(True, True, ok, progress, Measurement(self._ema.copy(), m.sides), penalties)

        if not ok:
            state.issue = posture_msg
            return state

        if penalties:
            rule, worst = max(penalties.items(), key=lambda kv: kv[1])
            if worst >= self.scoring_cfg.live_feedback_threshold:
                state.issue = self.exercise.form_messages.get(rule)

        if counting:
            state.event = self.counter.update(now, self._ema, progress, penalties)
        return state
