"""
analysis/calibration.py  ·  Calibración con una repetición de prueba.

Antes de cada ejercicio el usuario hace una repetición de prueba, lenta y
completa. De ella se sacan SUS referencias:
  * reposo  : mediana de los ángulos mientras espera quieto (instrucciones y
              cuenta atrás);
  * máximo  : ángulos en el punto de máximo esfuerzo de la repetición;
  * trayectoria : cómo ha ido de una pose a otra (perfil de ida y de vuelta).

Las repeticiones siguientes se comparan con esas referencias (como en los
juegos de "dibuja un círculo", pero el círculo de referencia es el tuyo).

Para que una repetición de prueba mala no estropee la partida:
  * la repetición se detecta con las referencias estándar y tiene que llegar
    al menos a `calibration_min_progress` (si no, se pide repetirla);
  * cada ángulo tiene que moverse al menos `calibration_min_span` veces su
    recorrido estándar y en el mismo sentido; si no, ese ángulo usa la
    referencia estándar.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.scoring import build_profile
from config import GameConfig, ScoringConfig
from exercises.base import Exercise


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass
class CalibrationResult:
    rest: np.ndarray
    peak: np.ndarray
    profile: tuple[np.ndarray, np.ndarray]
    fallback: tuple[str, ...]          # Features que se quedan con la referencia estándar

    def apply(self, exercise: Exercise) -> None:
        exercise.set_reference(self.rest, self.peak, self.profile, calibrated=True)


# =============================================================================
# Clases
# =============================================================================
class Calibrator:
    def __init__(self, exercise: Exercise, game_cfg: GameConfig, scoring_cfg: ScoringConfig):
        self.exercise = exercise
        self.game_cfg = game_cfg
        self.scoring_cfg = scoring_cfg
        self.reset()

    # --- Estado --------------------------------------------------------------
    def reset(self) -> None:
        self._rest: list[np.ndarray] = []
        self._restart()
        self.result: CalibrationResult | None = None

    def _restart(self) -> None:
        self._times: list[float] = []
        self._feats: list[np.ndarray] = []
        self._progress: list[float] = []
        self._last_rest: tuple[float, np.ndarray, float] | None = None
        self.max_progress = 0.0

    @property
    def recording(self) -> bool:
        return bool(self._times)

    # --- Entrada -------------------------------------------------------------
    def add_rest(self, features: np.ndarray) -> None:
        """Ángulos con el usuario quieto en reposo (antes de la repetición de prueba)."""
        self._rest.append(np.array(features, dtype=float))
        if len(self._rest) > 300:
            self._rest.pop(0)

    def update(self, t: float, features: np.ndarray, progress: float) -> str | None:
        """Un frame de la repetición de prueba (progreso con las referencias estándar).

        Devuelve "done" (hay resultado), "retry" (repetición demasiado corta) o None.
        """
        rest_th = self.scoring_cfg.rest_threshold
        f = np.array(features, dtype=float)
        if not self.recording:
            if progress > rest_th:
                if self._last_rest is not None:
                    self._append(*self._last_rest)
                self._append(t, f, progress)
            else:
                self._last_rest = (t, f, progress)
            return None

        self._append(t, f, progress)
        if t - self._times[0] > self.scoring_cfg.abandon_rep_seconds:
            self._restart()
            return None
        if progress > rest_th:
            return None

        # Vuelta al reposo: ¿ha sido una repetición válida?
        if self.max_progress >= self.game_cfg.calibration_min_progress:
            self.result = self._compute()
            if self.result is not None:
                return "done"
        retry = self.max_progress >= self.scoring_cfg.partial_threshold * 0.5
        self._restart()
        self._last_rest = (t, f, progress)
        return "retry" if retry else None

    def _append(self, t: float, f: np.ndarray, p: float) -> None:
        self._times.append(float(t))
        self._feats.append(f)
        self._progress.append(float(p))
        self.max_progress = max(self.max_progress, float(p))

    # --- Resultado -----------------------------------------------------------
    def _compute(self) -> CalibrationResult | None:
        ex = self.exercise
        feats = np.stack(self._feats)
        prog = np.array(self._progress)
        rest = np.median(np.stack(self._rest), axis=0) if len(self._rest) >= 3 else feats[0].copy()
        top = prog >= prog.max() - 0.03                  # Frames en el máximo (promediados)
        peak = feats[top].mean(axis=0)

        default_span = ex.default_peak - ex.default_rest
        span = peak - rest
        ok = (np.sign(span) == np.sign(default_span)) & \
             (np.abs(span) >= self.game_cfg.calibration_min_span * np.abs(default_span))
        if not ok[int(np.argmax(ex.weights))]:
            return None                                  # El ángulo principal no se ha movido bien
        rest = np.where(ok, rest, ex.default_rest)
        peak = np.where(ok, peak, ex.default_peak)

        u = (feats - rest) / (peak - rest)
        times = np.array(self._times)
        if len(times) < 4 or times[-1] - times[0] <= 0:
            return None
        profile = build_profile(times, u, ex.weights, self.scoring_cfg.trajectory_samples)
        fallback = tuple(f.name for f, good in zip(ex.features, ok) if not good)
        return CalibrationResult(rest, peak, profile, fallback)
