"""
Calibración con una repetición de prueba, lenta y completa, de la que salen las
referencias del propio usuario:
  * reposo: mediana de los ángulos mientras espera quieto;
  * máximo: ángulos en el punto de máximo esfuerzo;
  * trayectoria: perfil de ida y de vuelta entre las dos poses.

Para que una prueba mala no estropee la partida:
  * la repetición se detecta con las referencias estándar y tiene que llegar a
    `calibration_min_progress`; si no, se pide repetirla;
  * cada ángulo tiene que moverse en el mismo sentido que el estándar y al
    menos `calibration_min_span` veces su recorrido; si no, usa el estándar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.scoring import build_profile
from config import GameConfig, ScoringConfig
from exercises.base import Exercise

_MAX_REST_SAMPLES = 300


@dataclass
class CalibrationResult:
    rest: np.ndarray
    peak: np.ndarray
    profile: tuple[np.ndarray, np.ndarray]
    fallback: tuple[str, ...]          # Ángulos que se quedan con la referencia estándar

    def apply(self, exercise: Exercise) -> None:
        exercise.set_reference(self.rest, self.peak, self.profile, calibrated=True)


class Calibrator:
    def __init__(self, exercise: Exercise, game_cfg: GameConfig, scoring_cfg: ScoringConfig):
        self.exercise = exercise
        self.game_cfg = game_cfg
        self.scoring_cfg = scoring_cfg
        self.reset()

    def reset(self) -> None:
        self._rest: list[np.ndarray] = []
        self._restart()
        self.result: CalibrationResult | None = None

    def _restart(self) -> None:
        self._times: list[float] = []
        self._feats: list[np.ndarray] = []
        self._progress: list[float] = []
        self._last_rest: tuple[float, np.ndarray, float] | None = None
        self._lowest: tuple[float, np.ndarray] | None = None   # (progreso, ángulos) del frame más en reposo
        self.max_progress = 0.0

    @property
    def recording(self) -> bool:
        return bool(self._times)

    def add_rest(self, features: np.ndarray) -> None:
        """Guarda los ángulos con el usuario quieto en reposo, antes de la repetición."""
        self._rest.append(np.array(features, dtype=float))
        if len(self._rest) > _MAX_REST_SAMPLES:
            self._rest.pop(0)

    def update(self, t: float, features: np.ndarray, progress: float) -> str | None:
        """Procesa un frame de la repetición de prueba (progreso con las referencias estándar).

        Devuelve "done" si ya hay resultado, "retry" si la repetición fue demasiado
        corta, o None.
        """
        rest_threshold = self.scoring_cfg.rest_threshold
        f = np.array(features, dtype=float)
        if not self.recording:
            if progress > rest_threshold:
                if self._last_rest is not None:
                    self._append(*self._last_rest)
                self._append(t, f, progress)
            else:
                self._last_rest = (t, f, progress)
                self._note_lowest(f, progress)
            return None

        self._append(t, f, progress)
        if t - self._times[0] > self.scoring_cfg.abandon_rep_seconds:
            self._restart()
            return None
        if progress > rest_threshold:
            return None

        # Ha vuelto al reposo: ¿fue una repetición válida?
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
        self._note_lowest(f, p)

    def _note_lowest(self, f: np.ndarray, p: float) -> None:
        if self._lowest is None or p < self._lowest[0]:
            self._lowest = (float(p), f)

    def _compute(self) -> CalibrationResult | None:
        ex = self.exercise
        feats = np.stack(self._feats)
        progress = np.array(self._progress)

        if len(self._rest) >= 3:
            rest = np.median(np.stack(self._rest), axis=0)
        else:
            # Sin datos de reposo, el frame más cercano al reposo que se haya visto
            rest = self._lowest[1].copy()
        at_peak = progress >= progress.max() - 0.03
        peak = feats[at_peak].mean(axis=0)

        default_span = ex.default_peak - ex.default_rest
        span = peak - rest
        same_direction = np.sign(span) == np.sign(default_span)
        enough_range = np.abs(span) >= self.game_cfg.calibration_min_span * np.abs(default_span)
        ok = same_direction & enough_range
        if not ok[int(np.argmax(ex.weights))]:
            return None                       # El ángulo principal no se ha movido bien
        rest = np.where(ok, rest, ex.default_rest)
        peak = np.where(ok, peak, ex.default_peak)

        times = np.array(self._times)
        if len(times) < 4 or times[-1] - times[0] <= 0:
            return None
        u = (feats - rest) / (peak - rest)
        profile = build_profile(times, u, ex.weights, self.scoring_cfg.trajectory_samples)
        fallback = tuple(spec.name for spec, good in zip(ex.features, ok) if not good)
        return CalibrationResult(rest, peak, profile, fallback)
