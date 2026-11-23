"""
exercises/base.py  ·  Definición genérica de un ejercicio.

Cada ejercicio describe dos poses de referencia mediante ángulos articulares:
  * reposo (rest)            -> progreso 0
  * máximo esfuerzo (peak)   -> progreso 1

Los ángulos se usan en lugar de las coordenadas crudas porque no dependen de
la distancia a la cámara ni de la posición de la persona en el encuadre.

El progreso se calcula según lo cerca que está la pose actual de cada
referencia:   p = d(reposo) / (d(reposo) + d(máximo))

Las referencias de cada clase son las "estándar". Con la calibración
(repetición de prueba, ver analysis/calibration.py) se sustituyen por las del
propio usuario con set_reference(); reset_reference() vuelve a las estándar.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import numpy as np

from vision.pose import SIDES, Landmark, PoseFrame, angle_to_vertical, distance, side_landmark


# =============================================================================
# Constantes
# =============================================================================
NOSE = int(Landmark.NOSE)


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    joints: tuple[str, str, str]   # Articulaciones sin lado: ("SHOULDER", "ELBOW", "WRIST")
    rest: float                    # Ángulo en reposo (grados)
    peak: float                    # Ángulo en máximo esfuerzo (grados)
    weight: float = 1.0


@dataclass
class Measurement:
    features: np.ndarray           # Ángulos (grados), uno por FeatureSpec
    sides: tuple[str, ...]         # Lados usados para medir


# =============================================================================
# Clases
# =============================================================================
class Exercise(ABC):
    key: str = ""
    name: str = ""
    instructions: tuple[str, ...] = ()
    camera_tip: str = ""
    rest_hint: str = ""            # Cómo colocarse en la posición inicial
    visibility_hint: str = ""
    rom_feedback: str = "Haz el recorrido completo"
    features: tuple[FeatureSpec, ...] = ()
    required_joints: tuple[str, ...] = ()
    bilateral: bool = True         # Promedia ambos lados si son visibles
    form_messages: dict[str, str] = {}

    def __init__(self) -> None:
        if not self.features:
            raise ValueError(f"{type(self).__name__} no define features")
        self.default_rest = np.array([f.rest for f in self.features], dtype=float)
        self.default_peak = np.array([f.peak for f in self.features], dtype=float)
        w = np.array([f.weight for f in self.features], dtype=float)
        self.weights = w / w.sum()
        self.reset_reference()

    # --- Poses de referencia (estándar o calibradas) -------------------------
    def set_reference(self, rest: np.ndarray, peak: np.ndarray,
                      profile: tuple[np.ndarray, np.ndarray] | None = None,
                      calibrated: bool = True) -> None:
        """Fija las poses de referencia (ángulos) y, opcionalmente, la trayectoria
        de referencia (perfil de ida y de vuelta, ver analysis/scoring.py)."""
        rest = np.array(rest, dtype=float)
        peak = np.array(peak, dtype=float)
        span = peak - rest
        if np.any(np.abs(span) < 1e-6):
            raise ValueError("rest y peak deben ser distintos en cada feature")
        self.rest_vector, self.peak_vector, self._span = rest, peak, span
        self.reference_profile = profile
        self.calibrated = calibrated

    def reset_reference(self) -> None:
        """Vuelve a las referencias estándar del ejercicio."""
        self.set_reference(self.default_rest, self.default_peak, None, calibrated=False)

    # --- Estado interno (p. ej. pierna adelantada en la sentadilla búlgara) --
    def reset(self) -> None:
        pass

    # --- Medición ------------------------------------------------------------
    def side_visibility(self, pose: PoseFrame, side: str) -> float:
        return pose.mean_visibility(side_landmark(side, j) for j in self.required_joints)

    def active_sides(self, pose: PoseFrame, min_vis: float) -> list[str]:
        scored = sorted(((self.side_visibility(pose, s), s) for s in SIDES), reverse=True)
        visible = [s for v, s in scored if v >= min_vis]
        return visible if self.bilateral else visible[:1]

    def measure(self, pose: PoseFrame, min_vis: float) -> Measurement | None:
        sides = self.active_sides(pose, min_vis)
        if not sides:
            return None
        per_side = [
            [pose.angle(*(side_landmark(s, j) for j in f.joints)) for f in self.features]
            for s in sides
        ]
        return Measurement(np.mean(np.array(per_side), axis=0), tuple(sides))

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """0 = ángulo de reposo, 1 = ángulo de máximo esfuerzo (por feature)."""
        return (np.asarray(features, dtype=float) - self.rest_vector) / self._span

    def progress(self, features: np.ndarray) -> float:
        u = self.normalize(features)
        d_rest = np.sqrt(np.sum(self.weights * u ** 2))
        d_peak = np.sqrt(np.sum(self.weights * (u - 1.0) ** 2))
        total = d_rest + d_peak
        return float(d_rest / total) if total > 1e-9 else 0.0

    # --- Postura y técnica (a sobreescribir) ---------------------------------
    def check_posture(self, pose: PoseFrame, sides: tuple[str, ...]) -> tuple[bool, str]:
        """¿Está la persona en la posición del ejercicio? (si no, no se cuentan reps)."""
        return True, ""

    def form_penalties(self, pose: PoseFrame, sides: tuple[str, ...], progress: float) -> dict[str, float]:
        """Penalizaciones de técnica por frame: {regla: 0 (perfecto) .. 1 (mal)}."""
        return {}

    # --- Utilidades para las subclases ---------------------------------------
    @staticmethod
    def _mid(pose: PoseFrame, joint: str, sides: tuple[str, ...]) -> np.ndarray:
        return np.mean([pose.point(side_landmark(s, joint)) for s in sides], axis=0)

    def _torso_lean(self, pose: PoseFrame, sides: tuple[str, ...]) -> float:
        """Inclinación del tronco (cadera -> hombro) respecto a la vertical de la imagen."""
        return angle_to_vertical(self._mid(pose, "SHOULDER", sides) - self._mid(pose, "HIP", sides))

    def _torso_length(self, pose: PoseFrame, sides: tuple[str, ...]) -> float:
        return max(distance(self._mid(pose, "SHOULDER", sides), self._mid(pose, "HIP", sides)), 1.0)

    @staticmethod
    def _angle(pose: PoseFrame, side: str, a: str, b: str, c: str) -> float:
        return pose.angle(side_landmark(side, a), side_landmark(side, b), side_landmark(side, c))

    @staticmethod
    def _ramp(value: float, ok: float, bad: float) -> float:
        """0 si value está en 'ok', 1 si llega a 'bad', lineal entre medias."""
        if bad == ok:
            return 0.0
        return float(np.clip((value - ok) / (bad - ok), 0.0, 1.0))

    def _asymmetry(self, pose: PoseFrame, sides: tuple[str, ...], joints: tuple[str, str, str],
                   tolerance: float = 15.0, bad: float = 40.0) -> float | None:
        if len(sides) < 2:
            return None
        diff = abs(self._angle(pose, "LEFT", *joints) - self._angle(pose, "RIGHT", *joints))
        return self._ramp(diff, tolerance, bad)

    @staticmethod
    def _visible(pose: PoseFrame, idx: int, min_vis: float = 0.5) -> bool:
        return pose.vis(idx) >= min_vis
