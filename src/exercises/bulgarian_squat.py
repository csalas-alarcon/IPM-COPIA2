"""
exercises/bulgarian_squat.py  ·  Sentadilla búlgara (Bulgarian split squat).

Reposo: de pie con el pie trasero en un banco. Máximo: rodilla delantera a ~90º.
Se mide sólo la pierna adelantada, que se detecta automáticamente.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from exercises.base import Exercise, FeatureSpec
from vision.pose import SIDES, PoseFrame, distance, side_landmark


# =============================================================================
# Clases
# =============================================================================
class BulgarianSquat(Exercise):
    # --- Definición ----------------------------------------------------------
    key = "bulgarian_squat"
    name = "Sentadilla búlgara"
    instructions = (
        "Apoya el empeine del pie trasero en un banco o silla.",
        "Baja flexionando la pierna delantera hasta ~90º.",
        "Sube empujando con el talón delantero, torso erguido.",
    )
    camera_tip = "Cámara de lado, a la altura de la cadera: que se vea el cuerpo entero."
    rest_hint = "De pie, con el pie trasero apoyado en el banco"
    visibility_hint = "No veo bien tus piernas: colócate de lado a la cámara"
    rom_feedback = "Baja más: rodilla delantera a 90º"
    features = (
        # Medidos sobre la pierna ADELANTADA
        FeatureSpec("rodilla", ("HIP", "KNEE", "ANKLE"), rest=170.0, peak=90.0, weight=1.0),
        FeatureSpec("cadera", ("SHOULDER", "HIP", "KNEE"), rest=170.0, peak=100.0, weight=0.6),
    )
    required_joints = ("SHOULDER", "HIP", "KNEE", "ANKLE")
    bilateral = False
    form_messages = {
        "torso": "Mantén el torso más erguido",
    }

    def __init__(self) -> None:
        super().__init__()
        self._front: str | None = None

    # --- Estado (pierna adelantada) ------------------------------------------
    def reset(self) -> None:
        self._front = None

    def _front_leg(self, pose: PoseFrame) -> str:
        """La pierna delantera es la que tiene el tobillo más bajo en la imagen
        (el pie trasero está elevado). Con histéresis para que no parpadee."""
        ly = pose.point(side_landmark("LEFT", "ANKLE"))[1]
        ry = pose.point(side_landmark("RIGHT", "ANKLE"))[1]
        leg = max(
            distance(pose.point(side_landmark(s, "HIP")), pose.point(side_landmark(s, "ANKLE")))
            for s in SIDES
        )
        margin = 0.08 * leg
        if ly - ry > margin:
            self._front = "LEFT"
        elif ry - ly > margin:
            self._front = "RIGHT"
        elif self._front is None:
            self._front = max(SIDES, key=lambda s: self.side_visibility(pose, s))
        return self._front

    # --- Medición ------------------------------------------------------------
    def active_sides(self, pose: PoseFrame, min_vis: float) -> list[str]:
        front = self._front_leg(pose)
        return [front] if self.side_visibility(pose, front) >= min_vis else []

    # --- Postura -------------------------------------------------------------
    def check_posture(self, pose: PoseFrame, sides: tuple[str, ...]) -> tuple[bool, str]:
        if self._torso_lean(pose, sides) > 55.0:
            return False, "Ponte de pie, con el torso erguido"
        return True, ""

    # --- Técnica -------------------------------------------------------------
    def form_penalties(self, pose: PoseFrame, sides: tuple[str, ...], progress: float) -> dict[str, float]:
        return {"torso": self._ramp(self._torso_lean(pose, sides), 30.0, 50.0)}
