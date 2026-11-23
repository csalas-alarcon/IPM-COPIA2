"""
exercises/pushup.py  ·  Flexión (push-up).

Reposo: plancha con los brazos estirados. Máximo: codos a ~90º.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from exercises.base import Exercise, FeatureSpec
from vision.pose import PoseFrame


# =============================================================================
# Clases
# =============================================================================
class PushUp(Exercise):
    # --- Definición ----------------------------------------------------------
    key = "pushup"
    name = "Flexión"
    instructions = (
        "Colócate en plancha con los brazos extendidos.",
        "Baja el pecho doblando los codos hasta ~90º o más.",
        "Mantén el cuerpo recto (sin hundir ni subir la cadera).",
    )
    camera_tip = "Cámara de lado, a la altura del suelo: que se vea el cuerpo entero."
    rest_hint = "Ponte en plancha con los brazos estirados"
    visibility_hint = "No te veo entero: colócate de lado a la cámara"
    rom_feedback = "Baja más: dobla los codos hasta 90º"
    features = (
        FeatureSpec("codo", ("SHOULDER", "ELBOW", "WRIST"), rest=165.0, peak=80.0, weight=1.0),
    )
    required_joints = ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE")
    bilateral = True
    form_messages = {
        "cuerpo": "Mantén el cuerpo recto: cadera alineada",
        "rodillas": "Mantén las piernas estiradas",
        "simetria": "Empuja por igual con los dos brazos",
    }

    # --- Postura -------------------------------------------------------------
    def check_posture(self, pose: PoseFrame, sides: tuple[str, ...]) -> tuple[bool, str]:
        if self._torso_lean(pose, sides) < 45.0:
            return False, "Colócate en plancha: cuerpo horizontal"
        return True, ""

    # --- Técnica -------------------------------------------------------------
    def form_penalties(self, pose: PoseFrame, sides: tuple[str, ...], progress: float) -> dict[str, float]:
        body = min(self._angle(pose, s, "SHOULDER", "HIP", "ANKLE") for s in sides)
        knee = min(self._angle(pose, s, "HIP", "KNEE", "ANKLE") for s in sides)
        pens = {
            "cuerpo": self._ramp(-body, -165.0, -140.0),
            "rodillas": self._ramp(-knee, -160.0, -130.0),
        }
        sym = self._asymmetry(pose, sides, ("SHOULDER", "ELBOW", "WRIST"))
        if sym is not None:
            pens["simetria"] = sym
        return pens
