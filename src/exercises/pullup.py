"""
Dominada (pull-up).

Reposo: colgado con los brazos estirados. Máximo: barbilla sobre la barra.
"""

from __future__ import annotations

import numpy as np

from exercises.base import NOSE, Exercise, FeatureSpec
from vision.pose import PoseFrame, side_landmark


class PullUp(Exercise):
    key = "pullup"
    name = "Dominada"
    instructions = (
        "Cuélgate de la barra con los brazos totalmente extendidos.",
        "Tira hasta que la barbilla supere la barra.",
        "Baja de forma controlada hasta estirar los brazos.",
    )
    camera_tip = "Cámara de frente o de lado: que se vea de las manos a las rodillas."
    rest_hint = "Cuélgate con los brazos estirados"
    visibility_hint = "No veo bien tus brazos y tronco (manos, codos, hombros y cadera)"
    rom_feedback = "Sube más: la barbilla debe pasar la barra"
    features = (
        FeatureSpec("codo", ("SHOULDER", "ELBOW", "WRIST"), rest=165.0, peak=55.0, weight=1.0),
        FeatureSpec("hombro", ("HIP", "SHOULDER", "ELBOW"), rest=165.0, peak=45.0, weight=0.7),
    )
    required_joints = ("SHOULDER", "ELBOW", "WRIST", "HIP")
    bilateral = True
    form_messages = {
        "barbilla": "Lleva la barbilla por encima de las manos",
        "simetria": "Tira por igual con los dos brazos",
        "balanceo": "Evita balancearte o subir las rodillas",
    }

    def check_posture(self, pose: PoseFrame, sides: tuple[str, ...]) -> tuple[bool, str]:
        wrist_y = self._mid(pose, "WRIST", sides)[1]
        shoulder_y = self._mid(pose, "SHOULDER", sides)[1]
        # La y de la imagen crece hacia abajo: las manos deben quedar por encima de los hombros
        if wrist_y > shoulder_y + 0.15 * self._torso_length(pose, sides):
            return False, "Agárrate a la barra: manos por encima de la cabeza"
        return True, ""

    def form_penalties(self, pose: PoseFrame, sides: tuple[str, ...], progress: float) -> dict[str, float]:
        pens: dict[str, float] = {}
        torso = self._torso_length(pose, sides)
        # La barbilla solo se evalúa cerca de la parte alta
        if progress >= 0.8 and self._visible(pose, NOSE):
            wrist_y = float(np.min([pose.point(side_landmark(s, "WRIST"))[1] for s in sides]))
            nose_y = pose.point(NOSE)[1]
            pens["barbilla"] = self._ramp(nose_y - wrist_y, 0.0, 0.3 * torso)

        sym = self._asymmetry(pose, sides, ("SHOULDER", "ELBOW", "WRIST"))
        if sym is not None:
            pens["simetria"] = sym

        # Flexión de cadera: balanceo (kipping) o rodillas al pecho
        if all(self._visible(pose, side_landmark(s, "KNEE")) for s in sides):
            hip = float(np.mean([self._angle(pose, s, "SHOULDER", "HIP", "KNEE") for s in sides]))
            pens["balanceo"] = self._ramp(hip, 150.0, 115.0)
        return pens
