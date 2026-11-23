"""
vision/pose.py  ·  Representación de una pose.

Contiene:
  * Landmark / side_landmark / POSE_CONNECTIONS: los 33 puntos del modelo.
  * joint_angle / angle_to_vertical / distance: geometría sobre puntos 2D/3D.
  * PoseFrame: la pose detectada en un frame (puntos, visibilidad, 3D).
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


# =============================================================================
# Landmarks (índices y conexiones del Pose Landmarker)
# =============================================================================
class Landmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


SIDES: tuple[str, str] = ("LEFT", "RIGHT")

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
)


def side_landmark(side: str, joint: str) -> int:
    """side_landmark('LEFT', 'ELBOW') -> 13"""
    return int(Landmark[f"{side}_{joint}"])


# =============================================================================
# Geometría
# =============================================================================
def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo (grados, 0-180) en el vértice b formado por los segmentos b-a y b-c."""
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    cos = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_to_vertical(vec: np.ndarray) -> float:
    """Inclinación (grados, 0-90) de un vector 2D de imagen respecto a la vertical."""
    v = np.asarray(vec, dtype=float)[:2]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(abs(v[1]) / n, 0.0, 1.0))))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# =============================================================================
# PoseFrame
# =============================================================================
@dataclass(frozen=True)
class PoseFrame:
    image_xy: np.ndarray            # (33, 2) coordenadas en píxeles del frame mostrado
    visibility: np.ndarray          # (33,) visibilidad 0-1
    world_xyz: np.ndarray | None    # (33, 3) metros, centrado en la cadera
    timestamp_ms: int
    use_world: bool = True

    def vis(self, idx: int) -> float:
        return float(self.visibility[idx])

    def mean_visibility(self, idxs: Iterable[int]) -> float:
        idxs = list(idxs)
        return float(np.mean(self.visibility[idxs])) if idxs else 0.0

    def point(self, idx: int) -> np.ndarray:
        """Punto de imagen (píxeles)."""
        return self.image_xy[idx]

    def angle(self, a: int, b: int, c: int) -> float:
        """Ángulo en b. Usa coordenadas 3D world si están disponibles."""
        pts = self.world_xyz if (self.use_world and self.world_xyz is not None) else self.image_xy
        return joint_angle(pts[a], pts[b], pts[c])
