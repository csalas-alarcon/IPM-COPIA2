"""
Envoltorio del Pose Landmarker de MediaPipe (modo VIDEO): recibe un frame BGR
y devuelve un PoseFrame, o None si no hay nadie.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import ModelConfig
from vision.pose import PoseFrame


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class PoseDetector:
    def __init__(self, model_path: Path, cfg: ModelConfig):
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe el modelo: {model_path}")
        self.cfg = cfg
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=cfg.num_poses,
            min_pose_detection_confidence=cfg.min_pose_detection_confidence,
            min_pose_presence_confidence=cfg.min_pose_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        self._offset = 0

    def _next_timestamp(self, timestamp_ms: int | None = None) -> int:
        # El modo VIDEO exige timestamps estrictamente crecientes. En vivo se usa
        # el reloj real; con un vídeo, el suyo más un desplazamiento que crece
        # al rebobinar (así el seguimiento no se rompe).
        if timestamp_ms is None:
            ts = int(time.monotonic() * 1000)
        else:
            ts = int(timestamp_ms) + self._offset
        if ts <= self._last_ts:
            if timestamp_ms is not None:
                self._offset += self._last_ts + 1 - ts
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> PoseFrame | None:
        """Detecta la pose. timestamp_ms es el instante del frame en el vídeo (None = reloj real)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe espera RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._next_timestamp(timestamp_ms)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lms = result.pose_landmarks[0]
        image_xy = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float)
        visibility = np.array([lm.visibility if lm.visibility is not None else 0.0 for lm in lms])
        world = None
        if result.pose_world_landmarks:
            world = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=float)
        return PoseFrame(image_xy, visibility, world, ts, self.cfg.use_world_landmarks)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
