"""
vision/camera.py  ·  Captura de vídeo.

Lee frames de una webcam (índice 0, 1...) o de la cámara del móvil
(stream por URL, p. ej. IP Webcam / DroidCam).
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import cv2
import numpy as np

from config import CameraConfig


# =============================================================================
# Funciones
# =============================================================================
def parse_source(source: int | str) -> int | str:
    """'0' -> 0 ; 'http://...' se deja tal cual."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


# =============================================================================
# Clases
# =============================================================================
class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.source = parse_source(cfg.source)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara '{self.source}'. Comprueba que está conectada, "
                "que ningún otro programa la usa, o prueba otro índice con --source 1."
            )
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._failed = 0

    # --- Lectura -------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Devuelve un frame BGR (ya espejado si procede) o None si falló la lectura."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failed += 1
            if self._failed >= self.cfg.max_failed_reads:
                raise RuntimeError(f"La cámara '{self.source}' ha dejado de enviar imágenes.")
            return None
        self._failed = 0
        return cv2.flip(frame, 1) if self.cfg.mirror else frame

    # --- Cierre (también usable con "with") ----------------------------------
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
