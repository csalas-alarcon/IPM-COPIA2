"""
ui/recorder.py  ·  Grabación de la ventana del juego en un vídeo.

Guarda en un .mp4 exactamente lo que se ve en la ventana (vídeo o cámara con
el esqueleto, el marcador, las gráficas...). Sirve de plan B para la
presentación: si el equipo va lento al grabar la pantalla, se reproduce este
vídeo.

El vídeo se escribe a fps fijos según el reloj real: si el juego dibuja menos
frames por segundo, se repiten; si dibuja más, se descartan. Así la grabación
dura lo mismo que la partida.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Clases
# =============================================================================
class Recorder:
    def __init__(self, path: str | Path, fps: float = 30.0):
        self.path = Path(path).expanduser().resolve()
        self.fps = float(fps)
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._t0 = 0.0
        self._written = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def elapsed(self) -> float:
        return self._written / self.fps

    def _open(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._size = (w - w % 2, h - h % 2)       # Los códecs prefieren dimensiones pares
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, self._size)
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"No se pudo crear el vídeo {self.path}")
        self._t0 = time.monotonic()
        self._written = 0
        print(f"[grabación] Grabando en {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._open(frame)
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        due = int((time.monotonic() - self._t0) * self.fps) + 1
        for _ in range(min(due - self._written, int(self.fps * 2))):
            self._writer.write(frame)
        self._written = max(self._written, due)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[grabación] Guardada en {self.path} ({self.elapsed:.0f} s)")
