"""
vision/camera.py  ·  Captura de vídeo.

Lee frames de:
  * una webcam (índice 0, 1...);
  * la cámara del móvil (stream por URL, p. ej. IP Webcam / DroidCam);
  * un archivo de vídeo grabado (.mp4, .avi, .mov...).

Con un archivo de vídeo, el juego funciona igual que en vivo: el vídeo se
reproduce a su velocidad real (si el equipo va lento, se saltan frames para no
retrasarse) y el reloj del juego es el del propio vídeo, de modo que las
duraciones, la cuenta atrás y el tempo de las repeticiones son exactos. El
vídeo empieza en pausa (primer frame) y se controla con play / pause / rewind.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from config import CameraConfig


# =============================================================================
# Constantes
# =============================================================================
_DEFAULT_FPS = 30.0
_MAX_CATCH_UP_FRAMES = 120   # Máximo de frames que se saltan de golpe si el equipo va lento


# =============================================================================
# Funciones
# =============================================================================
def parse_source(source: int | str) -> int | str:
    """'0' -> 0 ; 'http://...' y rutas de archivo se dejan tal cual."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


def is_video_file(source: int | str) -> bool:
    """True si la fuente es un archivo local (vídeo grabado)."""
    if not isinstance(source, str) or "://" in source:
        return False
    return Path(source).expanduser().is_file()


# =============================================================================
# Clases
# =============================================================================
class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.source = parse_source(cfg.source)
        self.is_video = is_video_file(self.source)
        if self.is_video:
            self.source = str(Path(self.source).expanduser().resolve())
        self.frame_id = 0            # Aumenta con cada frame nuevo (no al repetir uno en pausa)
        self._failed = 0
        self._cap = None
        self._open()
        if self.is_video:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if 1.0 <= fps <= 240.0 else _DEFAULT_FPS
            self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._reset_playback()
        elif isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            if self.is_video:
                raise RuntimeError(f"No se pudo abrir el vídeo '{self.source}'. "
                                   "Comprueba que es un vídeo válido (.mp4, .avi, .mov...).")
            if isinstance(self.source, str) and "://" not in self.source:
                raise RuntimeError(f"No existe el archivo de vídeo '{self.source}'.")
            raise RuntimeError(
                f"No se pudo abrir la cámara '{self.source}'. Comprueba que está conectada, "
                "que ningún otro programa la usa, o prueba otro índice con --source 1."
            )

    # --- Vídeo: estado de reproducción ---------------------------------------
    def _reset_playback(self) -> None:
        self.paused = True
        self.ended = False
        self._pos = -1               # Índice del frame actual
        self._frame: np.ndarray | None = None
        self._t0 = 0.0               # Instante (reloj real) que corresponde al frame 0
        if not self._grab_next(retrieve=True):
            raise RuntimeError(f"El vídeo '{self.source}' no tiene frames legibles.")

    def _grab_next(self, retrieve: bool) -> bool:
        """Avanza un frame. Si retrieve, lo decodifica y lo guarda como frame actual."""
        if retrieve:
            ok, frame = self._cap.read()
            ok = ok and frame is not None
        else:
            ok = self._cap.grab()
        if not ok:
            return False
        self._pos += 1
        if retrieve:
            self._frame = frame
            self.frame_id += 1
        return True

    def _advance_video(self) -> None:
        """Avanza hasta el frame que toca según el reloj real (espera o salta frames)."""
        due = (self._pos + 1) / self.fps                     # Instante del siguiente frame
        elapsed = time.monotonic() - self._t0
        if elapsed < due:                                    # Vamos adelantados: esperar
            time.sleep(min(due - elapsed, 1.0 / self.fps))
            elapsed = time.monotonic() - self._t0
        target = max(int(elapsed * self.fps), self._pos + 1)  # Frame que corresponde ahora
        skip = min(target - self._pos - 1, _MAX_CATCH_UP_FRAMES)
        for _ in range(skip):                                # Vamos retrasados: saltar
            if not self._grab_next(retrieve=False):
                self._finish()
                return
        if not self._grab_next(retrieve=True):
            self._finish()
            return
        if skip == _MAX_CATCH_UP_FRAMES:                     # Re-sincroniza tras un parón largo
            self._t0 = time.monotonic() - self._pos / self.fps

    def _finish(self) -> None:
        self.ended = True
        self.paused = True

    def play(self) -> None:
        if self.is_video and self.paused and not self.ended:
            self._t0 = time.monotonic() - self._pos / self.fps
            self.paused = False

    def pause(self) -> None:
        if self.is_video:
            self.paused = True

    def toggle_pause(self) -> None:
        if not self.is_video:
            return
        if self.paused:
            self.play()
        else:
            self.pause()

    def rewind(self) -> None:
        """Vuelve al principio del vídeo, en pausa."""
        if not self.is_video:
            return
        self._cap.release()
        self._open()
        self._reset_playback()

    @property
    def position(self) -> float:
        """Segundo del vídeo del frame actual."""
        return max(self._pos, 0) / self.fps if self.is_video else 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        """(ancho, alto) del vídeo."""
        h, w = self._frame.shape[:2]
        return w, h

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.is_video and self.frame_count > 0 else 0.0

    # --- Reloj ---------------------------------------------------------------
    def clock(self) -> float:
        """Tiempo (s) del frame actual: el del vídeo, o el reloj real en vivo."""
        return self.position if self.is_video else time.monotonic()

    def timestamp_ms(self) -> int | None:
        """Timestamp del frame actual para el detector (None en vivo: usa el reloj real)."""
        return int(round(self.position * 1000)) if self.is_video else None

    # --- Lectura -------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Devuelve un frame BGR (ya espejado si procede) o None si falló la lectura.

        Con un vídeo en pausa o terminado devuelve otra vez el frame actual
        (sin aumentar frame_id).
        """
        if self.is_video:
            if not self.paused:
                self._advance_video()
            frame = self._frame
        else:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._failed += 1
                if self._failed >= self.cfg.max_failed_reads:
                    raise RuntimeError(f"La cámara '{self.source}' ha dejado de enviar imágenes.")
                return None
            self._failed = 0
            self.frame_id += 1
        return cv2.flip(frame, 1) if self.cfg.mirror else frame.copy()

    # --- Cierre (también usable con "with") ----------------------------------
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
