"""
Utilidades compartidas por las pruebas del juego.

Poses sintéticas, fuentes de vídeo falsas, un reloj simulado y una clase base
que aísla cada prueba de results/ y del tiempo real. Lo usan test_logic.py y
test_juego.py. No contiene pruebas.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Settings  # noqa: E402
from exercises.catalog import build_exercises  # noqa: E402
from game.game import Game  # noqa: E402
from ui.canvas import TextRenderer  # noqa: E402
from vision.pose import Landmark, PoseFrame  # noqa: E402

FRAME = np.zeros((720, 1280, 3), np.uint8)
FPS = 30


def pose_flexion(codo_grados: float) -> PoseFrame:
    """Esqueleto 2D de una flexión vista de lado, con el codo al ángulo dado.

    Con 165º es la plancha (reposo); con 80º, el máximo esfuerzo.
    """
    puntos = np.zeros((33, 2))
    largo_brazo = 110.0
    muneca = np.array([860.0, 600.0])
    distancia = np.sqrt(2 * largo_brazo ** 2 * (1 - np.cos(np.radians(codo_grados))))
    hombro = muneca + [0, -distancia]
    desvio_codo = np.sqrt(max(largo_brazo ** 2 - (distancia / 2) ** 2, 0))
    codo = (hombro + muneca) / 2 + [-desvio_codo, 0]
    tobillo = np.array([300.0, 590.0])
    cadera = hombro + 0.55 * (tobillo - hombro)
    rodilla = (cadera + tobillo) / 2
    articulaciones = (("SHOULDER", hombro), ("ELBOW", codo), ("WRIST", muneca),
                      ("HIP", cadera), ("KNEE", rodilla), ("ANKLE", tobillo))
    for lado, desplazamiento in (("LEFT", (0, 0)), ("RIGHT", (4, 2))):
        for nombre, punto in articulaciones:
            puntos[Landmark[f"{lado}_{nombre}"]] = punto + desplazamiento
    puntos[:11] = hombro + [70, -20]
    mundo = np.hstack([puntos / 400.0, np.zeros((33, 1))])
    return PoseFrame(puntos, np.full(33, 0.95), mundo, 0, True)


class CamaraFalsa:
    """Cámara en vivo simulada: las poses se pasan a mano al juego."""
    is_video = False
    source = 0
    label = "Cámara 0"


class VideoFalso:
    """Vídeo simulado: su reloj (position) lo mueve la prueba."""
    is_video = True
    source = "demo.mp4"

    def __init__(self):
        self.position = 0.0
        self.duration = 30.0
        self.paused = True
        self.ended = False
        self.liberado = False
        self.frame_size = (1280, 720)

    def clock(self):
        return self.position

    def play(self):
        if not self.ended:
            self.paused = False

    def toggle_pause(self):
        self.paused = not self.paused

    def rewind(self):
        self.position = 0.0
        self.paused = True
        self.ended = False

    def release(self):
        self.liberado = True


class RelojFalso:
    """Sustituye al módulo time: el tiempo solo avanza cuando la prueba lo avanza."""

    def __init__(self, inicio: float = 1000.0):
        self.ahora = inicio

    def monotonic(self) -> float:
        return self.ahora

    def avanzar(self, segundos: float) -> None:
        self.ahora += segundos

    strftime = staticmethod(time.strftime)


def crear_juego(ajustes: Settings | None = None, camara=None, claves=("pushup",), **opciones) -> Game:
    """Juego sin detector ni guardado de resultados. Las opciones se pasan a Game."""
    ajustes = ajustes or Settings()
    ajustes.game.save_results = False
    return Game(ajustes, camara or CamaraFalsa(), None, build_exercises(list(claves)),
                TextRenderer(), **opciones)


class PruebaDeJuego(unittest.TestCase):
    """Base de las pruebas que usan Game.

    Cada prueba tiene su carpeta de resultados temporal (self.resultados), un
    reloj simulado (self.reloj) y la salida por consola silenciada. Todo se
    restaura al terminar.
    """

    def setUp(self):
        self.resultados = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reloj = RelojFalso()
        self.enterContext(mock.patch("game.game.RESULTS_DIR", self.resultados))
        self.enterContext(mock.patch("game.game.time", self.reloj))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def pasar_poses(self, juego: Game, poses) -> None:
        """Simula un frame a 30 fps por cada pose: avanza los relojes y actualiza el juego.

        Dibujar es lo más lento, así que solo se dibuja el último frame (basta
        para comprobar que la pantalla de cada fase se pinta sin errores).
        """
        for pose in poses:
            self.reloj.avanzar(1 / FPS)
            if juego.is_video and not juego.camera.paused:
                juego.camera.position += 1 / FPS
            juego._update(pose, juego._now())
        juego._draw(FRAME.copy(), poses[-1], juego._now())

    def pasar_frames(self, juego: Game, pose, frames: int = 1) -> None:
        """Simula `frames` frames seguidos con la misma pose."""
        self.pasar_poses(juego, [pose] * frames)
