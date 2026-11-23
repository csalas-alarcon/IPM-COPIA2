"""
tests/test_logic.py  ·  Pruebas sin cámara.

Puntuación, conteo y juego completo con poses sintéticas.

Ejecutar:  .venv/bin/python -m unittest discover -s tests -v
(Windows:  .venv\\Scripts\\python -m unittest discover -s tests -v)
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.counter import RepCounter  # noqa: E402
from analysis.scoring import trajectory_error  # noqa: E402
from config import Settings  # noqa: E402
from exercises.catalog import EXERCISES, build_exercises  # noqa: E402
from vision.pose import Landmark, PoseFrame  # noqa: E402


# =============================================================================
# Utilidades: reps y poses sintéticas
# =============================================================================
def cosine(tau):
    return (1 - np.cos(2 * np.pi * tau)) / 2


def simulate(ex, cfg, profile=cosine, reps=3, dur=2.0, fps=30, noise=1.0, depth=1.0, pen=None, seed=0):
    """Genera reps sintéticas en espacio de ángulos y devuelve los eventos del contador."""
    rng = np.random.default_rng(seed)
    counter, t, events = RepCounter(ex, cfg), 0.0, []

    def feed(f, p_pen):
        nonlocal t
        ev = counter.update(t, f, ex.progress(f), p_pen)
        t += 1 / fps
        if ev:
            events.append(ev)

    for _ in range(15):
        feed(ex.rest_vector + rng.normal(0, noise, ex.rest_vector.shape), {})
    for _ in range(reps):
        n = int(dur * fps)
        for i in range(n + 1):
            s = profile(i / n) * depth
            feed(ex.rest_vector + s * (ex.peak_vector - ex.rest_vector)
                 + rng.normal(0, noise, ex.rest_vector.shape), pen or {})
        for _ in range(5):
            feed(ex.rest_vector + rng.normal(0, noise, ex.rest_vector.shape), {})
    return events


# =============================================================================
# Pruebas de lógica
# =============================================================================
class TestExercises(unittest.TestCase):
    def test_progress_endpoints(self):
        for ex in build_exercises(EXERCISES):
            self.assertAlmostEqual(ex.progress(ex.rest_vector), 0.0, places=6)
            self.assertAlmostEqual(ex.progress(ex.peak_vector), 1.0, places=6)
            mid = (ex.rest_vector + ex.peak_vector) / 2
            self.assertAlmostEqual(ex.progress(mid), 0.5, places=6)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings().scoring

    def scores(self, ex, **kw):
        evs = simulate(ex, self.cfg, **kw)
        return [e for e in evs if e.kind == "rep"], [e for e in evs if e.kind == "partial"]

    def test_ideal_reps_score_high(self):
        for ex in build_exercises(EXERCISES):
            reps, partial = self.scores(ex)
            self.assertEqual((len(reps), len(partial)), (3, 0), ex.key)
            self.assertGreater(min(r.score.total for r in reps), 90, ex.key)

    def test_short_range_is_partial(self):
        for ex in build_exercises(EXERCISES):
            reps, partial = self.scores(ex, depth=0.6)
            self.assertEqual((len(reps), len(partial)), (0, 3), ex.key)

    def test_defects_are_penalised(self):
        ex = build_exercises(["pushup"])[0]
        ideal = self.scores(ex)[0][0].score
        fast = self.scores(ex, dur=0.5)[0][0].score
        jerky = self.scores(ex, profile=lambda t: np.round(cosine(t) * 3) / 3)[0][0].score
        bad_form = self.scores(ex, pen={"cuerpo": 0.8})[0][0].score
        for name, s in (("rápida", fast), ("a tirones", jerky), ("mala técnica", bad_form)):
            self.assertLess(s.total, ideal.total - 10, name)
        self.assertIn("Más lento y controlado", fast.feedback)
        self.assertIn(ex.form_messages["cuerpo"], bad_form.feedback)

    def test_trajectory_error_zero_for_ideal_curve(self):
        # Curva ideal 0 -> 1 -> 0 recortada donde supera el umbral de reposo,
        # igual que la graba el contador.
        t = np.linspace(0, 2, 121)
        u = cosine(t / 2)
        keep = u >= 0.2
        err, _, _ = trajectory_error(t[keep], u[keep][:, None], np.array([1.0]))
        self.assertLess(err, 0.01)


# =============================================================================
# Prueba del juego completo
# =============================================================================
def pushup_pose(theta_deg: float) -> PoseFrame:
    """Esqueleto 2D sintético de una flexión vista de lado con el codo a theta grados."""
    pts = np.zeros((33, 2))
    l1 = l2 = 110.0
    wrist = np.array([860.0, 600.0])
    d = np.sqrt(l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * np.cos(np.radians(theta_deg)))
    shoulder = wrist + [0, -d]
    elbow = (shoulder + wrist) / 2 + [-np.sqrt(max(l1 ** 2 - (d / 2) ** 2, 0)), 0]
    ankle = np.array([300.0, 590.0])
    hip = shoulder + 0.55 * (ankle - shoulder)
    knee = (hip + ankle) / 2
    for side, off in (("LEFT", (0, 0)), ("RIGHT", (4, 2))):
        for joint, p in (("SHOULDER", shoulder), ("ELBOW", elbow), ("WRIST", wrist), ("HIP", hip),
                         ("KNEE", knee), ("ANKLE", ankle)):
            pts[Landmark[f"{side}_{joint}"]] = p + off
    pts[:11] = shoulder + [70, -20]
    world = np.hstack([pts / 400.0, np.zeros((33, 1))])
    return PoseFrame(pts, np.full(33, 0.95), world, 0, True)


class TestGameFlow(unittest.TestCase):
    """Recorre el juego completo (menú -> resumen) con poses sintéticas, sin ventana."""

    def test_full_session(self):
        import game.game as game_module
        from game.game import Game, Phase
        from ui.canvas import TextRenderer

        settings = Settings()
        settings.game.exercises = ("pushup",)
        clock = [1000.0]
        original = game_module.time.monotonic
        game_module.time.monotonic = lambda: clock[0]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                game_module.RESULTS_DIR = Path(tmp)
                g = Game(settings, None, None, build_exercises(["pushup"]), TextRenderer())
                frame = np.zeros((720, 1280, 3), np.uint8)

                def step(pose):
                    clock[0] += 1 / 30
                    g._update(pose, clock[0])
                    g._draw(frame.copy(), pose, clock[0])   # también prueba el dibujo

                rest = pushup_pose(165)
                step(rest)
                g._handle_key(13)                              # ENTER en el menú
                self.assertIs(g.phase, Phase.INSTRUCTIONS)
                for _ in range(60):                            # reposo mantenido -> cuenta atrás
                    step(rest)
                self.assertIs(g.phase, Phase.COUNTDOWN)
                for _ in range(100):
                    step(rest)
                self.assertIs(g.phase, Phase.ACTIVE)
                for peak in (80, 80, 125):                     # 2 reps buenas + 1 incompleta
                    for i in range(61):
                        step(pushup_pose(165 + cosine(i / 60) * (peak - 165)))
                    for _ in range(10):
                        step(rest)
                self.assertEqual(g.session.count, 2)
                self.assertEqual(g.session.partial_reps, 1)
                g._handle_key(13)                              # terminar ejercicio
                self.assertIs(g.phase, Phase.EXERCISE_RESULT)
                g._handle_key(13)                              # resumen
                self.assertIs(g.phase, Phase.SUMMARY)
                self.assertTrue(g.saved_path and g.saved_path.is_file())
        finally:
            game_module.time.monotonic = original


# =============================================================================
# Pruebas con un archivo de vídeo
# =============================================================================
def write_video(path: Path, frames: int = 45, fps: float = 30.0, size=(320, 240)) -> Path:
    """Vídeo sintético: cada frame lleva su índice codificado en el color."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 5) % 256, np.uint8))
    writer.release()
    return path


class TestVideoSource(unittest.TestCase):
    def test_playback_pause_end_and_rewind(self):
        from config import CameraConfig
        from vision.camera import Camera, is_video_file

        with tempfile.TemporaryDirectory() as tmp:
            path = write_video(Path(tmp) / "demo.mp4")
            self.assertTrue(is_video_file(str(path)))
            self.assertFalse(is_video_file("0") or is_video_file("http://x/video"))
            with Camera(CameraConfig(source=str(path), mirror=False)) as cam:
                self.assertTrue(cam.is_video and cam.paused)
                self.assertEqual(cam.frame_size, (320, 240))
                self.assertAlmostEqual(cam.duration, 1.5, places=2)
                first_id = cam.frame_id
                cam.read()
                cam.read()
                self.assertEqual(cam.frame_id, first_id)     # En pausa no avanza
                self.assertEqual(cam.clock(), 0.0)

                cam.play()
                for _ in range(10):
                    cam.read()
                self.assertGreater(cam.frame_id, first_id)
                self.assertGreater(cam.position, 0.0)
                self.assertEqual(cam.timestamp_ms(), round(cam.position * 1000))

                cam.pause()
                frozen = cam.clock()
                cam.read()
                self.assertEqual(cam.clock(), frozen)       # El reloj se congela en pausa

                cam.play()
                for _ in range(200):                       # Hasta el final (tiempo real, ~1.5 s)
                    cam.read()
                    if cam.ended:
                        break
                self.assertTrue(cam.ended and cam.paused)
                self.assertIsNotNone(cam.read())           # Al acabar sigue mostrando el último frame

                cam.rewind()
                self.assertFalse(cam.ended)
                self.assertTrue(cam.paused)
                self.assertEqual(cam.position, 0.0)


class FakeVideo:
    """Vídeo simulado para el juego: reloj controlado a mano."""
    is_video = True
    source = "demo.mp4"

    def __init__(self):
        self.position, self.duration = 0.0, 30.0
        self.paused, self.ended = True, False
        self.frame_size = (1280, 720)

    def clock(self):
        return self.position

    def play(self):
        if not self.ended:
            self.paused = False

    def toggle_pause(self):
        self.paused = not self.paused

    def rewind(self):
        self.position, self.paused, self.ended = 0.0, True, False


class TestVideoGameFlow(unittest.TestCase):
    def test_video_flow(self):
        from game.game import Game, Phase
        from ui.canvas import TextRenderer

        settings = Settings()
        settings.game.exercises = ("pushup", "bulgarian_squat")
        settings.game.save_results = False
        cam = FakeVideo()
        g = Game(settings, cam, None, build_exercises(settings.game.exercises), TextRenderer())
        frame = np.zeros((720, 1280, 3), np.uint8)

        def step(pose):
            if not cam.paused:
                cam.position += 1 / 30
            g._update(pose, g._now())
            g._draw(frame.copy(), pose, g._now())

        rest = pushup_pose(165)
        step(rest)
        self.assertTrue(cam.paused)                    # En el menú el vídeo está en pausa
        g._handle_key(13)                              # ENTER: empieza el vídeo
        self.assertFalse(cam.paused)
        self.assertIs(g.phase, Phase.INSTRUCTIONS)
        for _ in range(60):
            step(rest)
        self.assertIs(g.phase, Phase.COUNTDOWN)

        g._handle_key(ord("p"))                        # Pausa: el juego se congela
        self.assertTrue(cam.paused)
        for _ in range(200):
            step(rest)
        self.assertIs(g.phase, Phase.COUNTDOWN)
        g._handle_key(ord("p"))
        for _ in range(100):
            step(rest)
        self.assertIs(g.phase, Phase.ACTIVE)

        for i in range(61):                            # Una repetición buena
            step(pushup_pose(165 + cosine(i / 60) * (80 - 165)))
        for _ in range(10):
            step(rest)
        self.assertEqual(g.session.count, 1)

        cam.ended = cam.paused = True                  # Se acaba el vídeo a mitad del ejercicio
        g._on_video_end()
        self.assertIs(g.phase, Phase.EXERCISE_RESULT)
        g._draw(frame.copy(), None, g._now())
        g._handle_key(13)                              # Los ejercicios que faltan se saltan
        self.assertIs(g.phase, Phase.SUMMARY)
        self.assertEqual([s.skipped for s in g.sessions], [False, True])

        g._handle_key(13)                              # Jugar otra vez: rebobina
        self.assertIs(g.phase, Phase.MENU)
        self.assertFalse(cam.ended)
        self.assertEqual(cam.position, 0.0)


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
