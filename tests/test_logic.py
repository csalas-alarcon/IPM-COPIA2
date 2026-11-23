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


class FakeCamera:
    """Cámara en vivo simulada (las poses se pasan a mano)."""
    is_video = False
    source = 0
    label = "Cámara 0"


class TestGameFlow(unittest.TestCase):
    """Recorre el juego completo (menú -> resumen) con poses sintéticas, sin ventana."""

    def test_full_session(self):
        import game.game as game_module
        from game.game import Game, Phase
        from ui.canvas import TextRenderer

        settings = Settings()
        settings.game.exercises = ("pushup",)
        settings.game.calibrate = False
        clock = [1000.0]
        original = game_module.time.monotonic
        game_module.time.monotonic = lambda: clock[0]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                game_module.RESULTS_DIR = Path(tmp)
                g = Game(settings, FakeCamera(), None, build_exercises(["pushup"]), TextRenderer())
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
        settings.game.calibrate = False
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
# Calibración (repetición de prueba)
# =============================================================================
def feed_calibration(cal, ex, rest, peak, dur=2.0, fps=30, t0=0.0, noise=0.5, seed=1):
    """Reposo + una repetición coseno rest -> peak -> rest. Devuelve el último estado."""
    rng = np.random.default_rng(seed)
    t, status = t0, None
    for _ in range(20):
        f = rest + rng.normal(0, noise, rest.shape)
        cal.add_rest(f)
        status = cal.update(t, f, ex.progress(f)) or status
        t += 1 / fps
    n = int(dur * fps)
    for i in range(n + 1):
        f = rest + cosine(i / n) * (peak - rest) + rng.normal(0, noise, rest.shape)
        status = cal.update(t, f, ex.progress(f)) or status
        t += 1 / fps
        if status == "done":
            break
    for _ in range(5):
        if status == "done":
            break
        f = rest + rng.normal(0, noise, rest.shape)
        status = cal.update(t, f, ex.progress(f)) or status
        t += 1 / fps
    return status


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def calibrator(self, key):
        from analysis.calibration import Calibrator
        ex = build_exercises([key])[0]
        return ex, Calibrator(ex, self.settings.game, self.settings.scoring)

    def test_learns_personal_reference(self):
        ex, cal = self.calibrator("pushup")
        rest, peak = np.array([172.0]), np.array([100.0])     # Menos profundo que el estándar (80º)
        self.assertEqual(feed_calibration(cal, ex, rest, peak), "done")
        r = cal.result
        self.assertAlmostEqual(float(r.rest[0]), 172.0, delta=2.0)
        self.assertAlmostEqual(float(r.peak[0]), 100.0, delta=3.0)
        self.assertEqual(r.fallback, ())
        r.apply(ex)
        self.assertTrue(ex.calibrated)
        self.assertAlmostEqual(ex.progress(np.array([100.0])), 1.0, delta=0.05)

        # Con la referencia personal, sus repeticiones completas puntúan alto
        cfg = self.settings.scoring
        evs = simulate(ex, cfg)
        reps = [e for e in evs if e.kind == "rep"]
        self.assertEqual(len(reps), 3)
        self.assertGreater(min(e.score.total for e in reps), 90)
        self.assertTrue(reps[0].score.curve and reps[0].score.reference_curve)

        ex.reset_reference()
        self.assertFalse(ex.calibrated)
        self.assertEqual(float(ex.peak_vector[0]), 80.0)

    def test_too_shallow_asks_to_repeat(self):
        ex, cal = self.calibrator("pushup")
        # 30 % del recorrido estándar: no llega al mínimo para calibrar
        status = feed_calibration(cal, ex, np.array([165.0]), np.array([165.0 - 0.3 * 85]))
        self.assertEqual(status, "retry")
        self.assertIsNone(cal.result)

    def test_feature_that_does_not_move_uses_default(self):
        ex, cal = self.calibrator("pullup")
        names = [f.name for f in ex.features]
        rest = ex.default_rest.copy()
        peak = ex.default_peak.copy()
        peak[1] = rest[1] - 5.0                               # El hombro casi no se mueve
        self.assertEqual(feed_calibration(cal, ex, rest, peak), "done")
        self.assertEqual(cal.result.fallback, (names[1],))
        self.assertEqual(float(cal.result.peak[1]), float(ex.default_peak[1]))

    def test_profile_from_ideal_rep_is_close_to_cosine(self):
        from analysis.scoring import _ease_from, build_profile
        t = np.linspace(0, 2, 61)
        u = cosine(t / 2)[:, None]
        up, down = build_profile(t, u, np.array([1.0]), 20)
        ideal = _ease_from(np.zeros(1), 20)
        self.assertLess(np.abs(up - ideal).max(), 0.05)
        self.assertLess(np.abs(down - ideal).max(), 0.05)


# =============================================================================
# Juego con calibración, menú y pantallas
# =============================================================================
def make_game(settings=None, camera=None, keys=("pushup",)):
    from game.game import Game
    from ui.canvas import TextRenderer
    settings = settings or Settings()
    settings.game.save_results = False
    return Game(settings, camera or FakeCamera(), None, build_exercises(list(keys)), TextRenderer())


class TestCalibratedGame(unittest.TestCase):
    def test_flow_with_test_rep(self):
        from game.game import Phase
        g = make_game()
        clock = [0.0]
        frame = np.zeros((720, 1280, 3), np.uint8)

        def step(pose, n=1):
            for _ in range(n):
                clock[0] += 1 / 30
                g._update(pose, clock[0])
                g._draw(frame.copy(), pose, clock[0])

        import game.game as game_module
        original = game_module.time.monotonic
        game_module.time.monotonic = lambda: clock[0]
        try:
            rest = pushup_pose(170)
            step(rest)
            g._handle_key(13)
            step(rest, 60)
            self.assertIs(g.phase, Phase.COUNTDOWN)
            step(rest, 100)
            self.assertIs(g.phase, Phase.CALIBRATION)
            for i in range(61):                                    # Repetición de prueba hasta 100º
                step(pushup_pose(170 + cosine(i / 60) * (100 - 170)))
            step(rest, 3)
            self.assertIs(g.phase, Phase.ACTIVE)
            ex = g.exercises[0]
            self.assertTrue(ex.calibrated and g.session.calibrated)
            self.assertAlmostEqual(g.session.reference["peak"]["codo"], 100.0, delta=4.0)
            step(rest, 10)
            for _ in range(2):                                     # Dos repeticiones "suyas"
                for i in range(61):
                    step(pushup_pose(170 + cosine(i / 60) * (100 - 170)))
                step(rest, 10)
            self.assertEqual(g.session.count, 2)
            self.assertGreater(g.session.average, 85)
            g._do("advance")
            self.assertIs(g.phase, Phase.EXERCISE_RESULT)
            g._draw(frame.copy(), None, clock[0])                  # Pantalla con la gráfica
            g._do("advance")
            self.assertIs(g.phase, Phase.SUMMARY)
            self.assertIn('"calibrated": true', __import__("json").dumps(
                [s.to_dict() for s in g.sessions]))
        finally:
            game_module.time.monotonic = original

    def test_calibration_can_be_skipped_with_standard_reference(self):
        from game.game import Phase
        g = make_game()
        g._do("start")
        g._go(Phase.CALIBRATION)
        g._do("advance")
        self.assertIs(g.phase, Phase.ACTIVE)
        self.assertFalse(g.exercises[0].calibrated)


class TestMenu(unittest.TestCase):
    def test_menu_actions(self):
        from game.game import Phase
        g = make_game()
        self.assertEqual([e.key for e in g.available], ["pushup", "pullup", "bulgarian_squat"])
        g._handle_key(ord("3"))                                    # Activa la sentadilla búlgara
        self.assertEqual(g.selected, {"pushup", "bulgarian_squat"})
        g._handle_key(ord("k"))
        self.assertFalse(g.settings.game.calibrate)
        g._do("start")
        self.assertEqual([e.key for e in g.exercises], ["pushup", "bulgarian_squat"])
        self.assertIs(g.phase, Phase.INSTRUCTIONS)

        g._do("menu")
        g.selected.clear()
        g._do("start")                                             # Sin ejercicios no empieza
        self.assertIs(g.phase, Phase.MENU)
        self.assertIsNotNone(g._toast)

    def test_no_source_cannot_start(self):
        from game.game import Phase
        from vision.camera import NoSource
        g = make_game(camera=NoSource(320, 240))
        g._do("start")
        self.assertIs(g.phase, Phase.MENU)

    def test_click_on_button_triggers_action(self):
        from game.game import Phase
        g = make_game()
        g._draw(np.zeros((720, 1280, 3), np.uint8), None, 0.0)
        action, (x1, y1, x2, y2) = next(b for b in g._buttons if b[0] == "start")
        g._click = ((x1 + x2) // 2, (y1 + y2) // 2)
        g._handle_click()
        self.assertIs(g.phase, Phase.INSTRUCTIONS)

    def test_all_screens_draw(self):
        from datetime import datetime, timedelta
        from game.game import Phase
        from game.history import HistoryEntry
        from ui.screens import PickerInfo
        g = make_game(keys=("pushup", "pullup", "bulgarian_squat"))
        for size in ((720, 1280), (480, 640)):                     # 16:9 y 4:3
            frame = np.zeros((*size, 3), np.uint8)
            g._reset_game()
            g._draw(frame.copy(), None, 0.0)
            g._picker = PickerInfo([Path(f"/tmp/video_{i}.mp4") for i in range(3)], Path("/tmp"), False)
            g._go(Phase.PICK_VIDEO)
            g._draw(frame.copy(), None, 0.0)
            g._history = [HistoryEntry(Path("x"), datetime(2026, 11, 1) + timedelta(days=i), 300 + 50 * i,
                                       8, 70 + i, ["Flexión"]) for i in range(5)]
            g._go(Phase.HISTORY)
            g._draw(frame.copy(), None, 0.0)
            g._history = []
            g._draw(frame.copy(), None, 0.0)
            g._do("back")
            self.assertIs(g.phase, Phase.MENU)
            g._do("start")
            for phase in (Phase.INSTRUCTIONS, Phase.COUNTDOWN, Phase.CALIBRATION, Phase.ACTIVE,
                          Phase.EXERCISE_RESULT, Phase.SUMMARY):
                g._go(phase)
                g._draw(frame.copy(), pushup_pose(120), 0.0)


# =============================================================================
# Historial, grabación, sonido y adaptación de frames
# =============================================================================
class TestExtras(unittest.TestCase):
    def test_history_round_trip(self):
        from game.history import load_history
        from game.session import ExerciseSession, save_sessions
        ex = build_exercises(["pushup"])[0]
        reps = [e.score for e in simulate(ex, Settings().scoring) if e.kind == "rep"]
        with tempfile.TemporaryDirectory() as tmp:
            s = ExerciseSession(ex, reps=reps)
            save_sessions([s], Path(tmp))
            (Path(tmp) / "session_roto.json").write_text("{no es json")
            hist = load_history(Path(tmp))
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0].reps, 3)
            self.assertEqual(hist[0].total_points, s.points)
            self.assertEqual(hist[0].exercises, ["Flexión"])

    def test_recorder_writes_video(self):
        import cv2
        from ui.recorder import Recorder
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(Path(tmp) / "rec.mp4", fps=10)
            for _ in range(5):
                rec.write(np.zeros((240, 321, 3), np.uint8))     # Ancho impar: se ajusta
            rec.close()
            cap = cv2.VideoCapture(str(Path(tmp) / "rec.mp4"))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
            cap.release()

    def test_sound_synthesis_and_silent_mode(self):
        from ui.sound import SOUNDS, Sound, synthesize, to_wav_bytes
        sig = synthesize(SOUNDS["rep"])
        self.assertGreater(len(sig), 1000)
        self.assertLessEqual(float(np.abs(sig).max()), 0.36)
        self.assertTrue(to_wav_bytes(sig).startswith(b"RIFF"))
        snd = Sound(enabled=False)
        self.assertFalse(snd.enabled)
        snd.play("rep")                                          # No hace nada ni falla

    def test_fit_frame(self):
        from vision.camera import fit_frame
        portrait = np.full((1920, 1080, 3), 200, np.uint8)
        out = fit_frame(portrait, 720)
        self.assertEqual(out.shape[:2], (720, 1280))             # Reducido y centrado en 16:9
        self.assertEqual(int(out[360, 640, 0]), 200)
        self.assertNotEqual(int(out[360, 10, 0]), 200)
        webcam = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(fit_frame(webcam, 720).shape[:2], (480, 640))   # 4:3 no se toca

    def test_video_start_offset_and_list(self):
        from config import CameraConfig
        from ui.dialogs import list_videos
        from vision.camera import Camera
        with tempfile.TemporaryDirectory() as tmp:
            path = write_video(Path(tmp) / "demo.mp4", frames=60)
            (Path(tmp) / "notas.txt").write_text("x")
            self.assertEqual(list_videos([tmp, tmp]), [path.resolve()])
            with Camera(CameraConfig(source=str(path), start_seconds=1.0)) as cam:
                self.assertAlmostEqual(cam.position, 1.0, delta=0.05)
                cam.read()
                self.assertAlmostEqual(cam.position, 1.0, delta=0.05)   # En pausa no avanza


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
