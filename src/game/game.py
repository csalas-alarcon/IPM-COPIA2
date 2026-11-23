"""
game/game.py  ·  Bucle principal y máquina de estados del juego.

MENÚ -> [por cada ejercicio: INSTRUCCIONES -> CUENTA ATRÁS -> ACTIVO -> RESULTADO] -> RESUMEN
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from analysis.counter import ExerciseTracker, TrackerState
from config import RESULTS_DIR, Settings
from exercises.base import Exercise
from game.session import ExerciseSession, save_sessions
from ui.canvas import ORANGE, Canvas, TextRenderer, score_color
from ui.screens import (Feedback, draw_countdown, draw_exercise_result, draw_hud,
                        draw_instructions, draw_menu, draw_skeleton, draw_summary)
from vision.camera import Camera
from vision.detector import PoseDetector


# =============================================================================
# Constantes
# =============================================================================
KEY_ENTER = (13, 10)
KEY_SPACE = 32
KEY_ESC = 27


# =============================================================================
# Tipos de datos
# =============================================================================
class Phase(Enum):
    MENU = auto()
    INSTRUCTIONS = auto()
    COUNTDOWN = auto()
    ACTIVE = auto()
    EXERCISE_RESULT = auto()
    SUMMARY = auto()


# =============================================================================
# Clases
# =============================================================================
class Game:
    def __init__(self, settings: Settings, camera: Camera, detector: PoseDetector,
                 exercises: list[Exercise], renderer: TextRenderer):
        if not exercises:
            raise ValueError("Se necesita al menos un ejercicio")
        self.settings = settings
        self.camera = camera
        self.detector = detector
        self.exercises = exercises
        self.renderer = renderer
        self.trackers = [ExerciseTracker(ex, settings.game, settings.scoring) for ex in exercises]
        self.window = settings.game.window_name
        self._fullscreen = False
        self._running = True
        self._reset_game()

    # --- Estado --------------------------------------------------------------
    def _reset_game(self) -> None:
        self.phase = Phase.MENU
        self.index = 0
        self.sessions: list[ExerciseSession] = []
        self.saved_path: Path | None = None
        self.state = TrackerState()
        self.feedback: Feedback | None = None
        self._phase_start = time.monotonic()
        self._ready_since: float | None = None

    @property
    def tracker(self) -> ExerciseTracker:
        return self.trackers[self.index]

    @property
    def session(self) -> ExerciseSession:
        return self.sessions[self.index]

    def _go(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_start = time.monotonic()
        self._ready_since = None

    def _start_exercise(self, index: int) -> None:
        self.index = index
        self.sessions.append(ExerciseSession(self.exercises[index]))
        self.tracker.reset()
        self.feedback = None
        self._go(Phase.INSTRUCTIONS)

    def _finish_exercise(self, skipped: bool = False) -> None:
        if self.phase is Phase.ACTIVE:
            self.session.duration = time.monotonic() - self._phase_start
        self.session.skipped = skipped
        self._go(Phase.EXERCISE_RESULT)

    def _next(self) -> None:
        if self.index + 1 < len(self.exercises):
            self._start_exercise(self.index + 1)
        else:
            if self.settings.game.save_results:
                try:
                    self.saved_path = save_sessions(self.sessions, RESULTS_DIR)
                    print(f"[juego] Resultados guardados en {self.saved_path}")
                except OSError as e:
                    print(f"[juego] No se pudieron guardar los resultados: {e}")
            self._go(Phase.SUMMARY)

    # --- Lógica por fase -----------------------------------------------------
    def _update(self, pose, now: float) -> None:
        g = self.settings.game
        elapsed = now - self._phase_start

        if self.phase is Phase.INSTRUCTIONS:
            self.state = self.tracker.update(pose, now, counting=False)
            ready = (self.state.posture_ok and self.state.progress is not None
                     and self.state.progress <= g.ready_progress)
            if ready:
                self._ready_since = self._ready_since or now
                if now - self._ready_since >= g.ready_hold_seconds:
                    self._go(Phase.COUNTDOWN)
            else:
                self._ready_since = None

        elif self.phase is Phase.COUNTDOWN:
            self.state = self.tracker.update(pose, now, counting=False)
            if elapsed >= g.countdown_seconds:
                self.tracker.counter.reset()
                self._go(Phase.ACTIVE)

        elif self.phase is Phase.ACTIVE:
            self.state = self.tracker.update(pose, now, counting=True)
            ev = self.state.event
            if ev is not None:
                self._on_event(ev, now)
            if elapsed >= g.exercise_duration or self.session.count >= g.target_reps:
                self._finish_exercise()

    def _on_event(self, ev, now: float) -> None:
        until = now + self.settings.game.feedback_seconds
        if ev.kind == "rep" and ev.score is not None:
            self.session.reps.append(ev.score)
            s = ev.score
            title = "¡Perfecta!" if s.total >= 90 else f"+{s.total:.0f}"
            self.feedback = Feedback([title] + s.feedback[:2], score_color(s.total), until)
            print(f"[{self.session.exercise.key}] rep {self.session.count}: {s.total:.1f} "
                  f"(rango {s.rom:.2f}, trayectoria {s.trajectory:.2f}, tempo {s.tempo:.2f}, "
                  f"técnica {s.form:.2f}, {s.duration:.1f}s)")
        elif ev.kind == "partial":
            self.session.partial_reps += 1
            self.feedback = Feedback(["Incompleta", self.session.exercise.rom_feedback], ORANGE, until)

    # --- Dibujo --------------------------------------------------------------
    def _draw(self, frame: np.ndarray, pose, now: float) -> np.ndarray:
        c = Canvas(frame, self.renderer)
        g, sc = self.settings.game, self.settings.scoring
        ex = self.exercises[self.index] if self.phase is not Phase.MENU else None
        live = self.phase in (Phase.INSTRUCTIONS, Phase.COUNTDOWN, Phase.ACTIVE)
        draw_skeleton(c, pose, ex if live else None, self.state if live else None)

        if self.phase is Phase.MENU:
            draw_menu(c, self.exercises, g.exercise_duration, g.target_reps)
        elif self.phase is Phase.INSTRUCTIONS:
            frac = 0.0 if self._ready_since is None else (now - self._ready_since) / g.ready_hold_seconds
            draw_instructions(c, ex, self.index, len(self.exercises), self.state, min(frac, 1.0))
        elif self.phase is Phase.COUNTDOWN:
            draw_countdown(c, ex, g.countdown_seconds - (now - self._phase_start))
        elif self.phase is Phase.ACTIVE:
            time_left = max(g.exercise_duration - (now - self._phase_start), 0.0)
            draw_hud(c, self.session, self.state, time_left, g.target_reps, self.feedback, now,
                     sc.rest_threshold, sc.peak_threshold)
        elif self.phase is Phase.EXERCISE_RESULT:
            draw_exercise_result(c, self.session, self.index, len(self.exercises))
        elif self.phase is Phase.SUMMARY:
            draw_summary(c, self.sessions, self.saved_path)
        return c.finish()

    # --- Teclado -------------------------------------------------------------
    def _handle_key(self, key: int) -> None:
        if key == 255 or key == -1:
            return
        if key in (KEY_ESC, ord("q")):
            self._running = False
            return
        if key == ord("f"):
            self._fullscreen = not self._fullscreen
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL)
            return
        if key == ord("r"):
            self._reset_game()
            return

        advance = key in KEY_ENTER or key == KEY_SPACE
        skip = key == ord("n")
        if self.phase is Phase.MENU and advance:
            self.sessions = []
            self._start_exercise(0)
        elif self.phase is Phase.INSTRUCTIONS:
            if advance:
                self._go(Phase.COUNTDOWN)
            elif skip:
                self._finish_exercise(skipped=True)
        elif self.phase in (Phase.COUNTDOWN, Phase.ACTIVE) and (advance or skip):
            self._finish_exercise(skipped=skip and self.session.count == 0)
        elif self.phase is Phase.EXERCISE_RESULT and (advance or skip):
            self._next()
        elif self.phase is Phase.SUMMARY and advance:
            self._reset_game()

    # --- Bucle ---------------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.settings.camera.width, self.settings.camera.height)
        try:
            while self._running:
                frame = self.camera.read()
                if frame is None:
                    if (cv2.waitKey(10) & 0xFF) == KEY_ESC:
                        break
                    continue
                now = time.monotonic()
                pose = self.detector.detect(frame)
                self._update(pose, now)
                cv2.imshow(self.window, self._draw(frame, pose, now))

                # Un único waitKey por frame
                self._handle_key(cv2.waitKey(1) & 0xFF)
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break   # Ventana cerrada con la X
        finally:
            cv2.destroyAllWindows()
