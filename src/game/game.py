"""
game/game.py  ·  Bucle principal y máquina de estados del juego.

MENÚ -> [por cada ejercicio: INSTRUCCIONES -> CUENTA ATRÁS -> REPETICIÓN DE PRUEBA
         -> ACTIVO -> RESULTADO] -> RESUMEN
Desde el menú también: SELECTOR DE VÍDEO e HISTORIAL.

  * Menú: elegir la fuente (cámara o vídeo), los ejercicios y si se calibra.
  * Repetición de prueba (si la calibración está activada): el usuario hace una
    repetición y sus poses de reposo / máximo y su trayectoria pasan a ser la
    referencia del ejercicio (ver analysis/calibration.py).
  * Todo se puede hacer con el ratón (botones) o con el teclado.

Funciona igual con la cámara en vivo que con un vídeo grabado. Con un vídeo:
  * el vídeo está en pausa en el menú y empieza al pulsar EMPEZAR;
  * el reloj del juego es el del vídeo (pausar con P congela también el juego);
  * al terminar el vídeo se cierra el ejercicio en curso y se puede ver el resumen;
  * R (o "jugar otra vez") rebobina el vídeo y vuelve al menú.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import dataclasses
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from analysis.calibration import Calibrator
from analysis.counter import ExerciseTracker, TrackerState
from config import PROJECT_ROOT, RESULTS_DIR, VIDEOS_DIR, Settings
from exercises.base import Exercise
from exercises.catalog import EXERCISES
from game.history import HistoryEntry, load_history
from game.session import ExerciseSession, save_sessions
from ui.canvas import ACCENT, GREEN, ORANGE, RED, Canvas, TextRenderer, score_color
from ui.dialogs import ask_video_file, list_videos, tk_available
from ui.recorder import Recorder
from ui.screens import (Feedback, MenuInfo, PickerInfo, draw_calibration, draw_countdown,
                        draw_exercise_result, draw_history, draw_hud, draw_instructions,
                        draw_menu, draw_skeleton, draw_summary, draw_toast, draw_video_picker,
                        draw_video_status)
from ui.sound import Sound
from vision.camera import Camera, is_video_file
from vision.detector import PoseDetector


# =============================================================================
# Constantes
# =============================================================================
KEY_ENTER = (13, 10)
KEY_SPACE = 32
KEY_ESC = 27
TOAST_SECONDS = 4.0


# =============================================================================
# Tipos de datos
# =============================================================================
class Phase(Enum):
    MENU = auto()
    PICK_VIDEO = auto()
    HISTORY = auto()
    INSTRUCTIONS = auto()
    COUNTDOWN = auto()
    CALIBRATION = auto()
    ACTIVE = auto()
    EXERCISE_RESULT = auto()
    SUMMARY = auto()


LIVE_PHASES = (Phase.INSTRUCTIONS, Phase.COUNTDOWN, Phase.CALIBRATION, Phase.ACTIVE)


# =============================================================================
# Clases
# =============================================================================
class Game:
    def __init__(self, settings: Settings, camera, detector: PoseDetector | None,
                 exercises: list[Exercise], renderer: TextRenderer,
                 sound: Sound | None = None,
                 open_source: Callable[[int | str], Camera] | None = None):
        """exercises: los elegidos al empezar (en orden). En el menú se pueden
        activar o desactivar, y también los demás del catálogo."""
        if not exercises:
            raise ValueError("Se necesita al menos un ejercicio")
        self.settings = settings
        self.camera = camera
        self.detector = detector
        self.renderer = renderer
        self.sound = sound or Sound(enabled=False)
        self._open_source = open_source or self._default_open_source
        given = {ex.key for ex in exercises}
        self.available: list[Exercise] = list(exercises) + [
            cls() for key, cls in EXERCISES.items() if key not in given]
        self.selected: set[str] = set(given)
        self.trackers = {ex.key: ExerciseTracker(ex, settings.game, settings.scoring) for ex in self.available}
        self.calibrators = {ex.key: Calibrator(ex, settings.game, settings.scoring) for ex in self.available}
        self.exercises: list[Exercise] = list(exercises)
        # Cámara a la que vuelve el botón "Cámara" del menú
        src = settings.camera.source
        self._camera_source = 0 if src is None or is_video_file(src) else src
        self.window = settings.game.window_name
        self.recorder: Recorder | None = None
        if settings.game.record_path:
            self.recorder = Recorder(settings.game.record_path, settings.game.record_fps)
        self._fullscreen = False
        self._running = True
        self._window_open = False
        self._history_from_summary = False
        self._mouse: tuple[int, int] | None = None
        self._click: tuple[int, int] | None = None
        self._buttons: list[tuple[str, tuple[int, int, int, int]]] = []
        self._last_frame_id: int | None = None
        self._pose = None
        self._video_end_handled = False
        self._toast: tuple[str, tuple[int, int, int], float] | None = None
        self._picker = PickerInfo()
        self._history: list[HistoryEntry] = []
        self._previous: HistoryEntry | None = None
        self._last_tick: int | None = None
        self._reset_game(rewind=False)

    # --- Fuente: cámara o vídeo -----------------------------------------------
    @property
    def is_video(self) -> bool:
        return bool(getattr(self.camera, "is_video", False))

    @property
    def has_source(self) -> bool:
        return self.camera is not None and not getattr(self.camera, "is_null", False)

    @property
    def video_ended(self) -> bool:
        return self.is_video and bool(self.camera.ended)

    def _default_open_source(self, source: int | str) -> Camera:
        return Camera(dataclasses.replace(self.settings.camera, source=source))

    def _switch_source(self, source: int | str) -> bool:
        try:
            new = self._open_source(source)
        except (RuntimeError, ValueError) as e:
            self._show_toast(str(e), RED)
            return False
        old, self.camera = self.camera, new
        if old is not None:
            old.release()
        self._last_frame_id = None
        self._pose = None
        self._video_end_handled = False
        self.state = TrackerState()
        if self._window_open:
            cv2.resizeWindow(self.window, *self._window_size())
        self._show_toast(f"Fuente: {getattr(new, 'label', source)}", GREEN)
        return True

    def _now(self) -> float:
        """Reloj del juego: el del vídeo si se reproduce un archivo, si no el real."""
        return self.camera.clock() if self.is_video else time.monotonic()

    def _on_video_end(self) -> None:
        print("[juego] Fin del vídeo")
        if self.phase in (Phase.COUNTDOWN, Phase.CALIBRATION, Phase.ACTIVE):
            self._finish_exercise(skipped=self.session.count == 0)

    def _show_toast(self, text: str, color=ORANGE) -> None:
        self._toast = (text, color, time.monotonic() + TOAST_SECONDS)

    # --- Estado --------------------------------------------------------------
    @property
    def tracker(self) -> ExerciseTracker:
        return self.trackers[self.exercises[self.index].key]

    @property
    def calibrator(self) -> Calibrator:
        return self.calibrators[self.exercises[self.index].key]

    @property
    def session(self) -> ExerciseSession:
        return self.sessions[self.index]

    def _reset_game(self, rewind: bool = True) -> None:
        if rewind and self.is_video:
            self.camera.rewind()
            self._last_frame_id = None
            self._pose = None
            self._video_end_handled = False
        self.phase = Phase.MENU
        self.index = 0
        self.sessions: list[ExerciseSession] = []
        self.saved_path: Path | None = None
        self.state = TrackerState()
        self.feedback: Feedback | None = None
        self._phase_start = self._now()
        self._ready_since: float | None = None

    def _go(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_start = self._now()
        self._ready_since = None
        self._last_tick = None

    def _start_game(self) -> None:
        if not self.has_source:
            self._show_toast("Elige una fuente: cámara (C) o vídeo (V)")
            return
        self.exercises = [ex for ex in self.available if ex.key in self.selected]
        if not self.exercises:
            self._show_toast("Elige al menos un ejercicio")
            return
        self.sessions = []
        self.saved_path = None
        self._previous = None
        if self.is_video:
            self.camera.play()
        self._start_exercise(0)

    def _start_exercise(self, index: int) -> None:
        self.index = index
        ex = self.exercises[index]
        ex.reset_reference()                  # La calibración es de cada partida
        self.sessions.append(ExerciseSession(ex))
        self.tracker.reset()
        self.calibrator.reset()
        self.feedback = None
        self._go(Phase.INSTRUCTIONS)

    def _begin_active(self) -> None:
        ex = self.exercises[self.index]
        self.session.calibrated = ex.calibrated
        self.session.reference = {
            "rest": {f.name: round(float(v), 1) for f, v in zip(ex.features, ex.rest_vector)},
            "peak": {f.name: round(float(v), 1) for f, v in zip(ex.features, ex.peak_vector)},
        }
        self.tracker.counter.reset()
        self.sound.play("go")
        self._go(Phase.ACTIVE)

    def _finish_exercise(self, skipped: bool = False) -> None:
        if self.phase is Phase.ACTIVE:
            self.session.duration = self._now() - self._phase_start
        self.session.skipped = skipped
        if not skipped:
            self.sound.play("done")
        self._go(Phase.EXERCISE_RESULT)

    def _next(self) -> None:
        if self.video_ended:
            # Sin más vídeo no se pueden hacer los ejercicios que faltan: se saltan
            for i in range(self.index + 1, len(self.exercises)):
                self.sessions.append(ExerciseSession(self.exercises[i], skipped=True))
            self.index = len(self.exercises) - 1
        if self.index + 1 < len(self.exercises):
            self._start_exercise(self.index + 1)
        else:
            if self.settings.game.save_results:
                history = load_history(RESULTS_DIR)
                self._previous = history[-1] if history else None
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
            if self._is_resting():
                self.calibrator.add_rest(self.state.measurement.features)
                self._ready_since = self._ready_since or now
                if now - self._ready_since >= g.ready_hold_seconds:
                    self._go(Phase.COUNTDOWN)
            else:
                self._ready_since = None

        elif self.phase is Phase.COUNTDOWN:
            self.state = self.tracker.update(pose, now, counting=False)
            if self._is_resting():
                self.calibrator.add_rest(self.state.measurement.features)
            tick = int(np.ceil(g.countdown_seconds - elapsed))
            if tick != self._last_tick and tick > 0:
                self._last_tick = tick
                self.sound.play("tick")
            if elapsed >= g.countdown_seconds:
                if g.calibrate:
                    self._go(Phase.CALIBRATION)
                else:
                    self._begin_active()

        elif self.phase is Phase.CALIBRATION:
            self.state = self.tracker.update(pose, now, counting=False)
            st = self.state
            if st.posture_ok and st.measurement is not None and st.progress is not None:
                status = self.calibrator.update(now, st.measurement.features, st.progress)
                if status == "done":
                    self._on_calibrated(now)
                    return
                if status == "retry":
                    self.sound.play("partial")
                    self.feedback = Feedback(["Otra vez", "Repetición de prueba incompleta",
                                              self.exercises[self.index].rom_feedback],
                                             ORANGE, now + g.feedback_seconds)
            if elapsed >= g.calibration_timeout:
                self.feedback = Feedback(["Referencia estándar", "No he visto la repetición de prueba"],
                                         ACCENT, now + g.feedback_seconds)
                self._begin_active()

        elif self.phase is Phase.ACTIVE:
            self.state = self.tracker.update(pose, now, counting=True)
            ev = self.state.event
            if ev is not None:
                self._on_event(ev, now)
            if elapsed >= g.exercise_duration or self.session.count >= g.target_reps:
                self._finish_exercise()

    def _is_resting(self) -> bool:
        st = self.state
        return (st.posture_ok and st.progress is not None and st.measurement is not None
                and st.progress <= self.settings.game.ready_progress)

    def _on_calibrated(self, now: float) -> None:
        result = self.calibrator.result
        result.apply(self.exercises[self.index])
        lines = ["¡Referencia guardada!", "Ahora, las repeticiones de verdad"]
        if result.fallback:
            lines.append("Estándar para: " + ", ".join(result.fallback))
        self.feedback = Feedback(lines, GREEN, now + self.settings.game.feedback_seconds)
        self.sound.play("calibrated")
        print(f"[{self.exercises[self.index].key}] calibrado: reposo {np.round(result.rest, 1)}, "
              f"máximo {np.round(result.peak, 1)}" + (f", estándar: {result.fallback}" if result.fallback else ""))
        self._begin_active()

    def _on_event(self, ev, now: float) -> None:
        until = now + self.settings.game.feedback_seconds
        if ev.kind == "rep" and ev.score is not None:
            self.session.reps.append(ev.score)
            s = ev.score
            title = "¡Perfecta!" if s.total >= 90 else f"+{s.total:.0f}"
            self.feedback = Feedback([title] + s.feedback[:2], score_color(s.total), until)
            self.sound.play("perfect" if s.total >= 90 else "rep")
            print(f"[{self.session.exercise.key}] rep {self.session.count}: {s.total:.1f} "
                  f"(rango {s.rom:.2f}, trayectoria {s.trajectory:.2f}, tempo {s.tempo:.2f}, "
                  f"técnica {s.form:.2f}, {s.duration:.1f}s)")
        elif ev.kind == "partial":
            self.session.partial_reps += 1
            self.feedback = Feedback(["Incompleta", self.session.exercise.rom_feedback], ORANGE, until)
            self.sound.play("partial")

    # --- Dibujo --------------------------------------------------------------
    def _menu_info(self) -> MenuInfo:
        kind = "video" if self.is_video else ("camera" if self.has_source else "none")
        label = getattr(self.camera, "label", "Sin cámara") if self.camera is not None else "Sin cámara"
        g = self.settings.game
        return MenuInfo(self.available, self.selected, label, kind, g.calibrate,
                        g.exercise_duration, g.target_reps)

    def _draw(self, frame: np.ndarray, pose, now: float) -> np.ndarray:
        c = Canvas(frame, self.renderer)
        c.mouse = self._mouse
        g, sc = self.settings.game, self.settings.scoring
        live = self.phase in LIVE_PHASES
        ex = self.exercises[self.index] if live or self.phase is Phase.EXERCISE_RESULT else None
        draw_skeleton(c, pose, ex if live else None, self.state if live else None)

        if self.phase is Phase.MENU:
            draw_menu(c, self._menu_info())
        elif self.phase is Phase.PICK_VIDEO:
            draw_video_picker(c, self._picker)
        elif self.phase is Phase.HISTORY:
            draw_history(c, self._history)
        elif self.phase is Phase.INSTRUCTIONS:
            frac = 0.0 if self._ready_since is None else (now - self._ready_since) / g.ready_hold_seconds
            draw_instructions(c, ex, self.index, len(self.exercises), self.state, min(frac, 1.0), g.calibrate)
        elif self.phase is Phase.COUNTDOWN:
            draw_countdown(c, ex, g.countdown_seconds - (now - self._phase_start), g.calibrate)
        elif self.phase is Phase.CALIBRATION:
            cal = self.calibrator
            draw_calibration(c, ex, self.state, cal.recording, cal.max_progress,
                             max(g.calibration_timeout - (now - self._phase_start), 0.0),
                             self.feedback, now, sc.rest_threshold, sc.peak_threshold)
        elif self.phase is Phase.ACTIVE:
            time_left = max(g.exercise_duration - (now - self._phase_start), 0.0)
            draw_hud(c, self.session, self.state, time_left, g.target_reps, self.feedback, now,
                     sc.rest_threshold, sc.peak_threshold, self.is_video)
        elif self.phase is Phase.EXERCISE_RESULT:
            draw_exercise_result(c, self.session, self.index, len(self.exercises))
        elif self.phase is Phase.SUMMARY:
            draw_summary(c, self.sessions, self.saved_path, self._previous)
        if self.is_video and self.phase not in (Phase.MENU, Phase.PICK_VIDEO, Phase.HISTORY):
            draw_video_status(c, self.camera.position, self.camera.duration,
                              self.camera.paused, self.camera.ended, show_hint=live)
        if self._toast is not None:
            text, color, until = self._toast
            if time.monotonic() < until:
                draw_toast(c, text, color)
            else:
                self._toast = None
        out = c.finish()
        self._buttons = list(c.buttons)
        return out

    # --- Acciones (teclado y ratón) ------------------------------------------
    def _do(self, action: str) -> None:
        """Ejecuta una acción. Los botones y las teclas acaban todos aquí."""
        if action == "quit":
            self._running = False
            return
        if action == "fullscreen":
            self._fullscreen = not self._fullscreen
            if self._window_open:
                cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL)
            return
        if action == "menu":
            self._reset_game()
            return
        if action == "record":
            self._toggle_recording()
            return
        if action == "pause":
            if self.is_video and self.phase not in (Phase.MENU, Phase.PICK_VIDEO, Phase.HISTORY):
                self.camera.toggle_pause()
            return

        phase = self.phase
        if phase is Phase.MENU:
            if action in ("start", "advance"):
                self._start_game()
            elif action.startswith("toggle:"):
                key = action.split(":", 1)[1]
                self.selected ^= {key}
            elif action == "toggle_calibrate":
                self.settings.game.calibrate = not self.settings.game.calibrate
            elif action == "camera":
                if self.has_source and not self.is_video:
                    self._show_toast(f"Ya estás usando {self.camera.label}", GREEN)
                else:
                    self._switch_source(self._camera_source)
            elif action == "pick_video":
                folders = [VIDEOS_DIR, PROJECT_ROOT, Path.cwd()]
                if self.is_video:
                    folders.insert(0, Path(self.camera.source).parent)
                self._picker = PickerInfo(list_videos(folders), VIDEOS_DIR, tk_available())
                self._go(Phase.PICK_VIDEO)
            elif action == "history":
                self._open_history()
        elif phase is Phase.PICK_VIDEO:
            if action.startswith("video:"):
                i = int(action.split(":", 1)[1])
                if i < len(self._picker.videos) and self._switch_source(str(self._picker.videos[i])):
                    self._go(Phase.MENU)
            elif action == "browse":
                start = Path(self.camera.source).parent if self.is_video else VIDEOS_DIR
                path = ask_video_file(start if start.is_dir() else Path.home())
                if path and self._switch_source(path):
                    self._go(Phase.MENU)
            elif action == "back":
                self._go(Phase.MENU)
        elif phase is Phase.HISTORY:
            if action in ("back", "advance"):
                self._go(Phase.SUMMARY if self._history_from_summary else Phase.MENU)
        elif phase is Phase.INSTRUCTIONS:
            if action == "advance":
                self._go(Phase.COUNTDOWN)
            elif action == "skip":
                self._finish_exercise(skipped=True)
        elif phase is Phase.CALIBRATION:
            if action == "advance":            # Usar la referencia estándar
                self._begin_active()
            elif action == "skip":
                self._finish_exercise(skipped=True)
        elif phase in (Phase.COUNTDOWN, Phase.ACTIVE) and action in ("advance", "skip"):
            self._finish_exercise(skipped=action == "skip" and self.session.count == 0)
        elif phase is Phase.EXERCISE_RESULT and action in ("advance", "skip"):
            self._next()
        elif phase is Phase.SUMMARY:
            if action == "advance":
                self._reset_game()
            elif action == "history":
                self._open_history(from_summary=True)

    def _open_history(self, from_summary: bool = False) -> None:
        self._history = load_history(RESULTS_DIR)
        self._history_from_summary = from_summary
        self._go(Phase.HISTORY)

    def _toggle_recording(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            if self.recorder.elapsed > 0:
                self._show_toast(f"Grabación guardada en {self.recorder.path}", GREEN)
            self.recorder = None
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.recorder = Recorder(RESULTS_DIR / f"grabacion_{stamp}.mp4", self.settings.game.record_fps)
        self._show_toast("Grabando la ventana (G para parar)", RED)

    def _handle_key(self, key: int) -> None:
        """Traduce una tecla a una acción."""
        if key == 255 or key == -1:
            return
        phase = self.phase
        if key == KEY_ESC:
            self._do("back" if phase in (Phase.PICK_VIDEO, Phase.HISTORY) else "quit")
            return
        ch = chr(key).lower() if 32 < key < 127 else ""
        if key in KEY_ENTER or key == KEY_SPACE:
            self._do("advance")
            return
        global_keys = {"q": "quit", "f": "fullscreen", "r": "menu", "p": "pause", "g": "record", "n": "skip"}
        if phase is Phase.MENU:
            menu_keys = {"c": "camera", "v": "pick_video", "h": "history", "k": "toggle_calibrate"}
            if ch in menu_keys:
                self._do(menu_keys[ch])
                return
            if ch.isdigit() and 1 <= int(ch) <= len(self.available):
                self._do(f"toggle:{self.available[int(ch) - 1].key}")
                return
        elif phase is Phase.PICK_VIDEO:
            if ch.isdigit() and int(ch) >= 1:
                self._do(f"video:{int(ch) - 1}")
                return
            if ch == "b":
                self._do("browse")
                return
        elif phase is Phase.SUMMARY and ch == "h":
            self._do("history")
            return
        if ch in global_keys:
            self._do(global_keys[ch])

    def _on_mouse(self, event, x, y, flags, param) -> None:
        self._mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def _handle_click(self) -> None:
        if self._click is None:
            return
        x, y = self._click
        self._click = None
        for action, (x1, y1, x2, y2) in reversed(self._buttons):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.sound.play("click")
                self._do(action)
                return

    # --- Bucle ---------------------------------------------------------------
    def _window_size(self) -> tuple[int, int]:
        cam = self.settings.camera
        if not self.is_video:
            return cam.width, cam.height
        w, h = self.camera.frame_size
        scale = min(cam.width / w, cam.height / h, 1.0)   # Cabe en la resolución pedida
        return int(w * scale), int(h * scale)

    def _draw_rec_indicator(self, frame: np.ndarray) -> None:
        """Punto rojo de grabación (después de grabar el frame, para que no salga en el vídeo)."""
        h, w = frame.shape[:2]
        s = h / 720
        cx, cy = int(w / 2 - 50 * s), int(24 * s)
        cv2.circle(frame, (cx, cy), max(int(8 * s), 2), (40, 40, 230), -1, cv2.LINE_AA)
        secs = int(self.recorder.elapsed)
        cv2.putText(frame, f"REC {secs // 60}:{secs % 60:02d}", (cx + int(16 * s), cy + int(7 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (40, 40, 230), max(int(2 * s), 1), cv2.LINE_AA)

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self._window_open = True
        cv2.resizeWindow(self.window, *self._window_size())
        cv2.setMouseCallback(self.window, self._on_mouse)
        try:
            while self._running:
                frame = self.camera.read()
                if frame is None:
                    if (cv2.waitKey(10) & 0xFF) == KEY_ESC:
                        break
                    continue
                now = self._now()
                fresh = self.camera.frame_id != self._last_frame_id
                if fresh:
                    # Sólo se detecta y se actualiza con frames nuevos (no en pausa)
                    self._last_frame_id = self.camera.frame_id
                    if self.has_source and self.detector is not None:
                        self._pose = self.detector.detect(frame, self.camera.timestamp_ms())
                    else:
                        self._pose = None
                    self._update(self._pose, now)
                if self.video_ended and not self._video_end_handled:
                    self._video_end_handled = True
                    self._on_video_end()
                out = self._draw(frame, self._pose, now)
                if self.recorder is not None:
                    try:
                        self.recorder.write(out)
                    except RuntimeError as e:
                        self._show_toast(str(e), RED)
                        self.recorder = None
                    else:
                        out = out.copy()
                        self._draw_rec_indicator(out)
                cv2.imshow(self.window, out)

                # Un único waitKey por frame (más largo si no hay frames nuevos)
                self._handle_key(cv2.waitKey(1 if fresh else 30) & 0xFF)
                self._handle_click()
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break   # Ventana cerrada con la X
        finally:
            if self.recorder is not None:
                self.recorder.close()
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            self._window_open = False
