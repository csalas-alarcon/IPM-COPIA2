"""
Pantallas del juego: una función draw_* por pantalla, más los elementos comunes
(esqueleto, barra de progreso, avisos, indicador del vídeo). Todas dibujan en un
Canvas; los botones llevan el nombre de la acción que recibe Game._do().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.counter import TrackerState
from exercises.base import Exercise
from ui.canvas import (ACCENT, BLUE, GREEN, GREY, ORANGE, RED, WHITE, Canvas,
                       lerp_color, score_color)
from ui.widgets import button, checkbox, draw_line_chart, draw_rep_chart
from vision.pose import POSE_CONNECTIONS, PoseFrame, side_landmark


@dataclass
class Feedback:
    lines: list[str]
    color: tuple[int, int, int]
    until: float


@dataclass
class MenuInfo:
    """Lo que necesita el menú para dibujarse."""
    exercises: list[Exercise]              # Todos los disponibles, en orden
    selected: set[str]                     # Claves de los elegidos
    source_label: str                      # "Cámara 0", "demo.mp4", "Sin cámara"
    source_kind: str                       # "camera" | "video" | "none"
    calibrate: bool
    duration: float
    target: int


@dataclass
class PickerInfo:
    videos: list[Path] = field(default_factory=list)
    folder: Path | None = None
    can_browse: bool = False


# --- Elementos comunes ---
def draw_skeleton(c: Canvas, pose: PoseFrame | None, exercise: Exercise | None = None,
                  state: TrackerState | None = None, min_vis: float = 0.3) -> None:
    if pose is None:
        return
    pts, vis = pose.image_xy, pose.visibility
    for a, b in POSE_CONNECTIONS:
        if vis[a] >= min_vis and vis[b] >= min_vis:
            c.line_px(pts[a], pts[b], (235, 235, 235), 2)
    for i in range(11, len(pts)):          # se omiten los puntos de la cara
        if vis[i] >= min_vis:
            c.circle_px(pts[i], 4, BLUE)

    # Resalta las articulaciones que se están midiendo (azul = reposo, verde = máximo)
    if exercise is not None and state is not None and state.measurement is not None:
        color = lerp_color(BLUE, GREEN, state.progress or 0.0)
        for side in state.measurement.sides:
            for f in exercise.features:
                a, b, cc = (side_landmark(side, j) for j in f.joints)
                c.line_px(pts[a], pts[b], color, 5)
                c.line_px(pts[b], pts[cc], color, 5)
                c.circle_px(pts[b], 9, color)
                c.circle_px(pts[b], 9, WHITE, 2)


def draw_progress_bar(c: Canvas, progress: float | None, rest_th: float, peak_th: float) -> None:
    x1, x2 = c.W - 70, c.W - 40
    y1, y2 = 130, c.H - 150
    c.panel(x1 - 14, y1 - 44, x2 + 14, y2 + 40, alpha=0.55)
    c.text("MÁX", (x1 + x2) / 2, y1 - 22, 16, GREY, "mm", bold=True)
    c.text("REP.", (x1 + x2) / 2, y2 + 20, 16, GREY, "mm", bold=True)
    c.rect(x1, y1, x2, y2, (60, 60, 60))
    if progress is not None:
        p = float(np.clip(progress, 0, 1))
        top = y2 - (y2 - y1) * p
        c.rect(x1, top, x2, y2, lerp_color(BLUE, GREEN, p))
    for th, col in ((rest_th, BLUE), (peak_th, GREEN)):
        y = y2 - (y2 - y1) * th
        c.line(x1 - 8, y, x2 + 8, y, col, 2)


def draw_banner(c: Canvas, text: str, color=RED) -> None:
    half = min(c.text_width(text, 24, True) / 2 + 30, c.W / 2 - 20)
    c.panel(c.W / 2 - half, 215, c.W / 2 + half, 265, alpha=0.75, border=color)
    c.text(text, c.W / 2, 240, 24, WHITE, "mm", bold=True)


def draw_feedback(c: Canvas, feedback: Feedback | None, now: float) -> None:
    """Mensaje grande en el centro (tras una repetición, al calibrar...)."""
    if feedback is None or now >= feedback.until:
        return
    y = c.H / 2 - 20 * len(feedback.lines)
    for i, line in enumerate(feedback.lines):
        size = 54 if i == 0 else 26
        c.text(line, c.W / 2, y, size, feedback.color if i == 0 else WHITE, "mm", bold=True, stroke=3)
        y += 60 if i == 0 else 36


def draw_toast(c: Canvas, text: str, color=ORANGE) -> None:
    """Aviso corto en la parte inferior (p. ej. no se pudo abrir la cámara)."""
    half = min(c.text_width(text, 18, True) / 2 + 24, c.W / 2 - 10)
    c.panel(c.W / 2 - half, c.H - 40, c.W / 2 + half, c.H - 8, alpha=0.85, border=color)
    c.text(text, c.W / 2, c.H - 24, 18, WHITE, "mm", bold=True)


def _mmss(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def draw_video_status(c: Canvas, position: float, duration: float, paused: bool, ended: bool,
                      show_hint: bool = True) -> None:
    """Indicador del vídeo (arriba a la derecha) y, debajo, aviso de pausa o de fin del vídeo.

    El indicador es también un botón: al hacer clic pausa / reanuda.
    show_hint: mostrar el aviso de fin del vídeo (sólo tiene sentido durante un ejercicio).
    """
    if ended and duration > 0:
        position = duration
    label = "PAUSA" if paused and not ended else ("FIN" if ended else "VÍDEO")
    times = _mmss(position) + (f" / {_mmss(duration)}" if duration > 0 else "")
    text = f"{label}  {times}   ·   P: pausa"
    w = c.text_width(text, 16, True)
    x1 = c.W - w - 40
    c.panel(x1, 14, c.W - 14, 50, alpha=0.75 if c.hovered(x1, 14, c.W - 14, 50) else 0.6)
    c.text(text, c.W - 27, 32, 16, ORANGE if paused else WHITE, "rm", bold=True)
    if duration > 0:
        frac = float(np.clip(position / duration, 0, 1))
        c.rect(x1, 47, x1 + (w + 26) * frac, 50, ACCENT)
    if not ended:
        c.buttons.append(("pause", (c.px(x1), c.px(14), c.px(c.W - 14), c.px(50))))

    # Aviso bajo el indicador (no tapa los paneles de las pantallas)
    if ended and show_hint:
        msg, col = "Fin del vídeo  ·  R: reiniciar  ·  N: saltar", ACCENT
    elif paused and not ended:
        msg, col = "PAUSA  ·  P: continuar", ORANGE
    else:
        return
    w = c.text_width(msg, 17, True)
    c.panel(c.W - w - 46, 56, c.W - 14, 84, alpha=0.8, border=col)
    c.text(msg, c.W - 30, 70, 17, WHITE, "rm", bold=True)


# --- Menú, selector de vídeo e historial ---
def draw_menu(c: Canvas, m: MenuInfo) -> None:
    c.dim(0.65)
    cx = c.W / 2
    hw = min(330, c.W / 2 - 20)
    c.text("POSE TRAINER", cx, 66, 60, ACCENT, "mm", bold=True, stroke=2)
    c.text("Cuenta tus repeticiones y puntúa tu técnica", cx, 118, 22, WHITE, "mm")

    # Fuente: cámara o vídeo
    c.panel(cx - hw, 142, cx + hw, 240, alpha=0.6)
    col = GREEN if m.source_kind != "none" else ORANGE
    c.text("Fuente:", cx - hw + 20, 166, 18, GREY, "lm", bold=True)
    c.text(m.source_label, cx - hw + 100, 166, 20, col, "lm", bold=True)
    button(c, "camera", "Cámara (C)", cx - hw + 20, 186, cx - 10, 228,
           color=GREEN if m.source_kind == "camera" else ACCENT)
    button(c, "pick_video", "Abrir vídeo… (V)", cx + 10, 186, cx + hw - 20, 228,
           color=GREEN if m.source_kind == "video" else ACCENT)

    # Ejercicios
    c.text("Ejercicios", cx - hw, 262, 18, GREY, "lm", bold=True)
    y = 278
    for i, ex in enumerate(m.exercises):
        checkbox(c, f"toggle:{ex.key}", f"{i + 1}.  {ex.name}", ex.key in m.selected,
                 cx - hw, y, cx + hw, y + 44, size=22, detail=f"tecla {i + 1}")
        y += 52
    checkbox(c, "toggle_calibrate", "Repetición de prueba (calibración)", m.calibrate,
             cx - hw, y + 6, cx + hw, y + 46, size=18, detail="tecla K")
    y += 70
    c.text(f"Hasta {m.target} repeticiones o {int(m.duration)} s por ejercicio", cx, y, 18, GREY, "mm")

    can_start = bool(m.selected) and m.source_kind != "none"
    button(c, "start", "EMPEZAR  (ENTER)", cx - 200, y + 22, cx + 200, y + 80, color=GREEN,
           primary=True, enabled=can_start, size=28)
    button(c, "history", "Historial (H)", cx - 200, y + 92, cx - 8, y + 132)
    button(c, "quit", "Salir (ESC)", cx + 8, y + 92, cx + 200, y + 132, color=GREY)
    if not m.selected:
        draw_toast(c, "Elige al menos un ejercicio")
    elif m.source_kind == "none":
        draw_toast(c, "No hay cámara: abre un vídeo (V)")


def draw_video_picker(c: Canvas, info: PickerInfo) -> None:
    c.dim(0.7)
    cx = c.W / 2
    hw = min(380, c.W / 2 - 20)
    c.text("Elige un vídeo", cx, 70, 44, ACCENT, "mm", bold=True)
    if info.folder is not None:
        c.text(f"Carpeta: {info.folder}", cx, 112, 15, GREY, "mm")
    y = 140
    if not info.videos:
        c.panel(cx - hw, y, cx + hw, y + 120, alpha=0.6)
        c.text("No hay vídeos en la carpeta.", cx, y + 40, 22, WHITE, "mm", bold=True)
        c.text("Copia tus vídeos en la carpeta videos/ del proyecto", cx, y + 78, 18, GREY, "mm")
    for i, path in enumerate(info.videos[:8]):
        button(c, f"video:{i}", f"{i + 1}.  {path.name}", cx - hw, y, cx + hw, y + 44, size=19)
        y += 52
    button(c, "browse", "Buscar en el ordenador… (B)", cx - hw, c.H - 110, cx - 10, c.H - 66,
           enabled=info.can_browse)
    button(c, "back", "Volver (ESC)", cx + 10, c.H - 110, cx + hw, c.H - 66, color=GREY)
    if not info.can_browse:
        c.text("Para buscar en el ordenador hace falta tkinter (en Linux: sudo apt install python3-tk)",
               cx, c.H - 40, 14, GREY, "mm")


def _nice_ceil(v: float) -> float:
    """Redondea hacia arriba a 1, 2 o 5 × 10^n (para la escala de las gráficas)."""
    if v <= 0:
        return 1.0
    exp = 10 ** np.floor(np.log10(v))
    for m in (1, 2, 5, 10):
        if v <= m * exp:
            return float(m * exp)
    return float(10 * exp)


def draw_history(c: Canvas, entries: Sequence) -> None:
    """entries: game.history.HistoryEntry, de la más antigua a la más reciente."""
    c.dim(0.75)
    cx = c.W / 2
    hw = min(560, c.W / 2 - 20)
    c.text("HISTORIAL", cx, 52, 44, ACCENT, "mm", bold=True)
    if not entries:
        c.text("Todavía no hay partidas guardadas.", cx, c.H / 2 - 20, 26, WHITE, "mm", bold=True)
        c.text("Se guardan al llegar al resumen de una partida.", cx, c.H / 2 + 20, 20, GREY, "mm")
    else:
        recent = list(entries)[-12:]
        labels = [e.date.strftime("%d/%m") for e in recent]
        draw_line_chart(c, cx - hw, 88, cx - 8, 318,
                        [("Puntos", [e.total_points for e in recent], GREEN)],
                        _nice_ceil(max(e.total_points for e in recent) * 1.05), "Puntos por partida", labels)
        draw_line_chart(c, cx + 8, 88, cx + hw, 318,
                        [("Técnica", [e.average_score for e in recent], ACCENT)],
                        100, "Puntuación media por repetición", labels)
        rows = list(reversed(recent))[:6]
        c.panel(cx - hw, 334, cx + hw, 378 + 42 * len(rows), alpha=0.7)
        cols = (("Fecha", cx - hw + 20, "lm"), ("Ejercicios", cx - hw + 200, "lm"),
                ("Reps", cx + hw - 290, "mm"), ("Media", cx + hw - 170, "mm"), ("Puntos", cx + hw - 20, "rm"))
        for label, x, anchor in cols:
            c.text(label, x, 354, 16, GREY, anchor, bold=True)
        for i, e in enumerate(rows):
            y = 392 + 42 * i
            names = ", ".join(e.exercises) or "—"
            if c.text_width(names, 17) > hw - 120:
                names = f"{len(e.exercises)} ejercicios"
            c.text(e.date.strftime("%d/%m/%Y %H:%M"), cols[0][1], y, 17, WHITE, "lm")
            c.text(names, cols[1][1], y, 17, WHITE, "lm")
            c.text(str(e.reps), cols[2][1], y, 18, WHITE, "mm", bold=True)
            c.text(f"{e.average_score:.0f}", cols[3][1], y, 18, score_color(e.average_score), "mm", bold=True)
            c.text(str(e.total_points), cols[4][1], y, 18, GREEN, "rm", bold=True)
    button(c, "back", "Volver (ESC)", cx - 140, c.H - 64, cx + 140, c.H - 22, color=GREY)


# --- Ejercicio: instrucciones, cuenta atrás, repetición de prueba y HUD ---
def draw_instructions(c: Canvas, ex: Exercise, index: int, total: int,
                      state: TrackerState, ready_fraction: float, calibrate: bool = False) -> None:
    c.dim(0.45)
    c.panel(40, 40, 800, c.H - 30, alpha=0.7)
    c.text(f"EJERCICIO {index + 1} / {total}", 80, 80, 22, ACCENT, "la", bold=True)
    c.text(ex.name, 80, 112, 50, WHITE, "la", bold=True)
    y = 200
    for i, line in enumerate(ex.instructions):
        c.text(f"{i + 1}. {line}", 80, y, 22, WHITE, "la")
        y += 40
    c.text("Cámara:", 80, y + 10, 20, ACCENT, "la", bold=True)
    c.text(ex.camera_tip, 80, y + 38, 20, GREY, "la")
    if calibrate:
        c.text("Después de la cuenta atrás: 1 repetición de prueba, lenta y completa.",
               80, y + 76, 19, GREEN, "la", bold=True)

    # Estado de la detección + anillo de "preparado"
    cx, cy = 420, c.H - 210
    if not state.detected or not state.measurable:
        msg, col = state.issue or "Buscando…", RED
    elif not state.posture_ok:
        msg, col = state.issue or ex.rest_hint, ORANGE
    elif ready_fraction > 0:
        msg, col = "¡Perfecto! Mantén la posición…", GREEN
    else:
        msg, col = ex.rest_hint, ACCENT
    c.arc(cx, cy, 56, 1.0, (70, 70, 70), 10)
    c.arc(cx, cy, 56, ready_fraction, GREEN, 10)
    c.text(f"{int(ready_fraction * 100)}%", cx, cy, 26, WHITE, "mm", bold=True)
    c.text(msg, cx, cy + 84, 21, col, "mm", bold=True)
    button(c, "advance", "Empezar ya (ENTER)", 110, c.H - 92, 410, c.H - 50, color=GREEN)
    button(c, "skip", "Saltar ejercicio (N)", 430, c.H - 92, 730, c.H - 50, color=GREY)


def draw_countdown(c: Canvas, ex: Exercise, remaining: float, calibrate: bool = False) -> None:
    c.dim(0.35)
    n = max(int(np.ceil(remaining)), 1)
    c.text(ex.name.upper(), c.W / 2, c.H / 2 - 150, 40, WHITE, "mm", bold=True, stroke=2)
    c.text(str(n), c.W / 2, c.H / 2, 200, ACCENT, "mm", bold=True, stroke=4)
    sub = "Primero: 1 repetición de prueba" if calibrate else "¡Prepárate!"
    c.text(sub, c.W / 2, c.H / 2 + 150, 34, WHITE, "mm", stroke=2)


def draw_calibration(c: Canvas, ex: Exercise, state: TrackerState, recording: bool,
                     max_progress: float, time_left: float, feedback: Feedback | None,
                     now: float, rest_th: float, peak_th: float) -> None:
    c.dim(0.2)
    x2 = min(660, c.W - 120)
    c.panel(20, 20, x2, 250, alpha=0.7, border=ACCENT)
    c.text("REPETICIÓN DE PRUEBA", 40, 36, 22, ACCENT, "la", bold=True)
    c.text(f"{int(np.ceil(time_left))} s", x2 - 20, 36, 22, RED if time_left < 6 else WHITE, "ra", bold=True)
    c.text(ex.name, 40, 68, 36, WHITE, "la", bold=True)
    c.text("Haz 1 repetición lenta y completa.", 40, 118, 21, WHITE, "la")
    c.text("Así aprendo tu reposo, tu máximo y cómo te mueves.", 40, 148, 18, GREY, "la")
    if recording:
        c.text(f"Grabando…  profundidad {max_progress * 100:.0f}%", 40, 190, 22, GREEN, "la", bold=True)
    elif state.issue and (not state.measurable or not state.posture_ok):
        c.text(state.issue, 40, 190, 20, ORANGE, "la", bold=True)
    else:
        c.text("Cuando quieras: empieza la repetición", 40, 190, 21, ACCENT, "la", bold=True)
    draw_progress_bar(c, state.progress, rest_th, peak_th)
    draw_feedback(c, feedback, now)
    cx = c.W / 2
    button(c, "advance", "Usar referencia estándar (ENTER)", cx - 390, c.H - 72, cx - 10, c.H - 28, size=18)
    button(c, "skip", "Saltar ejercicio (N)", cx + 10, c.H - 72, cx + 270, c.H - 28, color=GREY, size=18)


def draw_hud(c: Canvas, session, state: TrackerState, time_left: float, target: int,
             feedback: Feedback | None, now: float, rest_th: float, peak_th: float,
             is_video: bool = False) -> None:
    ex = session.exercise
    # Marcador superior izquierdo
    c.panel(20, 20, 400, 214, alpha=0.65)
    name_size = 28 if c.text_width(ex.name, 28, True) <= 250 else 22   # "Sentadilla búlgara" no pisa el tiempo
    c.text(ex.name, 40, 36 + (28 - name_size) / 2, name_size, ACCENT, "la", bold=True)
    count = str(session.count)
    c.text(count, 40, 76, 80, WHITE, "la", bold=True)
    c.text(f"/ {target} reps", 40 + c.text_width(count, 80, True) + 12, 120, 26, GREY, "la")
    c.text(f"{int(np.ceil(time_left))} s", 380, 36, 28, RED if time_left < 10 else WHITE, "ra", bold=True)
    c.text(f"{session.points} pts", 380, 160, 26, GREEN, "ra", bold=True)
    if session.partial_reps:
        c.text(f"incompletas: {session.partial_reps}", 40, 172, 16, ORANGE, "la")
    ref = "referencia: tu repetición de prueba" if ex.calibrated else "referencia: estándar"
    c.text(ref, 40, 196, 15, GREY, "la")

    draw_progress_bar(c, state.progress, rest_th, peak_th)

    # Aviso en vivo
    if state.issue:
        draw_banner(c, state.issue, ORANGE if state.measurable else RED)

    # Última repetición, con su gráfica frente a la referencia
    if session.reps:
        last = session.reps[-1]
        x2 = min(720, c.W - 385)
        c.panel(20, c.H - 130, x2, c.H - 20, alpha=0.65)
        c.text("Última rep", 40, c.H - 112, 18, GREY, "la")
        c.text(f"{last.total:.0f}", 40, c.H - 90, 56, score_color(last.total), "la", bold=True)
        c.text(f"Rango {last.rom * 100:.0f}", 170, c.H - 104, 18, WHITE, "la")
        c.text(f"Trayectoria {last.trajectory * last.tempo * 100:.0f}", 170, c.H - 78, 18, WHITE, "la")
        c.text(f"Técnica {last.form * 100:.0f}", 170, c.H - 52, 18, WHITE, "la")
        if last.curve and x2 - 360 > 100:
            draw_rep_chart(c, 345, c.H - 124, x2 - 6, c.H - 26, last.curve, last.reference_curve,
                           color=score_color(last.total), labels=False)
            c.text(f"{last.duration:.1f} s", x2 - 12, c.H - 116, 14, GREY, "ra")

    draw_feedback(c, feedback, now)
    button(c, "advance", "Terminar (ENTER)", c.W - 370, c.H - 66, c.W - 186, c.H - 24, size=17)
    if is_video:
        button(c, "pause", "Pausa (P)", c.W - 176, c.H - 66, c.W - 26, c.H - 24, size=17, color=ORANGE)


# --- Resultado del ejercicio y resumen ---
def draw_exercise_result(c: Canvas, session, index: int, total: int) -> None:
    c.dim(0.6)
    cx = c.W / 2
    hw = min(560, c.W / 2 - 20)
    c.panel(cx - hw, 24, cx + hw, c.H - 24, alpha=0.78)
    c.text(f"{session.exercise.name}", cx, 62, 40, ACCENT, "mm", bold=True)
    if session.skipped and not session.reps:
        c.text("Ejercicio saltado", cx, c.H / 2, 34, GREY, "mm")
    else:
        ref = "Referencia: tu repetición de prueba" if session.calibrated else "Referencia: estándar"
        c.text(ref, cx, 100, 17, GREY, "mm")
        x0, xv = cx - hw + 36, cx - 30
        c.text(f"{session.count} repeticiones", x0, 145, 30, WHITE, "lm", bold=True)
        c.text(f"{session.points} puntos", x0, 185, 28, GREEN, "lm", bold=True)
        rows = (("Puntuación media", session.average), ("Mejor repetición", session.best),
                ("Rango de movimiento", session.component_average("rom") * 100),
                ("Trayectoria", session.component_average("trajectory") * 100),
                ("Reglas de técnica", session.component_average("form") * 100))
        y = 236
        for label, value in rows:
            c.text(label, x0, y, 21, GREY, "lm")
            c.text(f"{value:.0f}", xv, y, 23, score_color(value), "rm", bold=True)
            y += 36
        if session.partial_reps:
            c.text(f"Repeticiones incompletas: {session.partial_reps}", x0, y + 2, 18, ORANGE, "lm")
            y += 30
        tips = session.top_feedback()
        if tips:
            c.text("Consejos:", x0, y + 14, 19, ACCENT, "lm", bold=True)
            for i, tip in enumerate(tips):
                c.text(f"• {tip}", x0, y + 44 + 28 * i, 17, WHITE, "lm")

        # Gráfica: mejor repetición (y las demás en gris) frente a la referencia
        best = session.best_rep
        if best is not None and best.curve:
            others = [r.curve for r in session.reps if r is not best]
            draw_rep_chart(c, cx + 10, 128, cx + hw - 28, 420, best.curve, best.reference_curve,
                           "Tu mejor repetición frente a la referencia", others, score_color(best.total))
            ly = 444
            c.line(cx + 30, ly, cx + 70, ly, score_color(best.total), 4)
            c.text(f"mejor ({best.total:.0f})", cx + 80, ly, 15, WHITE, "lm")
            c.polyline([(cx + 190 + 8 * i, ly) for i in range(6)], WHITE, 2, dashed=True)
            c.text("referencia", cx + 240, ly, 15, WHITE, "lm")
            if others:
                c.line(cx + 340, ly, cx + 380, ly, (110, 110, 110), 2)
                c.text("resto", cx + 390, ly, 15, WHITE, "lm")
            c.text("Cuanto más se parece tu curva a la discontinua, más puntos.",
                   (cx + 10 + cx + hw - 28) / 2, 474, 14, GREY, "mm")
    nxt = "Siguiente ejercicio" if index + 1 < total else "Ver resumen"
    button(c, "advance", f"{nxt} (ENTER)", cx - 290, c.H - 84, cx + 30, c.H - 40, color=GREEN,
           primary=True, size=20)
    button(c, "menu", "Menú (R)", cx + 50, c.H - 84, cx + 290, c.H - 40, color=GREY)


def draw_summary(c: Canvas, sessions: list, saved_path: Path | None, previous=None) -> None:
    """previous: HistoryEntry de la partida anterior (para comparar), o None."""
    c.dim(0.65)
    cx = c.W / 2
    total = sum(s.points for s in sessions)
    c.text("RESUMEN", cx, 70, 52, ACCENT, "mm", bold=True)
    c.text(f"{total} puntos", cx, 136, 46, GREEN, "mm", bold=True, stroke=1)
    if previous is not None:
        diff = total - previous.total_points
        sign = "+" if diff >= 0 else ""
        c.text(f"Partida anterior: {previous.total_points} puntos ({sign}{diff})", cx, 182, 20,
               GREEN if diff >= 0 else ORANGE, "mm", bold=True)
    hw = min(420, c.W / 2 - 20)
    c.panel(cx - hw, 210, cx + hw, 250 + 60 * len(sessions), alpha=0.7)
    heads = (("Ejercicio", -hw + 30, "lm"), ("Reps", 60, "mm"), ("Media", 200, "mm"), ("Puntos", hw - 30, "rm"))
    for label, dx, anchor in heads:
        c.text(label, cx + dx, 232, 18, GREY, anchor, bold=True)
    for i, s in enumerate(sessions):
        y = 276 + 60 * i
        c.text(s.exercise.name, cx - hw + 30, y, 26, WHITE, "lm", bold=True)
        if s.skipped and not s.reps:
            c.text("saltado", cx + 200, y, 22, GREY, "mm")
            continue
        c.text(str(s.count), cx + 60, y, 26, WHITE, "mm", bold=True)
        c.text(f"{s.average:.0f}", cx + 200, y, 26, score_color(s.average), "mm", bold=True)
        c.text(str(s.points), cx + hw - 30, y, 26, GREEN, "rm", bold=True)
    if saved_path is not None:
        c.text(f"Resultados guardados en {saved_path}", cx, c.H - 112, 15, GREY, "mm")
    button(c, "advance", "Jugar otra vez (ENTER)", cx - 330, c.H - 88, cx - 40, c.H - 44,
           color=GREEN, primary=True)
    button(c, "history", "Historial (H)", cx - 25, c.H - 88, cx + 165, c.H - 44)
    button(c, "quit", "Salir (ESC)", cx + 180, c.H - 88, cx + 330, c.H - 44, color=GREY)
