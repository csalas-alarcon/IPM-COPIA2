"""
ui/screens.py  ·  Pantallas del juego.

Una función draw_* por pantalla (menú, instrucciones, cuenta atrás, HUD,
resultado del ejercicio y resumen) más los elementos comunes (esqueleto,
barra de progreso, avisos, indicador del vídeo). Todas reciben un Canvas y dibujan en unidades
de diseño.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.counter import TrackerState
from exercises.base import Exercise
from ui.canvas import (ACCENT, BLUE, GREEN, GREY, ORANGE, RED, WHITE, Canvas,
                       lerp_color, score_color)
from vision.pose import POSE_CONNECTIONS, PoseFrame, side_landmark


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass
class Feedback:
    lines: list[str]
    color: tuple[int, int, int]
    until: float


# =============================================================================
# Elementos comunes
# =============================================================================
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


# =============================================================================
# Pantallas
# =============================================================================
def draw_menu(c: Canvas, exercises: list[Exercise], duration: float, target: int,
              video_name: str | None = None) -> None:
    c.dim(0.6)
    cx = c.W / 2
    c.text("POSE TRAINER", cx, 150, 72, ACCENT, "mm", bold=True, stroke=2)
    if video_name is None:
        c.text("Cuenta tus repeticiones y puntúa tu técnica con la cámara", cx, 215, 26, WHITE, "mm")
    else:
        c.text(f"Vídeo: {video_name}", cx, 215, 26, WHITE, "mm")
    c.panel(cx - 330, 260, cx + 330, 290 + 52 * len(exercises), alpha=0.6)
    for i, ex in enumerate(exercises):
        c.text(f"{i + 1}.  {ex.name}", cx - 290, 300 + 52 * i, 32, WHITE, "lm", bold=True)
    c.text(f"Hasta {target} repeticiones o {int(duration)} s por ejercicio", cx,
           330 + 52 * len(exercises), 22, GREY, "mm")
    start = "Pulsa ENTER o haz clic para empezar"
    c.text(start, cx, c.H - 120, 34, GREEN, "mm", bold=True, stroke=1)
    hint = "ESC para salir" if video_name is None else "El vídeo empezará al pulsar ENTER   ·   P: pausa   ·   ESC: salir"
    c.text(hint, cx, c.H - 72, 22, GREY, "mm")


def _mmss(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def draw_video_status(c: Canvas, position: float, duration: float, paused: bool, ended: bool,
                      show_hint: bool = True) -> None:
    """Indicador del vídeo (arriba a la derecha) y, debajo, aviso de pausa o de fin del vídeo.

    show_hint: mostrar el aviso de fin del vídeo (sólo tiene sentido durante un ejercicio).
    """
    label = "PAUSA" if paused and not ended else ("FIN" if ended else "VÍDEO")
    if ended and duration > 0:
        position = duration
    times = _mmss(position) + (f" / {_mmss(duration)}" if duration > 0 else "")
    text = f"{label}  {times}   ·   P: pausa"
    w = c.text_width(text, 16, True)
    c.panel(c.W - w - 40, 14, c.W - 14, 50, alpha=0.6)
    c.text(text, c.W - 27, 32, 16, ORANGE if paused else WHITE, "rm", bold=True)
    if duration > 0:
        frac = float(np.clip(position / duration, 0, 1))
        c.rect(c.W - w - 40, 47, c.W - w - 40 + (w + 26) * frac, 50, ACCENT)

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


def draw_instructions(c: Canvas, ex: Exercise, index: int, total: int,
                      state: TrackerState, ready_fraction: float) -> None:
    c.dim(0.45)
    c.panel(40, 40, 800, c.H - 40, alpha=0.7)
    c.text(f"EJERCICIO {index + 1} / {total}", 80, 80, 22, ACCENT, "la", bold=True)
    c.text(ex.name, 80, 112, 50, WHITE, "la", bold=True)
    y = 210
    for i, line in enumerate(ex.instructions):
        c.text(f"{i + 1}. {line}", 80, y, 22, WHITE, "la")
        y += 44
    c.text("Cámara:", 80, y + 14, 20, ACCENT, "la", bold=True)
    c.text(ex.camera_tip, 80, y + 44, 20, GREY, "la")

    # Estado de la detección + anillo de "preparado"
    cx, cy = 420, c.H - 200
    if not state.detected or not state.measurable:
        msg, col = state.issue or "Buscando…", RED
    elif not state.posture_ok:
        msg, col = state.issue or ex.rest_hint, ORANGE
    elif ready_fraction > 0:
        msg, col = "¡Perfecto! Mantén la posición…", GREEN
    else:
        msg, col = ex.rest_hint, ACCENT
    c.arc(cx, cy, 60, 1.0, (70, 70, 70), 10)
    c.arc(cx, cy, 60, ready_fraction, GREEN, 10)
    c.text(f"{int(ready_fraction * 100)}%", cx, cy, 26, WHITE, "mm", bold=True)
    c.text(msg, cx, cy + 92, 21, col, "mm", bold=True)
    c.text("ENTER: empezar ya  ·  N: saltar  ·  ESC: salir", cx, c.H - 62, 17, GREY, "mm")


def draw_countdown(c: Canvas, ex: Exercise, remaining: float) -> None:
    c.dim(0.35)
    n = max(int(np.ceil(remaining)), 1)
    c.text(ex.name.upper(), c.W / 2, c.H / 2 - 150, 40, WHITE, "mm", bold=True, stroke=2)
    c.text(str(n), c.W / 2, c.H / 2, 200, ACCENT, "mm", bold=True, stroke=4)
    c.text("¡Prepárate!", c.W / 2, c.H / 2 + 150, 34, WHITE, "mm", stroke=2)


def draw_hud(c: Canvas, session, state: TrackerState, time_left: float, target: int,
             feedback: Feedback | None, now: float, rest_th: float, peak_th: float) -> None:
    ex = session.exercise
    # Marcador superior izquierdo
    c.panel(20, 20, 400, 200, alpha=0.65)
    c.text(ex.name, 40, 36, 28, ACCENT, "la", bold=True)
    count = str(session.count)
    c.text(count, 40, 76, 80, WHITE, "la", bold=True)
    c.text(f"/ {target} reps", 40 + c.text_width(count, 80, True) + 12, 120, 26, GREY, "la")
    c.text(f"{int(np.ceil(time_left))} s", 380, 36, 28, RED if time_left < 10 else WHITE, "ra", bold=True)
    c.text(f"{session.points} pts", 380, 160, 26, GREEN, "ra", bold=True)
    if session.partial_reps:
        c.text(f"incompletas: {session.partial_reps}", 40, 172, 16, ORANGE, "la")

    draw_progress_bar(c, state.progress, rest_th, peak_th)

    # Aviso en vivo
    if state.issue:
        draw_banner(c, state.issue, ORANGE if state.measurable else RED)

    # Última repetición
    if session.reps:
        last = session.reps[-1]
        c.panel(20, c.H - 130, 560, c.H - 20, alpha=0.65)
        c.text("Última rep", 40, c.H - 112, 18, GREY, "la")
        c.text(f"{last.total:.0f}", 40, c.H - 90, 56, score_color(last.total), "la", bold=True)
        c.text(f"Rango {last.rom * 100:.0f}", 190, c.H - 100, 20, WHITE, "la")
        c.text(f"Trayectoria {last.trajectory * last.tempo * 100:.0f}", 190, c.H - 72, 20, WHITE, "la")
        c.text(f"Técnica {last.form * 100:.0f}", 190, c.H - 44, 20, WHITE, "la")
        c.text(f"{last.duration:.1f} s", 540, c.H - 112, 18, GREY, "ra")

    # Mensaje tras la repetición
    if feedback is not None and now < feedback.until:
        y = c.H / 2 - 20 * len(feedback.lines)
        for i, line in enumerate(feedback.lines):
            size = 54 if i == 0 else 26
            c.text(line, c.W / 2, y, size, feedback.color if i == 0 else WHITE, "mm", bold=True, stroke=3)
            y += 60 if i == 0 else 36


def draw_exercise_result(c: Canvas, session, index: int, total: int) -> None:
    c.dim(0.6)
    cx = c.W / 2
    c.panel(cx - 380, 30, cx + 380, c.H - 30, alpha=0.75)
    c.text(f"{session.exercise.name}", cx, 75, 44, ACCENT, "mm", bold=True)
    if session.skipped and not session.reps:
        c.text("Ejercicio saltado", cx, c.H / 2, 34, GREY, "mm")
    else:
        c.text(f"{session.count} repeticiones", cx, 140, 34, WHITE, "mm", bold=True)
        c.text(f"{session.points} puntos", cx, 188, 30, GREEN, "mm", bold=True)
        rows = (("Puntuación media", session.average), ("Mejor repetición", session.best),
                ("Rango de movimiento", session.component_average("rom") * 100),
                ("Trayectoria", session.component_average("trajectory") * 100),
                ("Reglas de técnica", session.component_average("form") * 100))
        y = 245
        for label, value in rows:
            c.text(label, cx - 300, y, 24, GREY, "lm")
            c.text(f"{value:.0f}", cx + 300, y, 26, score_color(value), "rm", bold=True)
            y += 40
        if session.partial_reps:
            c.text(f"Repeticiones incompletas: {session.partial_reps}", cx, y + 4, 20, ORANGE, "mm")
            y += 32
        tips = session.top_feedback()
        if tips:
            c.text("Consejos:", cx - 300, y + 16, 20, ACCENT, "lm", bold=True)
            for i, tip in enumerate(tips):
                c.text(f"• {tip}", cx - 300, y + 48 + 30 * i, 20, WHITE, "lm")
    nxt = "siguiente ejercicio" if index + 1 < total else "ver resumen"
    c.text(f"ENTER: {nxt}", cx, c.H - 62, 24, GREEN, "mm", bold=True)


def draw_summary(c: Canvas, sessions: list, saved_path: Path | None) -> None:
    c.dim(0.65)
    cx = c.W / 2
    total = sum(s.points for s in sessions)
    c.text("RESUMEN", cx, 90, 52, ACCENT, "mm", bold=True)
    c.text(f"{total} puntos", cx, 160, 46, GREEN, "mm", bold=True, stroke=1)
    c.panel(cx - 420, 210, cx + 420, 250 + 64 * len(sessions), alpha=0.7)
    heads = (("Ejercicio", -390, "lm"), ("Reps", 60, "mm"), ("Técnica", 200, "mm"), ("Puntos", 390, "rm"))
    for label, dx, anchor in heads:
        c.text(label, cx + dx, 232, 18, GREY, anchor, bold=True)
    for i, s in enumerate(sessions):
        y = 280 + 64 * i
        c.text(s.exercise.name, cx - 390, y, 28, WHITE, "lm", bold=True)
        if s.skipped and not s.reps:
            c.text("saltado", cx + 200, y, 22, GREY, "mm")
            continue
        c.text(str(s.count), cx + 60, y, 28, WHITE, "mm", bold=True)
        c.text(f"{s.average:.0f}", cx + 200, y, 28, score_color(s.average), "mm", bold=True)
        c.text(str(s.points), cx + 390, y, 28, GREEN, "rm", bold=True)
    if saved_path is not None:
        c.text(f"Resultados guardados en {saved_path}", cx, c.H - 150, 16, GREY, "mm")
    c.text("ENTER: jugar otra vez   ·   ESC: salir", cx, c.H - 100, 26, WHITE, "mm", bold=True)
