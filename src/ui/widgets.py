"""
ui/widgets.py  ·  Elementos interactivos y gráficas.

  * button / checkbox: se dibujan en el Canvas y se registran en
    Canvas.buttons, de modo que el juego sabe qué acción lanza cada clic.
  * draw_rep_chart: gráfica de una repetición (progreso reposo -> máximo ->
    reposo) frente a la de referencia.
  * draw_line_chart: gráfica de evolución (historial).
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from typing import Sequence

import numpy as np

from ui.canvas import ACCENT, DARK, GREEN, GREY, WHITE, Canvas, lerp_color


# =============================================================================
# Botones
# =============================================================================
def button(c: Canvas, action: str, label: str, x1: float, y1: float, x2: float, y2: float,
           color=ACCENT, primary: bool = False, enabled: bool = True, size: float = 20) -> None:
    """Botón con clic. `action` es lo que recibe Game._do() al pulsarlo."""
    hover = enabled and c.hovered(x1, y1, x2, y2)
    if not enabled:
        fill, border, text_col, alpha = DARK, (90, 90, 90), (120, 120, 120), 0.6
    elif primary:
        fill = lerp_color(color, WHITE, 0.25) if hover else color
        border, text_col, alpha = fill, DARK, 0.92
    else:
        fill = lerp_color(DARK, color, 0.35) if hover else DARK
        border, text_col, alpha = color, WHITE, 0.8
    c.panel(x1, y1, x2, y2, color=fill, alpha=alpha, border=border)
    c.text(label, (x1 + x2) / 2, (y1 + y2) / 2, size, text_col, "mm", bold=True)
    if enabled:
        c.buttons.append((action, (c.px(x1), c.px(y1), c.px(x2), c.px(y2))))


def checkbox(c: Canvas, action: str, label: str, checked: bool, x1: float, y1: float,
             x2: float, y2: float, size: float = 24, detail: str = "") -> None:
    """Casilla de verificación con texto; toda la fila es clicable."""
    hover = c.hovered(x1, y1, x2, y2)
    c.panel(x1, y1, x2, y2, color=lerp_color(DARK, ACCENT, 0.25) if hover else DARK, alpha=0.7,
            border=ACCENT if checked else (90, 90, 90))
    box = (y2 - y1) * 0.5
    bx, by = x1 + 16, (y1 + y2) / 2 - box / 2
    c.rect(bx, by, bx + box, by + box, WHITE, 2)
    if checked:
        c.rect(bx + 5, by + 5, bx + box - 5, by + box - 5, GREEN)
    c.text(label, bx + box + 16, (y1 + y2) / 2, size, WHITE if checked else GREY, "lm", bold=True)
    if detail:
        c.text(detail, x2 - 16, (y1 + y2) / 2, size * 0.7, GREY, "rm")
    c.buttons.append((action, (c.px(x1), c.px(y1), c.px(x2), c.px(y2))))


# =============================================================================
# Gráficas
# =============================================================================
def draw_rep_chart(c: Canvas, x1: float, y1: float, x2: float, y2: float,
                   curve: Sequence[float], reference: Sequence[float],
                   title: str = "", others: Sequence[Sequence[float]] = (),
                   color=GREEN, labels: bool = True) -> None:
    """Progreso de una repetición (curva) frente a la referencia (discontinua).

    El eje vertical es el progreso: abajo = reposo, arriba = máximo esfuerzo.
    El horizontal, el tiempo de la repetición (ida y vuelta).
    """
    c.panel(x1, y1, x2, y2, alpha=0.6)
    top = y1 + (34 if title else 12)
    pad_l = 58 if labels else 10
    gx1, gx2, gy1, gy2 = x1 + pad_l, x2 - 12, top, y2 - (28 if labels else 10)
    if title:
        c.text(title, x1 + 14, y1 + 18, 16, ACCENT, "lm", bold=True)

    def pts(values):
        v = np.asarray(values, dtype=float)
        xs = np.linspace(gx1, gx2, len(v))
        ys = gy2 - (gy2 - gy1) * np.clip(v, 0.0, 1.15) / 1.15
        return list(zip(xs, ys))

    y_peak = gy2 - (gy2 - gy1) / 1.15
    c.line(gx1, gy2, gx2, gy2, (90, 90, 90), 1)
    c.line(gx1, y_peak, gx2, y_peak, (90, 90, 90), 1)
    if labels:
        c.text("máx.", gx1 - 8, y_peak, 13, GREY, "rm")
        c.text("reposo", gx1 - 8, gy2, 13, GREY, "rm")
        c.text("ida", (gx1 + gx2) / 2 - 40, gy2 + 14, 13, GREY, "mm")
        c.text("vuelta", (gx1 + gx2) / 2 + 40, gy2 + 14, 13, GREY, "mm")
    for o in others:
        if len(o) >= 2:
            c.polyline(pts(o), (110, 110, 110), 1.5)
    if len(reference) >= 2:
        c.polyline(pts(reference), WHITE, 2, dashed=True)
    if len(curve) >= 2:
        c.polyline(pts(curve), color, 3.5)


def draw_line_chart(c: Canvas, x1: float, y1: float, x2: float, y2: float,
                    series: Sequence[tuple[str, Sequence[float], tuple]], y_max: float,
                    title: str = "", x_labels: Sequence[str] = ()) -> None:
    """Gráfica de líneas sencilla (una o varias series con la misma escala)."""
    c.panel(x1, y1, x2, y2, alpha=0.6)
    if title:
        c.text(title, x1 + 14, y1 + 18, 16, ACCENT, "lm", bold=True)
    gx1, gx2, gy1, gy2 = x1 + 50, x2 - 20, y1 + 44, y2 - 30
    y_max = max(float(y_max), 1.0)
    for frac in (0.0, 0.5, 1.0):
        y = gy2 - (gy2 - gy1) * frac
        c.line(gx1, y, gx2, y, (80, 80, 80), 1)
        c.text(f"{y_max * frac:.0f}", gx1 - 8, y, 13, GREY, "rm")
    n = max((len(v) for _, v, _ in series), default=0)
    xs = np.linspace(gx1, gx2, n) if n > 1 else np.array([(gx1 + gx2) / 2])
    for i, lab in enumerate(x_labels[:n]):
        if n <= 8 or i % max(n // 6, 1) == 0:
            c.text(lab, xs[i], gy2 + 14, 12, GREY, "mm")
    lx = gx2
    for name, values, color in reversed(series):
        v = np.asarray(values, dtype=float)
        if len(v) == 0:
            continue
        points = list(zip(xs[:len(v)], gy2 - (gy2 - gy1) * np.clip(v / y_max, 0, 1)))
        c.polyline(points, color, 2.5)
        for p in points:
            c.circle_px((c.px(p[0]), c.px(p[1])), 4, color)
        c.text(name, lx, y1 + 18, 14, color, "rm", bold=True)
        lx -= c.text_width(name, 14, True) + 20
