"""
ui/canvas.py  ·  Lienzo de dibujo y texto.

Contiene:
  * Paleta de colores (BGR) y utilidades de color.
  * TextRenderer: pinta texto con Pillow (cv2.putText no soporta tildes ni eñes).
  * Canvas: lienzo independiente de la resolución.

Todas las coordenadas del Canvas se expresan en "unidades de diseño" con
altura 720; el lienzo las escala al tamaño real del frame (webcam 480p,
móvil 1080p...).
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import BOLD_FONT_CANDIDATES, FONT_CANDIDATES


# =============================================================================
# Constantes
# =============================================================================
# Paleta (BGR)
WHITE = (245, 245, 245)
GREY = (170, 170, 170)
DARK = (18, 18, 22)
ACCENT = (0, 190, 255)     # ámbar
GREEN = (110, 210, 90)
ORANGE = (40, 150, 255)
RED = (70, 70, 235)
BLUE = (230, 160, 60)

DESIGN_HEIGHT = 720.0


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass
class TextItem:
    text: str
    xy: tuple[int, int]
    size: int
    color: tuple[int, int, int]      # BGR
    anchor: str = "la"               # Anclas de Pillow: la, mm, ra, ms, ...
    bold: bool = False
    stroke: int = 0


# =============================================================================
# Funciones
# =============================================================================
def lerp_color(c1, c2, t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def score_color(score: float) -> tuple[int, int, int]:
    if score >= 80:
        return GREEN
    if score >= 60:
        return ACCENT
    return RED


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


# =============================================================================
# TextRenderer
# =============================================================================
class TextRenderer:
    def __init__(self) -> None:
        self.regular = _first_existing(FONT_CANDIDATES)
        self.bold = _first_existing(BOLD_FONT_CANDIDATES) or self.regular
        if self.regular is None:
            print("[ui] No se encontró ninguna fuente TTF; se usará la de Pillow por defecto.")

    @lru_cache(maxsize=64)
    def font(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        path = self.bold if bold else self.regular
        size = max(int(size), 6)
        if path is not None:
            return ImageFont.truetype(str(path), size)
        return ImageFont.load_default(size=size)

    def draw(self, frame: np.ndarray, items: list[TextItem]) -> np.ndarray:
        """Pinta todos los textos de una vez (una sola conversión BGR<->RGB por frame)."""
        if not items:
            return frame
        img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        draw = ImageDraw.Draw(img)
        for it in items:
            b, g, r = it.color
            draw.text(it.xy, it.text, font=self.font(it.size, it.bold), fill=(r, g, b),
                      anchor=it.anchor, stroke_width=it.stroke, stroke_fill=(0, 0, 0))
        frame[:] = np.asarray(img)[:, :, ::-1]
        return frame


# =============================================================================
# Canvas
# =============================================================================
class Canvas:
    def __init__(self, frame: np.ndarray, renderer: TextRenderer):
        self.frame = frame
        self.renderer = renderer
        self.h, self.w = frame.shape[:2]
        self.s = self.h / DESIGN_HEIGHT
        self.W = self.w / self.s          # Ancho en unidades de diseño
        self.H = DESIGN_HEIGHT
        self._texts: list[TextItem] = []
        # Botones dibujados en este frame: (acción, rectángulo en píxeles). Ver ui/widgets.py
        self.buttons: list[tuple[str, tuple[int, int, int, int]]] = []
        self.mouse: tuple[int, int] | None = None      # Posición del ratón (píxeles)

    # --- Conversión de unidades ----------------------------------------------
    def px(self, v: float) -> int:
        return int(round(v * self.s))

    def pt(self, x: float, y: float) -> tuple[int, int]:
        return self.px(x), self.px(y)

    # --- Dibujo en unidades de diseño ----------------------------------------
    def dim(self, alpha: float = 0.55) -> None:
        self.frame[:] = (self.frame * (1.0 - alpha)).astype(np.uint8)

    def panel(self, x1, y1, x2, y2, color=DARK, alpha: float = 0.65, border=None) -> None:
        X1, Y1 = max(self.px(x1), 0), max(self.px(y1), 0)
        X2, Y2 = min(self.px(x2), self.w), min(self.px(y2), self.h)
        if X2 <= X1 or Y2 <= Y1:
            return
        roi = self.frame[Y1:Y2, X1:X2]
        overlay = np.empty_like(roi)
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
        if border is not None:
            cv2.rectangle(self.frame, (X1, Y1), (X2 - 1, Y2 - 1), border, max(self.px(2), 1))

    def rect(self, x1, y1, x2, y2, color, thickness: float = -1) -> None:
        t = -1 if thickness < 0 else max(self.px(thickness), 1)
        cv2.rectangle(self.frame, self.pt(x1, y1), self.pt(x2, y2), color, t)

    def line(self, x1, y1, x2, y2, color, thickness: float = 2) -> None:
        cv2.line(self.frame, self.pt(x1, y1), self.pt(x2, y2), color, max(self.px(thickness), 1), cv2.LINE_AA)

    def arc(self, cx, cy, r, fraction: float, color, thickness: float = 8) -> None:
        cv2.ellipse(self.frame, self.pt(cx, cy), (self.px(r), self.px(r)), -90, 0,
                    360 * float(np.clip(fraction, 0, 1)), color, max(self.px(thickness), 1), cv2.LINE_AA)

    def text(self, text: str, x: float, y: float, size: float = 26, color=WHITE,
             anchor: str = "la", bold: bool = False, stroke: float = 0) -> None:
        self._texts.append(TextItem(text, self.pt(x, y), self.px(size), color, anchor, bold,
                                    self.px(stroke) if stroke else 0))

    def text_width(self, text: str, size: float = 26, bold: bool = False) -> float:
        """Ancho del texto en unidades de diseño."""
        return self.renderer.font(self.px(size), bold).getlength(text) / self.s

    def polyline(self, points, color, thickness: float = 2, dashed: bool = False) -> None:
        """Línea poligonal en unidades de diseño (discontinua si dashed)."""
        pts = np.array([self.pt(x, y) for x, y in points], dtype=np.int32)
        t = max(self.px(thickness), 1)
        if len(pts) < 2:
            return
        if not dashed:
            cv2.polylines(self.frame, [pts], False, color, t, cv2.LINE_AA)
            return
        for i in range(0, len(pts) - 1, 2):
            cv2.line(self.frame, tuple(map(int, pts[i])), tuple(map(int, pts[i + 1])), color, t, cv2.LINE_AA)

    def hovered(self, x1, y1, x2, y2) -> bool:
        if self.mouse is None:
            return False
        mx, my = self.mouse
        return self.px(x1) <= mx <= self.px(x2) and self.px(y1) <= my <= self.px(y2)

    # --- Dibujo directo en píxeles (el esqueleto ya viene en píxeles) --------
    def line_px(self, p1, p2, color, thickness: int = 2) -> None:
        cv2.line(self.frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color,
                 max(int(thickness * self.s), 1), cv2.LINE_AA)

    def circle_px(self, p, radius: float, color, thickness: int = -1) -> None:
        cv2.circle(self.frame, (int(p[0]), int(p[1])), max(int(radius * self.s), 1), color,
                   thickness, cv2.LINE_AA)

    # --- Final del frame -----------------------------------------------------
    def finish(self) -> np.ndarray:
        self.renderer.draw(self.frame, self._texts)
        self._texts.clear()
        return self.frame
