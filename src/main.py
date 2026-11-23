#!/usr/bin/env python3
"""
main.py  ·  Punto de entrada.

Orquesta configuración, modelo, cámara, detector y juego.

Uso:
    python src/main.py                         # webcam 0, modelo full
    python src/main.py --source 1              # otra webcam
    python src/main.py --source http://IP:8080/video   # cámara del móvil
    python src/main.py --model heavy --exercises pushup bulgarian_squat
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garantiza que los paquetes de src/ se importan igual se lance desde donde se lance
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_VARIANTS, settings  # noqa: E402
from exercises.catalog import EXERCISES, build_exercises  # noqa: E402
from game.game import Game  # noqa: E402
from ui.canvas import TextRenderer  # noqa: E402
from vision.camera import Camera  # noqa: E402
from vision.detector import PoseDetector  # noqa: E402
from vision.models import ensure_model  # noqa: E402


# =============================================================================
# Argumentos de línea de comandos
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose Trainer: cuenta y puntúa dominadas, flexiones y sentadillas búlgaras.")
    p.add_argument("--source", default=None,
                   help="Índice de cámara (0, 1...) o URL de stream (p. ej. la cámara del móvil).")
    p.add_argument("--model", choices=MODEL_VARIANTS, default=settings.model.variant,
                   help="Variante del Pose Landmarker (lite = rápido, heavy = preciso).")
    p.add_argument("--exercises", nargs="+", choices=list(EXERCISES), default=list(settings.game.exercises),
                   help="Ejercicios y orden.")
    p.add_argument("--duration", type=float, default=settings.game.exercise_duration,
                   help="Segundos por ejercicio.")
    p.add_argument("--reps", type=int, default=settings.game.target_reps,
                   help="Repeticiones objetivo por ejercicio.")
    p.add_argument("--width", type=int, default=settings.camera.width)
    p.add_argument("--height", type=int, default=settings.camera.height)
    p.add_argument("--no-mirror", action="store_true", help="No mostrar la imagen en espejo.")
    p.add_argument("--no-download", action="store_true", help="No descargar el modelo si falta.")
    p.add_argument("--no-save", action="store_true", help="No guardar los resultados en results/.")
    return p.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    if args.source is not None:
        settings.camera.source = args.source
    settings.camera.width = args.width
    settings.camera.height = args.height
    settings.camera.mirror = not args.no_mirror
    settings.model.variant = args.model
    settings.game.exercises = tuple(args.exercises)
    settings.game.exercise_duration = args.duration
    settings.game.target_reps = args.reps
    settings.game.save_results = not args.no_save


# =============================================================================
# Punto de entrada
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_args(args)

    try:
        model_path = ensure_model(settings.model.variant, download=not args.no_download)
        exercises = build_exercises(settings.game.exercises)
        renderer = TextRenderer()
        with Camera(settings.camera) as camera, PoseDetector(model_path, settings.model) as detector:
            Game(settings, camera, detector, exercises, renderer).run()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
