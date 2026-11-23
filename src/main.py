#!/usr/bin/env python3
"""
main.py  ·  Punto de entrada.

Orquesta configuración, modelo, cámara, detector y juego.

Uso:
    python src/main.py                         # webcam 0, modelo full
    python src/main.py --source 1              # otra webcam
    python src/main.py --source http://IP:8080/video   # cámara del móvil
    python src/main.py --model heavy --exercises pushup bulgarian_squat
    python src/main.py --video demo.mp4        # vídeo grabado en vez de la cámara
    python src/main.py --video demo.mp4 --exercises pushup --start 4
    python src/main.py --record                # graba la ventana en results/grabacion_*.mp4

Sin cámara (y sin --source / --video), el juego arranca igual y deja elegir un
vídeo desde el menú.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Garantiza que los paquetes de src/ se importan igual se lance desde donde se lance
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_VARIANTS, RESULTS_DIR, settings  # noqa: E402
from exercises.catalog import EXERCISES, build_exercises  # noqa: E402
from game.game import Game  # noqa: E402
from ui.canvas import TextRenderer  # noqa: E402
from ui.sound import Sound  # noqa: E402
from vision.camera import Camera, NoSource, is_video_file  # noqa: E402
from vision.detector import PoseDetector  # noqa: E402
from vision.models import ensure_model  # noqa: E402


# =============================================================================
# Argumentos de línea de comandos
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose Trainer: cuenta y puntúa dominadas, flexiones y sentadillas búlgaras.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--source", default=None,
                     help="Índice de cámara (0, 1...), URL de stream (p. ej. la cámara del móvil) "
                          "o ruta de un archivo de vídeo.")
    src.add_argument("--video", default=None, metavar="ARCHIVO",
                     help="Usa un vídeo grabado (.mp4, .avi, .mov...) en vez de la cámara. "
                          "Se juega igual que en vivo: el vídeo empieza al pulsar ENTER.")
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
    p.add_argument("--start", type=float, default=0.0, metavar="SEGUNDOS",
                   help="Con un vídeo: empezar en este segundo (p. ej. para saltarse el principio).")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Sin repetición de prueba: usar las referencias estándar.")
    p.add_argument("--record", nargs="?", const="", default=None, metavar="ARCHIVO",
                   help="Grabar la ventana del juego en un .mp4 (por defecto en results/). "
                        "También se puede empezar y parar con la tecla G.")
    p.add_argument("--no-sound", action="store_true", help="Sin efectos de sonido.")
    mirror = p.add_mutually_exclusive_group()
    mirror.add_argument("--no-mirror", action="store_true",
                        help="No mostrar la imagen en espejo (por defecto, espejo sólo con cámara).")
    mirror.add_argument("--mirror", action="store_true",
                        help="Mostrar la imagen en espejo también con un vídeo.")
    p.add_argument("--no-download", action="store_true", help="No descargar el modelo si falta.")
    p.add_argument("--no-save", action="store_true", help="No guardar los resultados en results/.")
    return p.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    if args.video is not None:
        if not is_video_file(args.video):
            raise FileNotFoundError(f"No existe el archivo de vídeo '{args.video}'.")
        settings.camera.source = args.video
    elif args.source is not None:
        settings.camera.source = args.source
    settings.camera.width = args.width
    settings.camera.height = args.height
    # Espejo: natural con la cámara (como un espejo); un vídeo grabado se ve tal cual
    settings.camera.mirror = not args.no_mirror
    settings.camera.mirror_video = args.mirror
    settings.camera.start_seconds = max(args.start, 0.0)
    settings.game.calibrate = not args.no_calibrate
    settings.game.sound = not args.no_sound
    if args.record is not None:
        settings.game.record_path = args.record or str(
            RESULTS_DIR / f"grabacion_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    settings.model.variant = args.model
    settings.game.exercises = tuple(args.exercises)
    settings.game.exercise_duration = args.duration
    settings.game.target_reps = args.reps
    settings.game.save_results = not args.no_save


def open_initial_source(args: argparse.Namespace):
    """Abre la fuente pedida. Si no se pidió ninguna y no hay cámara, arranca sin
    fuente para poder elegir un vídeo desde el menú."""
    try:
        return Camera(settings.camera)
    except RuntimeError as e:
        if args.source is not None or args.video is not None:
            raise
        print(f"[aviso] {e}\n[aviso] Arrancando sin cámara: elige un vídeo desde el menú.")
        return NoSource(settings.camera.width, settings.camera.height)


# =============================================================================
# Punto de entrada
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        apply_args(args)
        model_path = ensure_model(settings.model.variant, download=not args.no_download)
        exercises = build_exercises(settings.game.exercises)
        renderer = TextRenderer()
        camera = open_initial_source(args)
        sound = Sound(enabled=settings.game.sound)
        with camera, PoseDetector(model_path, settings.model) as detector:
            Game(settings, camera, detector, exercises, renderer, sound).run()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
