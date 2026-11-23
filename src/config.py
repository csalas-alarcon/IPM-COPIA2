"""
config.py  ·  Configuración central del proyecto.

Rutas absolutas, URLs de los modelos, fuentes y todos los parámetros
ajustables (cámara, modelo, juego y puntuación).

Todas las rutas se calculan como rutas ABSOLUTAS a partir de la ubicación de
este archivo, de modo que el programa funciona igual se lance desde donde se
lance (raíz del repo, carpeta src/, IDE, script...).
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# Rutas absolutas
# =============================================================================
SRC_DIR: Path = Path(__file__).resolve().parent                  # .../src
PROJECT_ROOT: Path = SRC_DIR.parent                              # .../IPM_Prac_2
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# Se pueden sobreescribir con variables de entorno (siempre se resuelven a absolutas)
MODELS_DIR: Path = Path(
    os.environ.get("POSE_GAME_MODELS_DIR", PROJECT_ROOT / "models")
).expanduser().resolve()
RESULTS_DIR: Path = Path(
    os.environ.get("POSE_GAME_RESULTS_DIR", PROJECT_ROOT / "results")
).expanduser().resolve()
# Carpeta donde el juego busca vídeos para el selector del menú
VIDEOS_DIR: Path = Path(
    os.environ.get("POSE_GAME_VIDEOS_DIR", PROJECT_ROOT / "videos")
).expanduser().resolve()
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")


# =============================================================================
# Modelos de MediaPipe Pose Landmarker
# =============================================================================
_MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MODEL_VARIANTS: tuple[str, ...] = ("lite", "full", "heavy")
MODEL_URLS: dict[str, str] = {
    v: f"{_MODEL_BASE_URL}/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
    for v in MODEL_VARIANTS
}


def model_filename(variant: str) -> str:
    return f"pose_landmarker_{variant}.task"


def model_path(variant: str) -> Path:
    return MODELS_DIR / model_filename(variant)


# =============================================================================
# Fuentes (para poder pintar tildes y eñes; cv2.putText no las soporta)
# =============================================================================
_WINDIR = Path(os.environ.get("WINDIR", "C:/Windows"))
FONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    _WINDIR / "Fonts" / "arial.ttf",
)
BOLD_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
    _WINDIR / "Fonts" / "arialbd.ttf",
)


# =============================================================================
# Parámetros
# =============================================================================
@dataclass
class CameraConfig:
    # Índice de la webcam (0, 1, ...) o URL de un stream (p. ej. la cámara del
    # móvil con IP Webcam / DroidCam: "http://192.168.1.20:8080/video").
    source: int | str = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True              # Efecto espejo con la cámara (más natural para el usuario)
    mirror_video: bool = False       # Efecto espejo con un vídeo grabado (se ve tal cual)
    max_failed_reads: int = 60       # Lecturas fallidas seguidas antes de abortar
    start_seconds: float = 0.0       # Con un vídeo: segundo desde el que empieza


@dataclass
class ModelConfig:
    variant: str = "full"            # lite | full | heavy
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Usar las coordenadas 3D "world" (en metros) para calcular ángulos.
    # Son más robustas a la posición de la cámara que las 2D de la imagen.
    use_world_landmarks: bool = True

    @property
    def path(self) -> Path:
        return model_path(self.variant)


@dataclass
class GameConfig:
    exercises: tuple[str, ...] = ("pullup", "pushup", "bulgarian_squat")
    exercise_duration: float = 60.0  # Segundos por ejercicio
    target_reps: int = 10            # Termina antes si se llega a estas reps
    countdown_seconds: int = 3
    ready_hold_seconds: float = 1.5  # Tiempo en posición de reposo para empezar
    ready_progress: float = 0.2      # Progreso máximo para considerar "en reposo"
    min_visibility: float = 0.5      # Visibilidad mínima media de las articulaciones
    smoothing_alpha: float = 0.45    # EMA sobre los ángulos (1 = sin suavizado)
    pose_lost_reset: float = 0.6     # s sin ver a la persona -> reinicia suavizado
    feedback_seconds: float = 2.5    # Tiempo que se muestra el mensaje tras una rep
    save_results: bool = True
    window_name: str = "Pose Trainer"
    sound: bool = True               # Pitidos de cuenta atrás, repeticiones...
    record_path: str | None = None   # Grabar lo que muestra la ventana en este .mp4
    record_fps: float = 30.0
    # --- Calibración: repetición de prueba antes de cada ejercicio -----------
    calibrate: bool = True
    calibration_timeout: float = 20.0      # s para hacer la repetición de prueba
    calibration_min_progress: float = 0.5  # Profundidad mínima (respecto a la referencia estándar)
    calibration_min_span: float = 0.4      # Recorrido mínimo de cada ángulo (fracción del estándar)


@dataclass
class ScoringConfig:
    # Umbrales sobre el progreso (0 = reposo, 1 = máximo esfuerzo)
    rest_threshold: float = 0.25
    peak_threshold: float = 0.75
    partial_threshold: float = 0.45
    # Pesos de la puntuación de cada repetición (se normalizan al usarse)
    weight_rom: float = 0.35
    weight_trajectory: float = 0.35
    weight_form: float = 0.30
    # Comparación de la trayectoria con la de referencia
    trajectory_samples: int = 40
    trajectory_sensitivity: float = 6.0   # Más alto = más exigente
    trajectory_noise_floor: float = 0.015  # Error que se tolera por ruido del detector
    # Tempo
    min_rep_seconds: float = 0.8
    max_rep_seconds: float = 8.0
    abandon_rep_seconds: float = 15.0
    # Consejos
    feedback_threshold: float = 0.3      # Penalización media para dar consejo de técnica
    rom_feedback_threshold: float = 0.75  # Rango por debajo del cual se avisa
    live_feedback_threshold: float = 0.6  # Penalización instantánea para avisar en vivo


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    game: GameConfig = field(default_factory=GameConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


settings = Settings()  # Instancia única
