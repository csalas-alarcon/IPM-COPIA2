"""
analysis/scoring.py  ·  Puntuación de una repetición (0-100).

Componentes (todos 0..1):
  * rom        : qué tan cerca se llegó a la pose de máximo esfuerzo.
  * trajectory : parecido de la transición reposo->máximo->reposo con la de
                 referencia (alineada por fases), multiplicado por un factor de tempo.
  * form       : reglas de técnica propias de cada ejercicio.

Trayectoria de referencia:
La repetición se divide en dos fases separadas por el pico (máximo esfuerzo):
  * ida    : reposo -> máximo
  * vuelta : máximo -> reposo
Cada fase se remuestrea y se compara con un perfil de referencia: el coseno
ideal (arranque y frenado suaves, articulaciones coordinadas) o, si se ha
calibrado, el de la repetición de prueba del propio usuario. Alinear por fases
permite que la ida y la vuelta tengan duraciones distintas (p. ej. bajar
lento y subir rápido) pero sí penaliza parones, tirones y rebotes, cosa que
un DTW libre "absorbería" deformando el tiempo.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from config import ScoringConfig
from exercises.base import Exercise


# =============================================================================
# Tipos de datos
# =============================================================================
@dataclass
class RepScore:
    total: float
    rom: float
    trajectory: float
    form: float
    tempo: float
    duration: float
    feedback: list[str] = field(default_factory=list)
    # Curvas para dibujar (progreso 0..1 a lo largo de la rep): la real y la de referencia
    curve: list[float] = field(default_factory=list)
    reference_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        def r(v):
            if isinstance(v, float):
                return round(v, 3)
            if isinstance(v, list) and v and isinstance(v[0], float):
                return [round(x, 3) for x in v]
            return v
        return {k: r(v) for k, v in asdict(self).items()}


# =============================================================================
# Trayectoria de referencia
# =============================================================================
def resample(times: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Remuestrea (N, F) a (n, F) uniformemente en el tiempo."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = np.linspace(times[0], times[-1], n)
    return np.stack([np.interp(grid, times, values[:, k]) for k in range(values.shape[1])], axis=1)


def _ease_from(level: np.ndarray, n: int) -> np.ndarray:
    """Tramo del perfil coseno ideal s(τ) = (1 - cos πτ)/2 que va desde `level`
    (por feature) hasta 1. La repetición se graba a partir del umbral de reposo,
    así que no empieza parada: se compara con el MISMO tramo de la curva ideal."""
    level = np.clip(np.asarray(level, dtype=float), 0.0, 0.95)
    tau0 = np.arccos(1.0 - 2.0 * level) / np.pi                 # s(tau0) = level
    tau = tau0[None, :] + (1.0 - tau0[None, :]) * np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - np.cos(np.pi * tau)) / 2.0


def _profile_from(profile: np.ndarray, level: np.ndarray, n: int) -> np.ndarray:
    """Igual que _ease_from, pero con un perfil medido (m, F) que sube de 0 a 1."""
    level = np.clip(np.asarray(level, dtype=float), 0.0, 0.95)
    m, n_feat = profile.shape
    grid = np.linspace(0.0, 1.0, m)
    out = np.empty((n, n_feat))
    for k in range(n_feat):
        col = profile[:, k]
        tau0 = float(np.interp(level[k], col, grid))            # col(tau0) = level
        out[:, k] = np.interp(np.linspace(tau0, 1.0, n), grid, col)
    return out


def build_profile(times: np.ndarray, u: np.ndarray, weights: np.ndarray,
                  n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Perfil de referencia (ida, vuelta) a partir de una repetición medida.

    u: (N, F) normalizados con las referencias calibradas. Cada fase se
    remuestrea, se fuerza monótona (sin rebotes del detector) y se escala a
    0 -> 1. La vuelta se guarda invertida en el tiempo (también sube de 0 a 1).
    Si una fase de un feature apenas se mueve, se usa el coseno ideal.
    """
    times = np.asarray(times, dtype=float)
    u = np.asarray(u, dtype=float)
    k = int(np.clip(np.argmax(u @ weights), 1, len(u) - 2))
    ideal = _ease_from(np.zeros(u.shape[1]), n)
    up = resample(times[:k + 1], u[:k + 1], n)             # reposo -> pico
    down = resample(times[k:], u[k:], n)[::-1]              # (pico -> reposo) invertido
    phases = []
    for p in (up, down):
        p = np.maximum.accumulate(np.clip(p, 0.0, 1.2), axis=0)
        lo, rng = p[0], p[-1] - p[0]
        ok = rng > 0.3
        phases.append(np.where(ok, (p - lo) / np.where(ok, rng, 1.0), ideal))
    return phases[0], phases[1]


def reference_trajectory(start: np.ndarray, end: np.ndarray, n_per_phase: int,
                         profile: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Trayectoria de referencia start -> 1 (máximo) -> end, en espacio normalizado.

    profile: (ida, vuelta) medidos en la calibración; None = coseno ideal.
    """
    if profile is None:
        up = _ease_from(start, n_per_phase)
        down = _ease_from(end, n_per_phase)[::-1]
    else:
        up = _profile_from(profile[0], start, n_per_phase)
        down = _profile_from(profile[1], end, n_per_phase)[::-1]
    return np.vstack([up, down])


def trajectory_error(times: np.ndarray, u: np.ndarray, weights: np.ndarray,
                     n_samples: int = 40,
                     profile: tuple[np.ndarray, np.ndarray] | None = None,
                     ) -> tuple[float, np.ndarray, np.ndarray]:
    """Error medio (distancia ponderada) entre la rep y la referencia.

    u: (N, F) features normalizados (0 = reposo, 1 = máximo).
    profile: trayectoria de referencia calibrada (None = coseno ideal).
    Devuelve (error, trayectoria_real_remuestreada, referencia).
    """
    times = np.asarray(times, dtype=float)
    u = np.asarray(u, dtype=float)
    level = u @ weights
    k = int(np.clip(np.argmax(level), 1, len(u) - 2))   # índice del pico
    half = max(n_samples // 2, 2)
    actual = np.vstack([resample(times[:k + 1], u[:k + 1], half),
                        resample(times[k:], u[k:], half)])
    ref = reference_trajectory(u[0], u[-1], half, profile)
    err = float(np.mean(np.sqrt(np.sum(weights * (actual - ref) ** 2, axis=1))))
    return err, actual, ref


# =============================================================================
# Puntuación
# =============================================================================
def _tempo_factor(duration: float, cfg: ScoringConfig) -> float:
    if duration < cfg.min_rep_seconds:
        return max(0.4, duration / cfg.min_rep_seconds)
    if duration > cfg.max_rep_seconds:
        return max(0.5, cfg.max_rep_seconds / duration)
    return 1.0


def score_repetition(
    exercise: Exercise,
    times: np.ndarray,
    features: np.ndarray,
    penalties: list[dict[str, float]],
    cfg: ScoringConfig,
) -> RepScore:
    times = np.asarray(times, dtype=float)
    u = exercise.normalize(np.asarray(features, dtype=float))   # (N, F)
    w = exercise.weights
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    # --- Rango de movimiento -------------------------------------------------
    # Pasarse de la referencia (más profundo) no penaliza: se recorta a 1.
    dist_to_peak = np.sqrt(np.sum(w * (np.clip(u, 0.0, 1.0) - 1.0) ** 2, axis=1))
    rom = float(np.clip(1.0 - dist_to_peak.min(), 0.0, 1.0))

    # --- Trayectoria ---------------------------------------------------------
    # Se normaliza por el pico alcanzado para medir sólo la FORMA de la
    # transición (la profundidad ya la mide 'rom').
    tempo = _tempo_factor(duration, cfg)
    curve: list[float] = []
    reference_curve: list[float] = []
    if len(times) >= 3 and duration > 0:
        scale = max(float((u @ w).max()), 0.3)
        err, actual, ref = trajectory_error(times, np.clip(u / scale, -0.3, 1.3), w,
                                            cfg.trajectory_samples, exercise.reference_profile)
        err = max(err - cfg.trajectory_noise_floor, 0.0)
        trajectory = float(np.exp(-cfg.trajectory_sensitivity * err))
        # Para la gráfica: progreso real (con su profundidad) frente a la referencia
        curve = [float(x) for x in np.clip(actual @ w * scale, 0.0, 1.2)]
        reference_curve = [float(x) for x in np.clip(ref @ w, 0.0, 1.2)]
    else:
        trajectory = 0.0

    # --- Técnica -------------------------------------------------------------
    per_rule: dict[str, list[float]] = defaultdict(list)
    for frame_pens in penalties:
        for rule, value in frame_pens.items():
            per_rule[rule].append(value)
    rule_means = {rule: float(np.mean(v)) for rule, v in per_rule.items() if v}
    form = float(1.0 - np.mean(list(rule_means.values()))) if rule_means else 1.0

    # --- Total ---------------------------------------------------------------
    ws = np.array([cfg.weight_rom, cfg.weight_trajectory, cfg.weight_form], dtype=float)
    ws = ws / ws.sum()
    total = 100.0 * float(ws @ np.array([rom, trajectory * tempo, form]))

    # --- Consejos ------------------------------------------------------------
    feedback: list[str] = []
    if rom < cfg.rom_feedback_threshold:
        feedback.append(exercise.rom_feedback)
    for rule, mean in sorted(rule_means.items(), key=lambda kv: -kv[1]):
        if mean >= cfg.feedback_threshold and rule in exercise.form_messages:
            feedback.append(exercise.form_messages[rule])
    if duration < cfg.min_rep_seconds:
        feedback.append("Más lento y controlado")
    elif duration > cfg.max_rep_seconds:
        feedback.append("Intenta un ritmo más constante")
    elif trajectory < 0.6:
        feedback.append("Movimiento más fluido, sin parones ni rebotes")

    return RepScore(total, rom, trajectory, form, tempo, duration, feedback, curve, reference_curve)
