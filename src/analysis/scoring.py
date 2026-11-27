"""
Puntuación de una repetición (0-100), con tres componentes entre 0 y 1:
  * rom: lo cerca que se llegó a la pose de máximo esfuerzo;
  * trajectory: parecido de la transición reposo -> máximo -> reposo con la de
    referencia, multiplicado por un factor de tempo;
  * form: reglas de técnica propias de cada ejercicio.

La repetición se divide en el pico en dos fases, ida (reposo -> máximo) y
vuelta (máximo -> reposo), y cada una se compara con su perfil de referencia:
el coseno ideal o, si hay calibración, el de la repetición de prueba. Así la
ida y la vuelta pueden durar distinto, pero se penalizan parones, tirones y
rebotes, que un DTW libre absorbería deformando el tiempo.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from config import ScoringConfig
from exercises.base import Exercise


@dataclass
class RepScore:
    total: float
    rom: float
    trajectory: float
    form: float
    tempo: float
    duration: float
    feedback: list[str] = field(default_factory=list)
    # Progreso 0..1 a lo largo de la repetición, para dibujar: el real y el de referencia
    curve: list[float] = field(default_factory=list)
    reference_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Diccionario para guardar en JSON, con los decimales redondeados a 3."""
        datos = asdict(self)
        for clave, valor in datos.items():
            if isinstance(valor, float):
                datos[clave] = round(valor, 3)
            elif isinstance(valor, list) and valor and isinstance(valor[0], float):
                datos[clave] = [round(x, 3) for x in valor]
        return datos


# --- Trayectoria de referencia ---
def resample(times: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Remuestrea (N, F) a (n, F) con puntos uniformes en el tiempo."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = np.linspace(times[0], times[-1], n)
    return np.stack([np.interp(grid, times, values[:, k]) for k in range(values.shape[1])], axis=1)


def _ease_from(level: np.ndarray, n: int) -> np.ndarray:
    """Tramo del perfil coseno ideal s(τ) = (1 - cos πτ) / 2 desde `level` hasta 1.

    La repetición se graba desde el umbral de reposo, así que no empieza parada:
    se compara con el mismo tramo de la curva ideal.
    """
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

    u: (N, F) ángulos normalizados con las referencias calibradas. Cada fase se
    remuestrea, se fuerza monótona (sin rebotes del detector) y se escala de 0 a 1.
    La vuelta se guarda invertida en el tiempo, así que también sube de 0 a 1.
    Si un ángulo apenas se mueve en una fase, usa el coseno ideal.
    """
    times = np.asarray(times, dtype=float)
    u = np.asarray(u, dtype=float)
    peak = int(np.clip(np.argmax(u @ weights), 1, len(u) - 2))
    ideal = _ease_from(np.zeros(u.shape[1]), n)
    up = resample(times[:peak + 1], u[:peak + 1], n)
    down = resample(times[peak:], u[peak:], n)[::-1]

    phases = []
    for phase in (up, down):
        phase = np.maximum.accumulate(np.clip(phase, 0.0, 1.2), axis=0)
        normalized = ideal.copy()
        for k in range(phase.shape[1]):
            travel = phase[-1, k] - phase[0, k]
            if travel > 0.3:
                normalized[:, k] = (phase[:, k] - phase[0, k]) / travel
        phases.append(normalized)
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
    """Error medio (distancia ponderada) entre la repetición y su referencia.

    u: (N, F) ángulos normalizados (0 = reposo, 1 = máximo).
    Devuelve (error, trayectoria real remuestreada, referencia).
    """
    times = np.asarray(times, dtype=float)
    u = np.asarray(u, dtype=float)
    peak = int(np.clip(np.argmax(u @ weights), 1, len(u) - 2))
    half = max(n_samples // 2, 2)
    actual = np.vstack([resample(times[:peak + 1], u[:peak + 1], half),
                        resample(times[peak:], u[peak:], half)])
    ref = reference_trajectory(u[0], u[-1], half, profile)
    err = float(np.mean(np.sqrt(np.sum(weights * (actual - ref) ** 2, axis=1))))
    return err, actual, ref


# --- Puntuación ---
def _tempo_factor(duration: float, cfg: ScoringConfig) -> float:
    """1 si la duración es razonable; menos si fue muy rápida o muy lenta."""
    if duration < cfg.min_rep_seconds:
        return max(0.4, duration / cfg.min_rep_seconds)
    if duration > cfg.max_rep_seconds:
        return max(0.5, cfg.max_rep_seconds / duration)
    return 1.0


def _form_score(penalties: list[dict[str, float]]) -> tuple[float, dict[str, float]]:
    """Técnica (1 - media de las penalizaciones) y la penalización media de cada regla."""
    per_rule: dict[str, list[float]] = defaultdict(list)
    for frame_penalties in penalties:
        for rule, value in frame_penalties.items():
            per_rule[rule].append(value)
    rule_means = {rule: float(np.mean(values)) for rule, values in per_rule.items()}
    form = 1.0 - float(np.mean(list(rule_means.values()))) if rule_means else 1.0
    return form, rule_means


def score_repetition(
    exercise: Exercise,
    times: np.ndarray,
    features: np.ndarray,
    penalties: list[dict[str, float]],
    cfg: ScoringConfig,
) -> RepScore:
    """Puntúa una repetición a partir de sus ángulos y penalizaciones frame a frame."""
    times = np.asarray(times, dtype=float)
    u = exercise.normalize(np.asarray(features, dtype=float))
    w = exercise.weights
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    # Rango: pasarse de la referencia (ir más profundo) no penaliza, por eso se recorta a 1
    dist_to_peak = np.sqrt(np.sum(w * (np.clip(u, 0.0, 1.0) - 1.0) ** 2, axis=1))
    rom = float(np.clip(1.0 - dist_to_peak.min(), 0.0, 1.0))

    # Trayectoria: se normaliza por el pico alcanzado para medir solo la forma
    # del movimiento (la profundidad ya la mide rom)
    tempo = _tempo_factor(duration, cfg)
    curve: list[float] = []
    reference_curve: list[float] = []
    if len(times) >= 3 and duration > 0:
        scale = max(float((u @ w).max()), 0.3)
        err, actual, ref = trajectory_error(times, np.clip(u / scale, -0.3, 1.3), w,
                                            cfg.trajectory_samples, exercise.reference_profile)
        err = max(err - cfg.trajectory_noise_floor, 0.0)
        trajectory = float(np.exp(-cfg.trajectory_sensitivity * err))
        curve = [float(x) for x in np.clip(actual @ w * scale, 0.0, 1.2)]
        reference_curve = [float(x) for x in np.clip(ref @ w, 0.0, 1.2)]
    else:
        trajectory = 0.0

    form, rule_means = _form_score(penalties)

    weights = np.array([cfg.weight_rom, cfg.weight_trajectory, cfg.weight_form])
    weights = weights / weights.sum()
    total = 100.0 * float(weights @ np.array([rom, trajectory * tempo, form]))

    feedback: list[str] = []
    if rom < cfg.rom_feedback_threshold:
        feedback.append(exercise.rom_feedback)
    for rule, mean in sorted(rule_means.items(), key=lambda item: -item[1]):
        if mean >= cfg.feedback_threshold and rule in exercise.form_messages:
            feedback.append(exercise.form_messages[rule])
    if duration < cfg.min_rep_seconds:
        feedback.append("Más lento y controlado")
    elif duration > cfg.max_rep_seconds:
        feedback.append("Intenta un ritmo más constante")
    elif trajectory < 0.6:
        feedback.append("Movimiento más fluido, sin parones ni rebotes")

    return RepScore(total, rom, trajectory, form, tempo, duration, feedback, curve, reference_curve)
