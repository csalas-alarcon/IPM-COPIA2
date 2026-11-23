"""
ui/sound.py  ·  Efectos de sonido.

Pitidos generados con numpy (no hace falta ningún archivo de audio):
cuenta atrás, inicio, repetición, repetición perfecta, incompleta, calibrado
y fin del ejercicio.

Reproducción sin bloquear el juego, con el primer sistema disponible:
  1. sounddevice (viene con mediapipe; en Windows y macOS funciona tal cual);
  2. winsound (Windows);
  3. paplay / aplay (Linux, si sounddevice no encuentra PortAudio).
Si no hay ninguno, el juego sigue sin sonido.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np


# =============================================================================
# Constantes
# =============================================================================
SAMPLE_RATE = 44100

# nombre -> lista de (frecuencia Hz, duración s); 0 Hz = silencio
SOUNDS: dict[str, list[tuple[float, float]]] = {
    "tick": [(880, 0.09)],
    "go": [(1320, 0.28)],
    "rep": [(660, 0.07), (990, 0.12)],
    "perfect": [(660, 0.07), (880, 0.07), (1320, 0.16)],
    "partial": [(330, 0.18)],
    "calibrated": [(990, 0.08), (0, 0.03), (1320, 0.14)],
    "done": [(784, 0.12), (988, 0.12), (1175, 0.12), (1568, 0.3)],
    "click": [(1500, 0.025)],
}


# =============================================================================
# Funciones
# =============================================================================
def synthesize(notes: list[tuple[float, float]], volume: float = 0.35) -> np.ndarray:
    """Notas -> señal float32 mono (con fundidos para que no chasquee)."""
    parts = []
    for freq, dur in notes:
        n = int(SAMPLE_RATE * dur)
        t = np.arange(n) / SAMPLE_RATE
        tone = np.sin(2 * np.pi * freq * t) if freq > 0 else np.zeros(n)
        fade = min(int(0.008 * SAMPLE_RATE), n // 2)
        env = np.ones(n)
        if fade:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        parts.append(tone * env)
    return (np.concatenate(parts) * volume).astype(np.float32)


def to_wav_bytes(signal: np.ndarray) -> bytes:
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# =============================================================================
# Clases
# =============================================================================
class Sound:
    def __init__(self, enabled: bool = True):
        self.backend: str | None = None
        self._signals = {name: synthesize(notes) for name, notes in SOUNDS.items()}
        if enabled:
            self._init_backend()

    def _init_backend(self) -> None:
        try:
            import sounddevice as sd  # noqa: F401  (falla si no hay PortAudio)
            sd.query_devices(kind="output")
            self._sd, self.backend = sd, "sounddevice"
            return
        except Exception:
            pass
        try:
            import winsound
            self._winsound, self.backend = winsound, "winsound"
            self._wavs = {k: to_wav_bytes(v) for k, v in self._signals.items()}
            return
        except ImportError:
            pass
        player = shutil.which("paplay") or shutil.which("aplay")
        if player:
            self._player, self.backend = player, "player"
            self._dir = Path(tempfile.mkdtemp(prefix="pose_trainer_sounds_"))
            for k, v in self._signals.items():
                (self._dir / f"{k}.wav").write_bytes(to_wav_bytes(v))
            return
        print("[sonido] No hay salida de audio disponible: el juego seguirá sin sonido.")

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def play(self, name: str) -> None:
        if self.backend is None or name not in self._signals:
            return
        try:
            if self.backend == "sounddevice":
                self._sd.play(self._signals[name], SAMPLE_RATE)
            elif self.backend == "winsound":
                data = self._wavs[name]
                threading.Thread(target=self._winsound.PlaySound,
                                 args=(data, self._winsound.SND_MEMORY), daemon=True).start()
            else:
                subprocess.Popen([self._player, str(self._dir / f"{name}.wav")],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:      # Un fallo de audio nunca debe parar el juego
            print(f"[sonido] Desactivado: {e}")
            self.backend = None
