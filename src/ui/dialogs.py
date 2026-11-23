"""
ui/dialogs.py  ·  Elegir un vídeo.

  * list_videos: vídeos de las carpetas conocidas (videos/ del proyecto...),
    para el selector dentro de la ventana del juego.
  * ask_video_file: explorador de archivos del sistema (tkinter). Se abre en
    un proceso aparte para que tkinter y la ventana de OpenCV no interfieran
    (en macOS no pueden convivir dos toolkits gráficos en el mismo proceso).
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from config import VIDEO_EXTENSIONS


# =============================================================================
# Constantes
# =============================================================================
_DIALOG_SCRIPT = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except tk.TclError:
    pass
exts = " ".join("*" + e for e in sys.argv[2:])
path = filedialog.askopenfilename(title="Elige un vídeo", initialdir=sys.argv[1],
                                  filetypes=[("Vídeos", exts), ("Todos los archivos", "*.*")])
root.destroy()
sys.stdout.write(path or "")
"""


# =============================================================================
# Funciones
# =============================================================================
def list_videos(folders, limit: int = 8) -> list[Path]:
    """Vídeos de las carpetas dadas (sin repetir), ordenados por nombre."""
    seen, found = set(), []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                key = path.resolve()
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    return found[:limit]


@lru_cache(maxsize=1)
def tk_available() -> bool:
    try:
        r = subprocess.run([sys.executable, "-c", "import tkinter"], capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ask_video_file(initial_dir: Path) -> str | None:
    """Abre el explorador de archivos y devuelve la ruta elegida (o None)."""
    if not tk_available():
        return None
    try:
        r = subprocess.run([sys.executable, "-c", _DIALOG_SCRIPT, str(initial_dir), *VIDEO_EXTENSIONS],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    path = r.stdout.strip()
    return path or None
