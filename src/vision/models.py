"""
Descarga y localiza los modelos .task del Pose Landmarker. No importa OpenCV
ni MediaPipe, para que scripts/download_models.py sea ligero.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import tqdm

from config import MODEL_URLS, MODEL_VARIANTS, MODELS_DIR, model_path


_MIN_MODEL_BYTES = 1_000_000  # Un .task válido pesa varios MB


def _is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_MODEL_BYTES


def download_model(variant: str, force: bool = False, timeout: float = 30.0) -> Path:
    """Descarga el modelo indicado a MODELS_DIR y devuelve su ruta absoluta."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Modelo desconocido '{variant}'. Opciones: {', '.join(MODEL_VARIANTS)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = model_path(variant)
    if _is_valid(dest) and not force:
        print(f"[modelos] {dest.name} ya existe en {dest.parent}. Omitiendo descarga.")
        return dest

    url = MODEL_URLS[variant]
    tmp = dest.with_name(dest.name + ".part")
    print(f"[modelos] Descargando {dest.name} desde {url}")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm.tqdm(total=total, unit="iB", unit_scale=True) as bar:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
        if not _is_valid(tmp):
            raise RuntimeError("El archivo descargado es demasiado pequeño; descarga corrupta.")
        os.replace(tmp, dest)  # Así nunca queda un .task a medias
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[modelos] Guardado en {dest}")
    return dest


def ensure_model(variant: str, download: bool = True) -> Path:
    """Devuelve la ruta absoluta del modelo, descargándolo si falta."""
    path = model_path(variant)
    if _is_valid(path):
        return path
    if not download:
        raise FileNotFoundError(
            f"No se encuentra el modelo {path}. Ejecuta: python scripts/download_models.py --model {variant}"
        )
    return download_model(variant)
