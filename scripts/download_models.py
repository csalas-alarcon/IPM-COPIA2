#!/usr/bin/env python3
"""
scripts/download_models.py  ·  Descarga de modelos.

Descarga los modelos Pose Landmarker en <raíz del proyecto>/models.

Uso:
    python scripts/download_models.py              # descarga lite, full y heavy
    python scripts/download_models.py --model full # sólo uno
    python scripts/download_models.py --force      # vuelve a descargar
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_VARIANTS  # noqa: E402
from vision.models import download_model  # noqa: E402


# =============================================================================
# Punto de entrada
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_VARIANTS, nargs="+", default=list(MODEL_VARIANTS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    failed = []
    for variant in args.model:
        try:
            download_model(variant, force=args.force)
        except Exception as e:  # noqa: BLE001
            print(f"[modelos] Error descargando '{variant}': {e}", file=sys.stderr)
            failed.append(variant)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
