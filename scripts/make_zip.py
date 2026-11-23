#!/usr/bin/env python3
"""
scripts/make_zip.py  ·  Generación del .zip de entrega.

Genera el .zip de entrega con el nombre de los integrantes.

No incluye .venv, modelos, resultados, cachés ni .git (el código debe poder
desplegarse en un equipo limpio con scripts/dependencies.sh|.bat).

Uso:
    python scripts/make_zip.py nombre_apellido1_apellido2 [otro_nombre_apellido1_apellido2]
    -> <raíz>/nombre_apellido1_apellido2_otro_nombre_apellido1_apellido2.zip
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import zipfile
from pathlib import Path


# =============================================================================
# Constantes
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {".venv", "venv", "mp_env", ".git", "models", "results", "__pycache__",
                 ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".task", ".part", ".pyc", ".pyo", ".zip"}


# =============================================================================
# Funciones
# =============================================================================
def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
            yield path, rel


# =============================================================================
# Punto de entrada
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("members", nargs="+", help="Nombre completo de cada integrante")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT)
    args = p.parse_args()

    name = "_".join(slug(m) for m in args.members if slug(m))
    if not name:
        print("Nombre de integrantes vacío", file=sys.stderr)
        return 1
    out = args.out_dir.resolve() / f"{name}.zip"
    root_in_zip = Path(name)

    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, rel in iter_files(PROJECT_ROOT):
            zf.write(path, root_in_zip / rel)
            count += 1
    print(f"[zip] {count} archivos -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
