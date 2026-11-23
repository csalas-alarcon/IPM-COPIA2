#!/usr/bin/env bash
# Lanza el juego con el Python del .venv del proyecto.
# Todos los argumentos se pasan a src/main.py, p. ej.:
#   bash scripts/run.sh --source 1 --model lite
#   bash scripts/run.sh --source http://192.168.1.20:8080/video

PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[run] No existe $PROJECT_ROOT/.venv. Ejecuta antes: source scripts/dependencies.sh"
    exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/src/main.py" "$@"
