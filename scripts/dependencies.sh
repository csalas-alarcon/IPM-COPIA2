#!/usr/bin/env bash
# Despliegue en un equipo limpio (Linux / macOS):
#   1. Busca un Python compatible (3.12 o 3.11; las versiones fijadas en
#      requirements.txt no tienen ruedas para otras versiones).
#   2. Crea el entorno virtual <raíz>/.venv
#   3. Instala requirements.txt
#   4. Descarga los modelos de MediaPipe en <raíz>/models
#
#   source scripts/dependencies.sh   -> además deja el .venv activado en tu terminal
#   bash   scripts/dependencies.sh   -> sólo instala
#
# Variables opcionales:
#   PYTHON=/ruta/a/python3.12  fuerza un intérprete concreto
#   RECREATE_VENV=1            borra y vuelve a crear .venv
#
# Funciona desde cualquier directorio: todas las rutas son absolutas.

_pose_find_python() {
    local candidates=("${PYTHON:-}" python3.12 python3.11 python3 python)
    local c ver
    for c in "${candidates[@]}"; do
        [ -z "$c" ] && continue
        command -v "$c" >/dev/null 2>&1 || continue
        ver="$("$c" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)" || continue
        case "$ver" in
            3.11|3.12) command -v "$c"; return 0 ;;
        esac
    done
    return 1
}

_pose_setup() {
    local script_path project_root venv_dir python_bin venv_ver
    script_path="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${BASH_SOURCE[0]}")"
    project_root="$(cd "$(dirname "$script_path")/.." && pwd)"
    venv_dir="$project_root/.venv"

    if ! python_bin="$(_pose_find_python)"; then
        echo "[setup] ERROR: se necesita Python 3.11 o 3.12 (mediapipe 0.10.21 no soporta otras versiones)."
        echo "[setup]   Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
        echo "[setup]   o indica uno con: PYTHON=/ruta/a/python3.12 bash scripts/dependencies.sh"
        return 1
    fi
    echo "[setup] Usando $python_bin ($("$python_bin" --version 2>&1))"

    # Si el venv existe pero con otra versión de Python, se recrea
    if [ -x "$venv_dir/bin/python" ]; then
        venv_ver="$("$venv_dir/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)"
        case "$venv_ver" in 3.11|3.12) ;; *) RECREATE_VENV=1 ;; esac
    fi
    if [ "${RECREATE_VENV:-0}" = "1" ] && [ -d "$venv_dir" ]; then
        echo "[setup] Recreando $venv_dir"
        rm -rf "$venv_dir"
    fi

    if [ ! -x "$venv_dir/bin/python" ]; then
        echo "[setup] Creando entorno virtual en $venv_dir"
        "$python_bin" -m venv "$venv_dir" || {
            echo "[setup] Error creando el venv. En Ubuntu/Debian: sudo apt install python3-venv (o python3.12-venv)"
            return 1
        }
    fi

    echo "[setup] Instalando dependencias"
    "$venv_dir/bin/python" -m pip install --upgrade pip || return 1
    "$venv_dir/bin/python" -m pip install -r "$project_root/requirements.txt" || return 1

    echo "[setup] Descargando modelos"
    "$venv_dir/bin/python" "$project_root/scripts/download_models.py" || return 1

    # Sólo tiene efecto en tu terminal si el script se ejecuta con `source`
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    echo "[setup] Listo. Lanza el juego con: bash scripts/run.sh"
}

_pose_setup
_pose_status=$?
unset -f _pose_setup _pose_find_python

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return $_pose_status   # ejecutado con source
else
    exit $_pose_status
fi
