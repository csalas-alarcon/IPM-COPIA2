@echo off
REM Despliegue en un equipo limpio (Windows).
REM Uso: scripts\dependencies.bat   (desde cualquier carpeta)
REM Requiere Python 3.11 o 3.12 instalado (https://www.python.org/downloads/).
setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "PY_CMD="

REM 1) Buscar un Python compatible con el lanzador "py" y, si no, con "python"
for %%V in (3.12 3.11) do (
    if not defined PY_CMD (
        py -%%V -c "import sys" >nul 2>&1 && set "PY_CMD=py -%%V"
    )
)
if not defined PY_CMD (
    for /f "delims=" %%v in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do (
        if "%%v"=="3.12" set "PY_CMD=python"
        if "%%v"=="3.11" set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    echo [setup] ERROR: se necesita Python 3.11 o 3.12 ^(mediapipe 0.10.21 no soporta otras versiones^).
    exit /b 1
)
echo [setup] Usando %PY_CMD%

REM 2) Crear el entorno virtual
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [setup] Creando entorno virtual en %VENV_DIR%
    %PY_CMD% -m venv "%VENV_DIR%" || exit /b 1
)

REM 3) Instalar dependencias
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%PROJECT_ROOT%\requirements.txt" || exit /b 1

REM 4) Descargar modelos
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_ROOT%\scripts\download_models.py" || exit /b 1

echo [setup] Listo. Lanza el juego con: scripts\run.bat
endlocal
