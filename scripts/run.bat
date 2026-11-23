@echo off
REM Lanza el juego con el Python del .venv del proyecto (Windows).
REM Los argumentos se pasan a src\main.py, p. ej.: scripts\run.bat --source 1 --model lite
REM Con un vídeo grabado: scripts\run.bat --video demo.mp4 --exercises pushup
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [run] No existe %PROJECT_ROOT%\.venv. Ejecuta antes: scripts\dependencies.bat
    exit /b 1
)
"%PY%" "%PROJECT_ROOT%\src\main.py" %*
