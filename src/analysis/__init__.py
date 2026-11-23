"""
analysis/  ·  Conteo y puntuación de repeticiones.

Archivos:
    counter.py  Máquina de estados que cuenta reps + ExerciseTracker, que une
                medición, suavizado, postura y contador.
    scoring.py  Puntuación 0-100 de cada rep y comparación con la trayectoria
                de referencia.
    calibration.py  Repetición de prueba: referencias (reposo, máximo y
                trayectoria) del propio usuario.

Importa siempre el módulo concreto, p. ej.
``from analysis.counter import RepCounter``.
"""
