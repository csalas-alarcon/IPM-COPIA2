"""
Pruebas de integración con poses sintéticas.

Escenarios completos sin cámara ni ventana: puntuación y calibración de los
tres ejercicios con repeticiones simuladas, y partidas del menú al resumen
(con cámara, con vídeo y con repetición de prueba). Las piezas sueltas se
prueban en test_analisis.py, test_juego.py y test_interfaz.py.

Ejecutar:  .venv/bin/python -m unittest discover -s tests -v
(Windows:  .venv\\Scripts\\python -m unittest discover -s tests -v)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for ruta in (SRC_DIR, TESTS_DIR):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from analysis.calibration import Calibrator  # noqa: E402
from analysis.counter import RepCounter  # noqa: E402
from config import Settings  # noqa: E402
from exercises.catalog import EXERCISES, build_exercises  # noqa: E402
from game.game import Phase  # noqa: E402
from utilidades import FRAME, PruebaDeJuego, VideoFalso, crear_juego, pose_flexion  # noqa: E402


# --- Utilidades: repeticiones sintéticas ---
def coseno(tau):
    """Perfil ideal de una repetición: 0 -> 1 -> 0 cuando tau va de 0 a 1."""
    return (1 - np.cos(2 * np.pi * tau)) / 2


def simular(ejercicio, cfg, perfil=coseno, reps=3, duracion=2.0, fps=30, ruido=1.0,
            profundidad=1.0, penalizaciones=None, semilla=0):
    """Repeticiones sintéticas en el espacio de ángulos; devuelve los eventos del contador.

    Hay 15 frames de reposo al principio y 5 tras cada repetición. `profundidad`
    es la fracción del recorrido estándar que se alcanza.
    """
    rng = np.random.default_rng(semilla)
    reposo = ejercicio.rest_vector
    maximo = ejercicio.peak_vector
    frames = [(reposo + rng.normal(0, ruido, reposo.shape), {}) for _ in range(15)]
    pasos = int(duracion * fps)
    for _ in range(reps):
        for i in range(pasos + 1):
            avance = perfil(i / pasos) * profundidad
            angulos = reposo + avance * (maximo - reposo) + rng.normal(0, ruido, reposo.shape)
            frames.append((angulos, penalizaciones or {}))
        for _ in range(5):
            frames.append((reposo + rng.normal(0, ruido, reposo.shape), {}))

    contador = RepCounter(ejercicio, cfg)
    eventos = []
    for i, (angulos, penalizacion) in enumerate(frames):
        evento = contador.update(i / fps, angulos, ejercicio.progress(angulos), penalizacion)
        if evento:
            eventos.append(evento)
    return eventos


def alimentar_calibracion(calibrador, ejercicio, reposo, maximo, duracion=2.0, fps=30, ruido=0.5, semilla=1):
    """Reposo y una repetición coseno reposo -> máximo -> reposo. Devuelve el último estado."""
    rng = np.random.default_rng(semilla)
    frames_reposo = [reposo + rng.normal(0, ruido, reposo.shape) for _ in range(20)]
    pasos = int(duracion * fps)
    frames_repeticion = [reposo + coseno(i / pasos) * (maximo - reposo) + rng.normal(0, ruido, reposo.shape)
                         for i in range(pasos + 1)]
    frames_final = [reposo + rng.normal(0, ruido, reposo.shape) for _ in range(5)]

    estado = None
    t = 0.0
    for angulos in frames_reposo:
        calibrador.add_rest(angulos)
        estado = calibrador.update(t, angulos, ejercicio.progress(angulos)) or estado
        t += 1 / fps
    for angulos in frames_repeticion + frames_final:
        if estado == "done":
            break
        estado = calibrador.update(t, angulos, ejercicio.progress(angulos)) or estado
        t += 1 / fps
    return estado


def poses_repeticion(inicio: float, pico: float, pasos: int = 60) -> list:
    """Poses de una flexión que va del ángulo de codo `inicio` a `pico` y vuelve."""
    return [pose_flexion(inicio + coseno(i / pasos) * (pico - inicio)) for i in range(pasos + 1)]


# --- Puntuación de los ejercicios reales ---
class TestPuntuacionDeLosEjercicios(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings().scoring

    def contar(self, ejercicio, **opciones):
        """Devuelve (repeticiones puntuadas, repeticiones incompletas) de una simulación."""
        eventos = simular(ejercicio, self.cfg, **opciones)
        return ([e for e in eventos if e.kind == "rep"],
                [e for e in eventos if e.kind == "partial"])

    def test_el_progreso_vale_0_05_y_1_en_reposo_punto_medio_y_maximo(self):
        for ejercicio in build_exercises(EXERCISES):
            with self.subTest(ejercicio=ejercicio.key):
                self.assertAlmostEqual(ejercicio.progress(ejercicio.rest_vector), 0.0, places=6)
                self.assertAlmostEqual(ejercicio.progress(ejercicio.peak_vector), 1.0, places=6)
                medio = (ejercicio.rest_vector + ejercicio.peak_vector) / 2
                self.assertAlmostEqual(ejercicio.progress(medio), 0.5, places=6)

    def test_repeticiones_ideales_puntuan_mas_de_90(self):
        for ejercicio in build_exercises(EXERCISES):
            with self.subTest(ejercicio=ejercicio.key):
                reps, incompletas = self.contar(ejercicio)
                self.assertEqual((len(reps), len(incompletas)), (3, 0))
                self.assertGreater(min(r.score.total for r in reps), 90)

    def test_recorrido_corto_cuenta_como_incompleta(self):
        for ejercicio in build_exercises(EXERCISES):
            with self.subTest(ejercicio=ejercicio.key):
                reps, incompletas = self.contar(ejercicio, profundidad=0.6)
                self.assertEqual((len(reps), len(incompletas)), (0, 3))

    def test_los_defectos_restan_puntos_y_dan_el_consejo_adecuado(self):
        flexion = build_exercises(["pushup"])[0]
        ideal = self.contar(flexion)[0][0].score
        rapida = self.contar(flexion, duracion=0.5)[0][0].score
        a_tirones = self.contar(flexion, perfil=lambda t: np.round(coseno(t) * 3) / 3)[0][0].score
        mala_tecnica = self.contar(flexion, penalizaciones={"cuerpo": 0.8})[0][0].score
        for nombre, puntuacion in (("rápida", rapida), ("a tirones", a_tirones), ("mala técnica", mala_tecnica)):
            with self.subTest(defecto=nombre):
                self.assertLess(puntuacion.total, ideal.total - 10)
        self.assertIn("Más lento y controlado", rapida.feedback)
        self.assertIn(flexion.form_messages["cuerpo"], mala_tecnica.feedback)


# --- Calibración con una repetición de prueba ---
class TestCalibracionDelUsuario(unittest.TestCase):
    def setUp(self):
        self.ajustes = Settings()

    def crear_calibrador(self, clave):
        ejercicio = build_exercises([clave])[0]
        return ejercicio, Calibrator(ejercicio, self.ajustes.game, self.ajustes.scoring)

    def test_aprende_la_referencia_personal_y_puntua_con_ella(self):
        flexion, calibrador = self.crear_calibrador("pushup")
        reposo, maximo = np.array([172.0]), np.array([100.0])    # Menos profundo que el estándar (80º)
        self.assertEqual(alimentar_calibracion(calibrador, flexion, reposo, maximo), "done")
        resultado = calibrador.result
        self.assertAlmostEqual(float(resultado.rest[0]), 172.0, delta=2.0)
        self.assertAlmostEqual(float(resultado.peak[0]), 100.0, delta=3.0)
        self.assertEqual(resultado.fallback, ())

        resultado.apply(flexion)
        self.assertTrue(flexion.calibrated)
        self.assertAlmostEqual(flexion.progress(maximo), 1.0, delta=0.05)

        # Con su referencia, las repeticiones que llegan a 100º son completas y puntúan alto
        eventos = simular(flexion, self.ajustes.scoring)
        reps = [e for e in eventos if e.kind == "rep"]
        self.assertEqual(len(reps), 3)
        self.assertGreater(min(r.score.total for r in reps), 90)
        self.assertTrue(reps[0].score.curve and reps[0].score.reference_curve)

        flexion.reset_reference()
        self.assertFalse(flexion.calibrated)
        self.assertEqual(float(flexion.peak_vector[0]), 80.0)

    def test_un_angulo_que_no_se_mueve_usa_la_referencia_estandar(self):
        dominada, calibrador = self.crear_calibrador("pullup")
        nombres = [f.name for f in dominada.features]
        reposo = dominada.default_rest.copy()
        maximo = dominada.default_peak.copy()
        maximo[1] = reposo[1] - 5.0                              # El hombro casi no se mueve
        self.assertEqual(alimentar_calibracion(calibrador, dominada, reposo, maximo), "done")
        self.assertEqual(calibrador.result.fallback, (nombres[1],))
        self.assertEqual(float(calibrador.result.peak[1]), float(dominada.default_peak[1]))


# --- Partidas completas ---
class TestPartidaCompleta(PruebaDeJuego):
    def test_sesion_de_flexiones_del_menu_al_resumen(self):
        sonido = mock.Mock()
        juego = crear_juego(sound=sonido)
        juego.settings.game.calibrate = False
        juego.settings.game.save_results = True
        reposo = pose_flexion(165)

        self.pasar_frames(juego, reposo)
        juego._handle_key(13)                                  # ENTER en el menú
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)
        self.pasar_frames(juego, reposo, 60)                   # Reposo mantenido 1,5 s
        self.assertIs(juego.phase, Phase.COUNTDOWN)
        self.pasar_frames(juego, reposo, 100)                  # Cuenta atrás de 3 s
        self.assertIs(juego.phase, Phase.ACTIVE)

        for pico in (80, 80, 125):                             # Dos repeticiones buenas y una incompleta
            self.pasar_poses(juego, poses_repeticion(165, pico))
            self.pasar_frames(juego, reposo, 10)
        self.assertEqual(juego.session.count, 2)
        self.assertEqual(juego.session.partial_reps, 1)

        juego._handle_key(13)                                  # Terminar el ejercicio
        self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
        juego._handle_key(13)
        self.assertIs(juego.phase, Phase.SUMMARY)

        guardado = json.loads(juego.saved_path.read_text(encoding="utf-8"))
        self.assertEqual(juego.saved_path.parent, self.resultados.resolve())
        self.assertEqual(guardado["exercises"][0]["reps"], 2)
        self.assertEqual(guardado["exercises"][0]["partial_reps"], 1)

        sonidos = [llamada.args[0] for llamada in sonido.play.call_args_list]
        principales = [s for s in sonidos if s in ("tick", "go", "partial", "done")]
        self.assertEqual(principales, ["tick", "tick", "tick", "go", "partial", "done"])


class TestPartidaConVideo(PruebaDeJuego):
    def test_el_video_manda_en_el_reloj_y_en_el_final_de_la_partida(self):
        video = VideoFalso()
        juego = crear_juego(camara=video, claves=("pushup", "bulgarian_squat"))
        juego.settings.game.calibrate = False
        reposo = pose_flexion(165)

        self.pasar_frames(juego, reposo)
        self.assertTrue(video.paused)                          # En el menú el vídeo está en pausa
        juego._handle_key(13)                                  # ENTER: empieza el vídeo
        self.assertFalse(video.paused)
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)
        self.pasar_frames(juego, reposo, 60)
        self.assertIs(juego.phase, Phase.COUNTDOWN)

        juego._handle_key(ord("p"))                            # En pausa el juego se congela
        self.assertTrue(video.paused)
        self.pasar_frames(juego, reposo, 200)
        self.assertIs(juego.phase, Phase.COUNTDOWN)
        juego._handle_key(ord("p"))
        self.pasar_frames(juego, reposo, 100)
        self.assertIs(juego.phase, Phase.ACTIVE)

        self.pasar_poses(juego, poses_repeticion(165, 80))     # Una repetición buena
        self.pasar_frames(juego, reposo, 10)
        self.assertEqual(juego.session.count, 1)

        video.ended = video.paused = True                      # El vídeo se acaba a mitad del ejercicio
        juego._on_video_end()
        self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
        juego._handle_key(13)                                  # El ejercicio que falta se salta
        self.assertIs(juego.phase, Phase.SUMMARY)
        self.assertEqual([s.skipped for s in juego.sessions], [False, True])

        juego._handle_key(13)                                  # Jugar otra vez: rebobina
        self.assertIs(juego.phase, Phase.MENU)
        self.assertFalse(video.ended)
        self.assertEqual(video.position, 0.0)


class TestPartidaConRepeticionDePrueba(PruebaDeJuego):
    def test_la_repeticion_de_prueba_fija_la_referencia_de_la_partida(self):
        juego = crear_juego()
        reposo = pose_flexion(170)

        self.pasar_frames(juego, reposo)
        juego._handle_key(13)
        self.pasar_frames(juego, reposo, 60)
        self.assertIs(juego.phase, Phase.COUNTDOWN)
        self.pasar_frames(juego, reposo, 100)
        self.assertIs(juego.phase, Phase.CALIBRATION)

        self.pasar_poses(juego, poses_repeticion(170, 100))    # Repetición de prueba hasta 100º
        self.pasar_frames(juego, reposo, 3)
        self.assertIs(juego.phase, Phase.ACTIVE)
        self.assertTrue(juego.exercises[0].calibrated)
        self.assertTrue(juego.session.calibrated)
        self.assertAlmostEqual(juego.session.reference["peak"]["codo"], 100.0, delta=4.0)

        self.pasar_frames(juego, reposo, 10)
        for _ in range(2):                                     # Dos repeticiones iguales que la de prueba
            self.pasar_poses(juego, poses_repeticion(170, 100))
            self.pasar_frames(juego, reposo, 10)
        self.assertEqual(juego.session.count, 2)
        self.assertGreater(juego.session.average, 85)

        juego._do("advance")
        self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
        juego._draw(FRAME.copy(), None, self.reloj.ahora)      # La pantalla con la gráfica se dibuja
        juego._do("advance")
        self.assertIs(juego.phase, Phase.SUMMARY)
        self.assertTrue(juego.sessions[0].to_dict()["calibrated"])

    def test_enter_en_la_repeticion_de_prueba_usa_la_referencia_estandar(self):
        juego = crear_juego()
        juego._do("start")
        juego._go(Phase.CALIBRATION)
        juego._do("advance")
        self.assertIs(juego.phase, Phase.ACTIVE)
        self.assertFalse(juego.exercises[0].calibrated)
        self.assertFalse(juego.session.calibrated)


if __name__ == "__main__":
    unittest.main()
