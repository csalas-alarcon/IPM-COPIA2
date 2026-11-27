"""
Pruebas de src/game, src/main.py, src/config.py y scripts/.

Teclas y botones del juego, cambio de fuente, fin del vídeo, guardado de
resultados, historial, argumentos de la línea de comandos, configuración y
scripts. Las partidas completas están en test_logic.py. Sin cámara, sin
ventana y sin red; los resultados se guardan en carpetas temporales.

Ejecutar:  .venv/bin/python -m unittest tests.test_juego -v
(Windows:  .venv\\Scripts\\python -m unittest tests.test_juego -v)
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
SCRIPTS_DIR = TESTS_DIR.parent / "scripts"
for ruta in (SRC_DIR, TESTS_DIR):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import config  # noqa: E402
import main as juego_main  # noqa: E402
from analysis.scoring import RepScore  # noqa: E402
from config import Settings  # noqa: E402
from exercises.catalog import build_exercises  # noqa: E402
from game.game import Phase  # noqa: E402
from game.history import HistoryEntry, load_history  # noqa: E402
from game.session import ExerciseSession, save_sessions  # noqa: E402
from ui.canvas import RED  # noqa: E402
from ui.screens import PickerInfo  # noqa: E402
from utilidades import (FRAME, CamaraFalsa, PruebaDeJuego, VideoFalso, crear_juego,  # noqa: E402
                        pose_flexion)
from vision.camera import NoSource  # noqa: E402


# --- Utilidades ---
class FuenteFalsa:
    """Sustituye a Game._open_source: anota qué se pidió y devuelve una fuente o falla."""

    def __init__(self, fuente=None, error=None):
        self.fuente = fuente
        self.error = error
        self.llamadas = []

    def __call__(self, source):
        self.llamadas.append(source)
        if self.error is not None:
            raise self.error
        return self.fuente


def repeticion(total, consejos=(), tecnica=1.0):
    """Puntuación de una repetición con ese total (el resto de componentes, perfectos)."""
    return RepScore(total=total, rom=1.0, trajectory=1.0, form=tecnica, tempo=1.0,
                    duration=1.5, feedback=list(consejos))


def guardar_partida(carpeta, nombre, fecha, puntos, ejercicios=()):
    """Escribe una partida con el formato de session.py (sólo los campos que lee el historial)."""
    datos = {"date": fecha, "total_points": puntos, "exercises": list(ejercicios)}
    (carpeta / nombre).write_text(json.dumps(datos), encoding="utf-8")


def cargar_script(nombre):
    """Importa un script de scripts/ sin ejecutar su punto de entrada."""
    ruta = SCRIPTS_DIR / nombre
    spec = importlib.util.spec_from_file_location("script_" + ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# --- Teclas del menú y de las pantallas generales ---
class TestTeclasGenerales(PruebaDeJuego):
    def test_escape_en_el_menu_sale(self):
        juego = crear_juego()
        juego._handle_key(27)
        self.assertFalse(juego._running)

    def test_escape_y_q_salen_desde_las_pantallas_de_juego(self):
        for fase in (Phase.INSTRUCTIONS, Phase.ACTIVE, Phase.EXERCISE_RESULT, Phase.SUMMARY):
            for tecla in (27, ord("q"), ord("Q")):
                with self.subTest(fase=fase.name, tecla=tecla):
                    juego = crear_juego()
                    juego._go(fase)
                    juego._handle_key(tecla)
                    self.assertFalse(juego._running)

    def test_enter_y_espacio_empiezan_el_juego_desde_el_menu(self):
        for tecla in (13, 10, 32):
            with self.subTest(tecla=tecla):
                juego = crear_juego()
                juego._handle_key(tecla)
                self.assertIs(juego.phase, Phase.INSTRUCTIONS)

    def test_f_alterna_pantalla_completa(self):
        juego = crear_juego()
        juego._handle_key(ord("f"))
        self.assertTrue(juego._fullscreen)
        juego._handle_key(ord("f"))
        self.assertFalse(juego._fullscreen)

    def test_g_empieza_y_para_la_grabacion_en_results(self):
        juego = crear_juego()
        juego._handle_key(ord("g"))
        self.assertEqual(juego.recorder.path.parent, self.resultados.resolve())
        self.assertRegex(juego.recorder.path.name, r"^grabacion_\d{8}_\d{6}\.mp4$")
        juego._handle_key(ord("g"))
        self.assertIsNone(juego.recorder)

    def test_sin_tecla_pulsada_no_hace_nada(self):
        juego = crear_juego()
        for valor in (-1, 255):                           # Lo que devuelve waitKey sin pulsar nada
            juego._handle_key(valor)
        self.assertIs(juego.phase, Phase.MENU)
        self.assertTrue(juego._running)


class TestMenu(PruebaDeJuego):
    def test_numeros_activan_y_desactivan_ejercicios(self):
        juego = crear_juego()                             # Al principio sólo está marcada la flexión
        juego._handle_key(ord("2"))
        self.assertEqual(juego.selected, {"pushup", "pullup"})
        juego._handle_key(ord("1"))
        self.assertEqual(juego.selected, {"pullup"})

    def test_numero_fuera_de_la_lista_no_hace_nada(self):
        juego = crear_juego()
        juego._handle_key(ord("9"))
        self.assertEqual(juego.selected, {"pushup"})

    def test_empezar_usa_los_ejercicios_marcados_y_la_calibracion_elegida(self):
        juego = crear_juego()
        self.assertEqual([e.key for e in juego.available], ["pushup", "pullup", "bulgarian_squat"])
        juego._handle_key(ord("3"))                       # Activa la sentadilla búlgara
        juego._handle_key(ord("k"))                       # Quita la repetición de prueba
        self.assertFalse(juego.settings.game.calibrate)
        juego._do("start")
        self.assertEqual([e.key for e in juego.exercises], ["pushup", "bulgarian_squat"])
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)

    def test_sin_ejercicios_marcados_no_empieza_y_avisa(self):
        juego = crear_juego()
        juego.selected.clear()
        juego._do("start")
        self.assertIs(juego.phase, Phase.MENU)
        self.assertIsNotNone(juego._toast)

    def test_sin_fuente_no_empieza(self):
        juego = crear_juego(camara=NoSource(320, 240))
        juego._do("start")
        self.assertIs(juego.phase, Phase.MENU)

    def test_r_vuelve_al_menu_y_borra_la_partida(self):
        juego = crear_juego()
        juego._do("start")
        juego._handle_key(ord("r"))
        self.assertIs(juego.phase, Phase.MENU)
        self.assertEqual(juego.sessions, [])


# --- Selector de vídeo ---
class TestSelectorDeVideo(PruebaDeJuego):
    def setUp(self):
        super().setUp()
        self.videos = [Path("videos/a.mp4"), Path("videos/b.mp4")]
        self.fuente = FuenteFalsa(VideoFalso())
        self.juego = crear_juego(camara=NoSource(320, 240), open_source=self.fuente)
        self.juego._picker = PickerInfo(self.videos, Path("videos"), False)
        self.juego._go(Phase.PICK_VIDEO)

    def test_escape_vuelve_al_menu_sin_salir(self):
        self.juego._handle_key(27)
        self.assertIs(self.juego.phase, Phase.MENU)
        self.assertTrue(self.juego._running)

    def test_numero_elige_el_video_de_la_lista(self):
        self.juego._handle_key(ord("2"))
        self.assertEqual(self.fuente.llamadas, [str(self.videos[1])])
        self.assertIs(self.juego.phase, Phase.MENU)
        self.assertIs(self.juego.camera, self.fuente.fuente)

    def test_numero_fuera_de_la_lista_no_abre_nada(self):
        self.juego._handle_key(ord("5"))
        self.assertEqual(self.fuente.llamadas, [])
        self.assertIs(self.juego.phase, Phase.PICK_VIDEO)

    def test_b_usa_el_archivo_elegido_en_el_explorador(self):
        with mock.patch("game.game.ask_video_file", return_value=Path("otro/c.mp4")):
            self.juego._handle_key(ord("b"))
        self.assertEqual(self.fuente.llamadas, [Path("otro/c.mp4")])
        self.assertIs(self.juego.phase, Phase.MENU)

    def test_b_cancelado_sigue_en_el_selector(self):
        with mock.patch("game.game.ask_video_file", return_value=None):
            self.juego._handle_key(ord("b"))
        self.assertEqual(self.fuente.llamadas, [])
        self.assertIs(self.juego.phase, Phase.PICK_VIDEO)

    def test_v_en_el_menu_abre_el_selector_con_los_videos_encontrados(self):
        juego = crear_juego()
        with mock.patch("game.game.list_videos", return_value=self.videos), \
                mock.patch("game.game.tk_available", return_value=False):
            juego._handle_key(ord("v"))
        self.assertIs(juego.phase, Phase.PICK_VIDEO)
        self.assertEqual(juego._picker.videos, self.videos)
        self.assertFalse(juego._picker.can_browse)


# --- Cambio de fuente ---
class TestCambioDeFuente(PruebaDeJuego):
    def test_boton_camara_sin_camara_la_abre(self):
        fuente = FuenteFalsa(CamaraFalsa())
        juego = crear_juego(camara=NoSource(320, 240), open_source=fuente)
        juego._do("camera")
        self.assertEqual(fuente.llamadas, [0])
        self.assertIs(juego.camera, fuente.fuente)
        self.assertEqual(juego._toast[0], "Fuente: Cámara 0")

    def test_si_falla_la_fuente_no_cambia(self):
        for error in (RuntimeError("No se pudo abrir la cámara 0"), ValueError("Fuente inválida")):
            with self.subTest(error=type(error).__name__):
                camara = NoSource(320, 240)
                juego = crear_juego(camara=camara, open_source=FuenteFalsa(error=error))
                juego._do("camera")
                self.assertIs(juego.camera, camara)
                self.assertEqual(juego._toast[0], str(error))
                self.assertEqual(juego._toast[1], RED)

    def test_boton_camara_ya_activa_no_vuelve_a_abrirla(self):
        fuente = FuenteFalsa(CamaraFalsa())
        juego = crear_juego(open_source=fuente)
        juego._do("camera")
        self.assertEqual(fuente.llamadas, [])
        self.assertEqual(juego._toast[0], "Ya estás usando Cámara 0")

    def test_boton_camara_con_video_vuelve_a_la_camara_y_libera_el_video(self):
        video = VideoFalso()
        fuente = FuenteFalsa(CamaraFalsa())
        juego = crear_juego(camara=video, open_source=fuente)
        juego._do("camera")
        self.assertEqual(fuente.llamadas, [0])
        self.assertIs(juego.camera, fuente.fuente)
        self.assertTrue(video.liberado)


# --- Historial (teclas) ---
class TestTeclasDelHistorial(PruebaDeJuego):
    def test_h_en_el_menu_abre_el_historial(self):
        juego = crear_juego()
        juego._handle_key(ord("h"))
        self.assertIs(juego.phase, Phase.HISTORY)

    def test_escape_y_enter_vuelven_al_menu(self):
        for tecla in (27, 13):
            with self.subTest(tecla=tecla):
                juego = crear_juego()
                juego._handle_key(ord("h"))
                juego._handle_key(tecla)
                self.assertIs(juego.phase, Phase.MENU)
                self.assertTrue(juego._running)

    def test_h_desde_el_resumen_vuelve_al_resumen(self):
        juego = crear_juego()
        juego._go(Phase.SUMMARY)
        juego._handle_key(ord("h"))
        self.assertIs(juego.phase, Phase.HISTORY)
        juego._handle_key(27)
        self.assertIs(juego.phase, Phase.SUMMARY)

    def test_h_durante_el_ejercicio_no_hace_nada(self):
        juego = crear_juego()
        juego._do("start")
        juego._handle_key(ord("h"))
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)


# --- Pausa, salto y reinicio ---
class TestPausaSaltoYReinicio(PruebaDeJuego):
    def test_pausa_no_actua_en_el_menu(self):
        video = VideoFalso()
        juego = crear_juego(camara=video)
        juego._handle_key(ord("p"))
        self.assertTrue(video.paused)                     # Sigue en pausa: no se ha alternado

    def test_pausa_alterna_el_video_durante_el_ejercicio(self):
        video = VideoFalso()
        juego = crear_juego(camara=video)
        juego._do("start")
        self.assertFalse(video.paused)
        juego._handle_key(ord("p"))
        self.assertTrue(video.paused)
        juego._handle_key(ord("p"))
        self.assertFalse(video.paused)

    def test_pausa_con_camara_no_hace_nada(self):
        # CamaraFalsa no tiene toggle_pause: si el juego intentara pausar, fallaría
        juego = crear_juego()
        juego._do("start")
        juego._handle_key(ord("p"))
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)

    def test_r_con_video_lo_rebobina(self):
        video = VideoFalso()
        juego = crear_juego(camara=video)
        juego._do("start")
        video.position = 12.0
        juego._handle_key(ord("r"))
        self.assertIs(juego.phase, Phase.MENU)
        self.assertEqual(video.position, 0.0)
        self.assertTrue(video.paused)

    def test_n_salta_el_ejercicio_en_instrucciones(self):
        juego = crear_juego(claves=("pushup", "pullup"))
        juego._do("start")
        juego._handle_key(ord("n"))
        self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
        self.assertTrue(juego.sessions[0].skipped)
        juego._handle_key(13)                             # Pasa al siguiente ejercicio
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)
        self.assertEqual(juego.index, 1)

    def test_n_salta_el_ejercicio_en_la_repeticion_de_prueba(self):
        juego = crear_juego()
        juego._do("start")
        juego._go(Phase.CALIBRATION)
        juego._handle_key(ord("n"))
        self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
        self.assertTrue(juego.session.skipped)


# --- Fin del vídeo ---
class TestFinDelVideo(PruebaDeJuego):
    def preparar(self, fase):
        """Juego con un vídeo, en la fase dada, cuyo vídeo acaba de terminar."""
        video = VideoFalso()
        juego = crear_juego(camara=video)
        juego._do("start")
        juego._go(fase)
        video.ended = True
        return juego

    def test_en_instrucciones_enter_y_n_saltan_el_ejercicio(self):
        # Sin más frames la cuenta atrás no avanzaría nunca: ENTER salta igual que N
        for tecla in (13, ord("n")):
            with self.subTest(tecla=tecla):
                juego = self.preparar(Phase.INSTRUCTIONS)
                juego._on_video_end()
                self.assertIs(juego.phase, Phase.INSTRUCTIONS)
                juego._handle_key(tecla)
                self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
                self.assertTrue(juego.session.skipped)

    def test_cierra_el_ejercicio_en_cada_fase_de_juego(self):
        for fase in (Phase.COUNTDOWN, Phase.CALIBRATION, Phase.ACTIVE):
            with self.subTest(fase=fase.name):
                juego = self.preparar(fase)
                juego._on_video_end()
                self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
                self.assertTrue(juego.session.skipped)    # Sin repeticiones: cuenta como saltado

    def test_con_repeticiones_el_ejercicio_no_se_salta(self):
        juego = self.preparar(Phase.ACTIVE)
        juego.session.reps.append(repeticion(90.0))
        juego._on_video_end()
        self.assertIs(juego.phase, Phase.EXERCISE_RESULT)
        self.assertFalse(juego.session.skipped)

    def test_fuera_de_un_ejercicio_no_cambia_nada(self):
        for fase in (Phase.MENU, Phase.EXERCISE_RESULT, Phase.SUMMARY):
            with self.subTest(fase=fase.name):
                juego = self.preparar(fase)
                juego._on_video_end()
                self.assertIs(juego.phase, fase)


# --- Ratón ---
class TestRaton(PruebaDeJuego):
    def hacer_clic_en(self, juego, accion):
        """Dibuja la pantalla y pulsa el centro del botón de esa acción."""
        juego._draw(FRAME.copy(), pose_flexion(165), 0.0)
        x1, y1, x2, y2 = next(rectangulo for nombre, rectangulo in juego._buttons if nombre == accion)
        juego._click = ((x1 + x2) // 2, (y1 + y2) // 2)
        juego._handle_click()

    def test_clic_en_un_boton_ejecuta_su_accion(self):
        juego = crear_juego()
        self.hacer_clic_en(juego, "start")
        self.assertIs(juego.phase, Phase.INSTRUCTIONS)

    def test_clic_fuera_de_los_botones_no_hace_nada(self):
        juego = crear_juego()
        juego._draw(FRAME.copy(), None, 0.0)
        juego._click = (-50, -50)
        juego._handle_click()
        self.assertIsNone(juego._click)
        self.assertIs(juego.phase, Phase.MENU)

    def test_clic_en_el_indicador_del_video_pausa_y_reanuda(self):
        video = VideoFalso()
        juego = crear_juego(camara=video)
        juego._do("start")
        self.hacer_clic_en(juego, "pause")
        self.assertTrue(video.paused)
        self.hacer_clic_en(juego, "pause")
        self.assertFalse(video.paused)


# --- Dibujo de las pantallas ---
class TestDibujoDeLasPantallas(PruebaDeJuego):
    def test_todas_las_fases_se_dibujan_en_16_9_y_en_4_3(self):
        juego = crear_juego(claves=("pushup", "pullup", "bulgarian_squat"))
        juego._picker = PickerInfo([Path(f"video_{i}.mp4") for i in range(3)], Path("videos"), True)
        juego._history = [HistoryEntry(Path("x.json"), datetime(2026, 11, 1) + timedelta(days=i),
                                       300 + 50 * i, 8, 70 + i, ["Flexión"]) for i in range(5)]
        juego._do("start")
        for alto, ancho in ((720, 1280), (480, 640)):
            frame = np.zeros((alto, ancho, 3), np.uint8)
            for fase in Phase:
                with self.subTest(fase=fase.name, tamano=(ancho, alto)):
                    juego._go(fase)
                    salida = juego._draw(frame.copy(), pose_flexion(120), 0.0)
                    self.assertEqual(salida.shape, frame.shape)
                    self.assertTrue(salida.any())


# --- Guardado de resultados y resumen ---
class TestGuardadoDeResultados(PruebaDeJuego):
    def terminar_partida_saltando_el_ejercicio(self, juego):
        juego._do("start")
        juego._handle_key(ord("n"))
        juego._handle_key(13)

    def test_al_terminar_se_guarda_el_json_en_results(self):
        juego = crear_juego()
        juego.settings.game.save_results = True
        self.terminar_partida_saltando_el_ejercicio(juego)
        self.assertIs(juego.phase, Phase.SUMMARY)
        self.assertEqual(juego.saved_path.parent, self.resultados.resolve())
        datos = json.loads(juego.saved_path.read_text(encoding="utf-8"))
        self.assertTrue(datos["exercises"][0]["skipped"])

    def test_el_resumen_compara_con_la_partida_anterior(self):
        guardar_partida(self.resultados, "session_20260101_100000.json", "2026-01-01T10:00:00", 250)
        juego = crear_juego()
        juego.settings.game.save_results = True
        self.terminar_partida_saltando_el_ejercicio(juego)
        self.assertEqual(juego._previous.total_points, 250)

    def test_sin_guardar_no_escribe_nada(self):
        juego = crear_juego()                             # crear_juego desactiva el guardado
        self.terminar_partida_saltando_el_ejercicio(juego)
        self.assertIsNone(juego.saved_path)
        self.assertEqual(list(self.resultados.iterdir()), [])


# --- Sesión de un ejercicio y guardado en JSON ---
class TestSesion(unittest.TestCase):
    def setUp(self):
        self.flexion = build_exercises(["pushup"])[0]
        self.carpeta = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_sesion_sin_repeticiones(self):
        sesion = ExerciseSession(self.flexion)
        self.assertEqual(sesion.count, 0)
        self.assertEqual(sesion.points, 0)
        self.assertEqual(sesion.average, 0.0)
        self.assertEqual(sesion.best, 0.0)
        self.assertIsNone(sesion.best_rep)
        self.assertEqual(sesion.top_feedback(), [])
        self.assertEqual(sesion.component_average("form"), 0.0)

    def test_puntos_media_y_mejor_repeticion(self):
        sesion = ExerciseSession(self.flexion, reps=[repeticion(80), repeticion(90), repeticion(100)])
        self.assertEqual(sesion.count, 3)
        self.assertEqual(sesion.points, 270)
        self.assertAlmostEqual(sesion.average, 90.0)
        self.assertEqual(sesion.best, 100)
        self.assertEqual(sesion.best_rep.total, 100)

    def test_los_puntos_se_redondean_al_entero(self):
        sesion = ExerciseSession(self.flexion, reps=[repeticion(90.4), repeticion(90.4)])
        self.assertEqual(sesion.points, 181)                  # 180,8 -> 181

    def test_media_por_componente(self):
        sesion = ExerciseSession(self.flexion, reps=[repeticion(80, tecnica=1.0), repeticion(90, tecnica=0.5)])
        self.assertAlmostEqual(sesion.component_average("form"), 0.75)
        self.assertAlmostEqual(sesion.component_average("rom"), 1.0)

    def test_consejos_mas_repetidos(self):
        sesion = ExerciseSession(self.flexion, reps=[
            repeticion(80, ["a", "b"]), repeticion(80, ["a"]),
            repeticion(80, ["b", "c"]), repeticion(80, ["a"])])
        self.assertEqual(sesion.top_feedback(), ["a", "b"])
        self.assertEqual(sesion.top_feedback(1), ["a"])

    def test_to_dict_tiene_los_campos_del_readme(self):
        sesion = ExerciseSession(self.flexion, reps=[repeticion(100)], partial_reps=1, duration=12.4,
                                 calibrated=True,
                                 reference={"rest": {"codo": 171.2}, "peak": {"codo": 98.4}})
        datos = sesion.to_dict()
        campos = {"exercise", "name", "skipped", "reps", "partial_reps", "points", "average_score",
                  "best_score", "duration_s", "calibrated", "reference", "repetitions"}
        self.assertEqual(set(datos), campos)
        self.assertEqual(datos["exercise"], "pushup")
        self.assertEqual(datos["name"], "Flexión")
        self.assertEqual(datos["reps"], 1)
        self.assertEqual(datos["partial_reps"], 1)
        self.assertTrue(datos["calibrated"])
        self.assertEqual(datos["repetitions"][0]["total"], 100)

    def test_save_sessions_escribe_un_json_valido(self):
        dominada = build_exercises(["pullup"])[0]
        sesion1 = ExerciseSession(self.flexion, reps=[repeticion(90), repeticion(100)])
        sesion2 = ExerciseSession(dominada, skipped=True)
        ruta = save_sessions([sesion1, sesion2], self.carpeta)
        texto = ruta.read_text(encoding="utf-8")
        datos = json.loads(texto)
        self.assertEqual(set(datos), {"date", "total_points", "exercises"})
        self.assertEqual(datos["total_points"], 190)
        self.assertEqual([e["exercise"] for e in datos["exercises"]], ["pushup", "pullup"])
        self.assertIn("Flexión", texto)                   # Las tildes se guardan sin escapar
        self.assertRegex(ruta.name, r"^session_\d{8}_\d{6}\.json$")
        datetime.fromisoformat(datos["date"])

    def test_dos_partidas_en_el_mismo_segundo_no_se_sobrescriben(self):
        sesion = ExerciseSession(self.flexion, reps=[repeticion(90)])
        rutas = [save_sessions([sesion], self.carpeta) for _ in range(3)]
        self.assertEqual(len(set(rutas)), 3)
        self.assertEqual(len(list(self.carpeta.glob("session_*.json"))), 3)
        self.assertEqual(len(load_history(self.carpeta)), 3)

    def test_save_sessions_crea_la_carpeta_si_no_existe(self):
        carpeta = self.carpeta / "nueva" / "resultados"
        ruta = save_sessions([ExerciseSession(self.flexion)], carpeta)
        self.assertTrue(ruta.is_file())
        self.assertEqual(ruta.parent, carpeta.resolve())


# --- Historial de partidas ---
class TestHistorial(unittest.TestCase):
    def setUp(self):
        self.carpeta = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_ordena_por_fecha_y_no_por_nombre(self):
        guardar_partida(self.carpeta, "session_zz.json", "2026-01-01T10:00:00", 100)
        guardar_partida(self.carpeta, "session_aa.json", "2026-01-02T10:00:00", 200)
        self.assertEqual([e.total_points for e in load_history(self.carpeta)], [100, 200])

    def test_ignora_archivos_corruptos_y_que_no_son_de_sesion(self):
        (self.carpeta / "session_roto.json").write_text("{no es json", encoding="utf-8")
        (self.carpeta / "session_sin_fecha.json").write_text('{"total_points": 5}', encoding="utf-8")
        (self.carpeta / "session_fecha_mala.json").write_text('{"date": "ayer"}', encoding="utf-8")
        guardar_partida(self.carpeta, "otro.json", "2026-01-01T10:00:00", 7)     # No empieza por session_
        guardar_partida(self.carpeta, "session_ok.json", "2026-01-01T10:00:00", 8)
        self.assertEqual([e.path.name for e in load_history(self.carpeta)], ["session_ok.json"])

    def test_media_y_ejercicios_ignoran_los_saltados(self):
        ejercicios = [
            {"name": "Flexión", "skipped": False, "repetitions": [{"total": 80.0}, {"total": 100.0}]},
            {"name": "Dominada", "skipped": True, "repetitions": []},
        ]
        guardar_partida(self.carpeta, "session_1.json", "2026-01-01T10:00:00", 180, ejercicios)
        (entrada,) = load_history(self.carpeta)
        self.assertEqual(entrada.reps, 2)
        self.assertAlmostEqual(entrada.average_score, 90.0)
        self.assertEqual(entrada.exercises, ["Flexión"])

    def test_lee_lo_que_escribe_save_sessions(self):
        flexion, dominada = build_exercises(["pushup", "pullup"])
        hecha = ExerciseSession(flexion, reps=[repeticion(90), repeticion(70)])
        saltada = ExerciseSession(dominada, skipped=True)
        save_sessions([hecha, saltada], self.carpeta)
        (entrada,) = load_history(self.carpeta)
        self.assertEqual(entrada.total_points, 160)
        self.assertEqual(entrada.reps, 2)
        self.assertAlmostEqual(entrada.average_score, 80.0)
        self.assertEqual(entrada.exercises, ["Flexión"])

    def test_carpeta_vacia_o_inexistente(self):
        self.assertEqual(load_history(self.carpeta), [])
        self.assertEqual(load_history(self.carpeta / "no_existe"), [])


# --- Argumentos de línea de comandos (main.py) ---
class TestMain(unittest.TestCase):
    def setUp(self):
        # apply_args modifica config.settings: se guarda y se restaura tras cada test
        self.guardada = copy.deepcopy(config.settings)
        self.addCleanup(self.restaurar)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def restaurar(self):
        for campo in dataclasses.fields(config.settings):
            setattr(config.settings, campo.name, getattr(self.guardada, campo.name))

    def aplicar(self, *argumentos):
        """Lee la línea de comandos dada y la aplica a config.settings, como hace main()."""
        juego_main.apply_args(juego_main.parse_args(list(argumentos)))

    def test_argumentos_por_defecto(self):
        args = juego_main.parse_args([])
        self.assertIsNone(args.video)
        self.assertIsNone(args.source)
        self.assertIsNone(args.record)
        self.assertEqual(args.start, 0.0)
        self.assertEqual(args.exercises, list(Settings().game.exercises))

    def test_fuente_y_video_son_excluyentes(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            juego_main.parse_args(["--video", "demo.mp4", "--source", "1"])

    def test_no_mirror_y_mirror_son_excluyentes(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            juego_main.parse_args(["--no-mirror", "--mirror"])

    def test_espejo_por_defecto_solo_en_la_camara(self):
        self.aplicar()
        self.assertTrue(config.settings.camera.mirror)
        self.assertFalse(config.settings.camera.mirror_video)

    def test_no_mirror_quita_el_espejo_y_mirror_lo_pone_en_el_video(self):
        self.aplicar("--no-mirror")
        self.assertFalse(config.settings.camera.mirror)
        self.aplicar("--mirror")
        self.assertTrue(config.settings.camera.mirror_video)

    def test_video_inexistente_lanza_error(self):
        with self.assertRaises(FileNotFoundError):
            self.aplicar("--video", str(self.tmp / "no_existe.mp4"))

    def test_video_existente_es_la_fuente(self):
        ruta = self.tmp / "demo.mp4"
        ruta.write_bytes(b"")
        self.aplicar("--video", str(ruta))
        self.assertEqual(config.settings.camera.source, str(ruta))

    def test_start_negativo_se_queda_en_cero(self):
        self.aplicar("--start", "-5")
        self.assertEqual(config.settings.camera.start_seconds, 0.0)
        self.aplicar("--start", "4.5")
        self.assertEqual(config.settings.camera.start_seconds, 4.5)

    def test_record_sin_archivo_usa_una_ruta_en_results(self):
        self.aplicar("--record")
        ruta = Path(config.settings.game.record_path)
        self.assertEqual(ruta.parent, config.RESULTS_DIR)
        self.assertRegex(ruta.name, r"^grabacion_\d{8}_\d{6}\.mp4$")
        self.assertFalse(ruta.exists())                   # Sólo se calcula; no se crea el archivo

    def test_record_con_archivo_usa_esa_ruta(self):
        self.aplicar("--record", "mi_partida.mp4")
        self.assertEqual(config.settings.game.record_path, "mi_partida.mp4")

    def test_flags_de_calibracion_sonido_y_guardado(self):
        self.aplicar("--no-calibrate", "--no-sound", "--no-save")
        self.assertFalse(config.settings.game.calibrate)
        self.assertFalse(config.settings.game.sound)
        self.assertFalse(config.settings.game.save_results)

    def test_ejercicios_partida_y_modelo(self):
        self.aplicar("--exercises", "pushup", "pullup", "--reps", "8", "--duration", "45",
                     "--model", "lite", "--width", "640", "--height", "480")
        self.assertEqual(config.settings.game.exercises, ("pushup", "pullup"))
        self.assertEqual(config.settings.game.target_reps, 8)
        self.assertEqual(config.settings.game.exercise_duration, 45.0)
        self.assertEqual(config.settings.model.variant, "lite")
        self.assertEqual((config.settings.camera.width, config.settings.camera.height), (640, 480))

    def test_ejercicio_desconocido_se_rechaza(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            juego_main.parse_args(["--exercises", "sprint"])

    def test_sin_camara_ni_fuente_pedida_arranca_sin_fuente(self):
        args = juego_main.parse_args([])
        with mock.patch.object(juego_main, "Camera", side_effect=RuntimeError("sin cámara")), \
                contextlib.redirect_stdout(io.StringIO()):
            fuente = juego_main.open_initial_source(args)
        self.assertIsInstance(fuente, NoSource)

    def test_si_se_pidio_una_fuente_y_falla_el_error_se_propaga(self):
        args = juego_main.parse_args(["--source", "3"])
        with mock.patch.object(juego_main, "Camera", side_effect=RuntimeError("sin cámara")), \
                self.assertRaises(RuntimeError):
            juego_main.open_initial_source(args)

    def test_main_con_video_inexistente_devuelve_1(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            codigo = juego_main.main(["--video", str(self.tmp / "no_existe.mp4")])
        self.assertEqual(codigo, 1)
        self.assertIn("No existe el archivo de vídeo", error.getvalue())


# --- Configuración (config.py) ---
class TestConfiguracion(unittest.TestCase):
    def test_rutas_son_absolutas(self):
        rutas = (config.PROJECT_ROOT, config.SRC_DIR, config.SCRIPTS_DIR,
                 config.MODELS_DIR, config.RESULTS_DIR, config.VIDEOS_DIR)
        for ruta in rutas:
            with self.subTest(ruta=str(ruta)):
                self.assertTrue(ruta.is_absolute())

    def test_raiz_del_proyecto_tiene_src_y_scripts(self):
        self.assertTrue((config.PROJECT_ROOT / "src" / "main.py").is_file())
        self.assertTrue((config.SCRIPTS_DIR / "run.sh").is_file())

    def test_variantes_del_modelo(self):
        self.assertEqual(config.MODEL_VARIANTS, ("lite", "full", "heavy"))
        self.assertEqual(set(config.MODEL_URLS), set(config.MODEL_VARIANTS))

    def test_ruta_del_modelo_esta_en_models(self):
        self.assertEqual(config.model_filename("heavy"), "pose_landmarker_heavy.task")
        self.assertEqual(config.model_path("heavy"), config.MODELS_DIR / "pose_landmarker_heavy.task")

    def test_valores_por_defecto_como_dice_el_readme(self):
        ajustes = Settings()
        self.assertEqual(ajustes.game.exercises, ("pullup", "pushup", "bulgarian_squat"))
        self.assertEqual(ajustes.model.variant, "full")
        self.assertEqual(ajustes.game.exercise_duration, 60.0)
        self.assertEqual(ajustes.game.target_reps, 10)
        self.assertTrue(ajustes.game.calibrate)

    def test_cada_ajuste_es_independiente(self):
        primeros, segundos = Settings(), Settings()
        primeros.camera.width = 640
        self.assertEqual(segundos.camera.width, 1280)

    def test_umbral_de_reposo_menor_que_el_de_maximo(self):
        ajustes = Settings()
        self.assertLess(ajustes.scoring.rest_threshold, ajustes.scoring.peak_threshold)


# --- Scripts (scripts/) ---
@unittest.skipUnless(shutil.which("bash"), "Hace falta bash")
class TestScriptsSh(unittest.TestCase):
    def test_run_y_dependencies_tienen_sintaxis_bash_correcta(self):
        for nombre in ("run.sh", "dependencies.sh"):
            with self.subTest(script=nombre):
                resultado = subprocess.run(["bash", "-n", str(SCRIPTS_DIR / nombre)],
                                           capture_output=True, text=True)
                self.assertEqual(resultado.returncode, 0, resultado.stderr)


class TestMakeZip(unittest.TestCase):
    def setUp(self):
        self.make_zip = cargar_script("make_zip.py")

    def test_slug_quita_tildes_y_espacios(self):
        self.assertEqual(self.make_zip.slug("José Pérez Ñúñez"), "jose_perez_nunez")
        self.assertEqual(self.make_zip.slug("  Ana   María  "), "ana_maria")
        self.assertEqual(self.make_zip.slug("¡¿?"), "")

    def test_iter_files_excluye_lo_que_no_va_en_la_entrega(self):
        archivos = ["README.md", "src/main.py", "src/__pycache__/main.cpython-312.pyc",
                    "src/vision/modelo.task", ".venv/bin/python", "models/pose_landmarker_full.task",
                    "results/session_1.json", "mp_env/x.py", ".git/config", "build.zip",
                    "videos/demo.mp4", "videos/LEEME.md", "videos/sub/otro.mp4"]
        raiz = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for nombre in archivos:
            ruta = raiz / nombre
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(b"x")
        incluidos = {rel.as_posix() for _, rel in self.make_zip.iter_files(raiz)}
        self.assertEqual(incluidos, {"README.md", "src/main.py", "videos/LEEME.md"})

    def test_sin_nombre_valido_no_genera_nada(self):
        error = io.StringIO()
        with mock.patch.object(sys, "argv", ["make_zip.py", "¡¿?"]), \
                contextlib.redirect_stderr(error):
            codigo = self.make_zip.main()
        self.assertEqual(codigo, 1)
        self.assertIn("vacío", error.getvalue())


class TestDownloadModels(unittest.TestCase):
    def setUp(self):
        self.dm = cargar_script("download_models.py")
        # Sin red: la descarga se simula
        self.descarga = self.enterContext(mock.patch.object(self.dm, "download_model"))

    def ejecutar(self, *argumentos):
        with mock.patch.object(sys, "argv", ["download_models.py", *argumentos]):
            return self.dm.main()

    def test_sin_argumentos_descarga_las_tres_variantes(self):
        self.assertEqual(self.ejecutar(), 0)
        self.assertEqual([c.args[0] for c in self.descarga.call_args_list], list(config.MODEL_VARIANTS))

    def test_model_y_force_se_pasan_a_la_descarga(self):
        self.ejecutar("--model", "lite", "--force")
        self.assertEqual(self.descarga.call_args_list, [mock.call("lite", force=True)])

    def test_un_error_de_descarga_devuelve_1(self):
        self.descarga.side_effect = RuntimeError("sin conexión")
        with contextlib.redirect_stderr(io.StringIO()) as error:
            codigo = self.ejecutar("--model", "full")
        self.assertEqual(codigo, 1)
        self.assertIn("sin conexión", error.getvalue())

    def test_modelo_desconocido_se_rechaza(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.ejecutar("--model", "tiny")


if __name__ == "__main__":
    unittest.main()
