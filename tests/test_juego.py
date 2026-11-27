"""
tests/test_juego.py  ·  Pruebas de teclas, fuentes, resultados, argumentos y scripts.

Complementa a tests/test_logic.py (flujo completo y calibración): aquí se prueban
el mapa de teclas, el cambio de fuente, el guardado y el historial, los argumentos
de línea de comandos, la configuración y los scripts. Sin cámara, sin ventana y sin red.

Ejecutar:  .venv/bin/python -m unittest tests.test_juego -v
(Windows:  .venv\\Scripts\\python -m unittest tests.test_juego -v)
"""

# =============================================================================
# Importaciones
# =============================================================================
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
from datetime import datetime
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
from game.game import Game, Phase  # noqa: E402
from game.history import load_history  # noqa: E402
from game.session import ExerciseSession, save_sessions  # noqa: E402
from test_logic import FakeCamera, FakeVideo, make_game, pushup_pose  # noqa: E402
from ui.canvas import RED, TextRenderer  # noqa: E402
from ui.screens import PickerInfo  # noqa: E402
from vision.camera import NoSource  # noqa: E402

FRAME = np.zeros((720, 1280, 3), np.uint8)


# =============================================================================
# Utilidades
# =============================================================================
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


class VideoConSalida(FakeVideo):
    """FakeVideo que anota si se liberó (al cambiar de fuente el juego lo libera)."""

    def __init__(self):
        super().__init__()
        self.liberado = False

    def release(self):
        self.liberado = True


def juego_con_fuente(camara, fuente):
    """Como make_game, pero con una función falsa para abrir las fuentes."""
    ajustes = Settings()
    ajustes.game.save_results = False
    return Game(ajustes, camara, None, build_exercises(["pushup"]), TextRenderer(), open_source=fuente)


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


class AisladoDeResultados(unittest.TestCase):
    """Los tests que guardan o leen partidas usan una carpeta temporal, nunca results/."""

    def setUp(self):
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.resultados = Path(carpeta.name)
        parche = mock.patch("game.game.RESULTS_DIR", self.resultados)
        parche.start()
        self.addCleanup(parche.stop)


# =============================================================================
# Teclas del menú y de las pantallas generales
# =============================================================================
class TestTeclasDelMenu(AisladoDeResultados):
    def test_escape_en_el_menu_sale(self):
        g = make_game()
        g._handle_key(27)
        self.assertFalse(g._running)

    def test_escape_y_q_salen_desde_las_pantallas_de_juego(self):
        for fase in (Phase.INSTRUCTIONS, Phase.ACTIVE, Phase.EXERCISE_RESULT, Phase.SUMMARY):
            for tecla in (27, ord("q"), ord("Q")):
                with self.subTest(fase=fase.name, tecla=tecla):
                    g = make_game()
                    g._go(fase)
                    g._handle_key(tecla)
                    self.assertFalse(g._running)

    def test_espacio_empieza_el_juego(self):
        g = make_game()
        g._handle_key(32)
        self.assertIs(g.phase, Phase.INSTRUCTIONS)

    def test_numeros_activan_y_desactivan_ejercicios(self):
        g = make_game()                                   # Al principio sólo está marcada la flexión
        g._handle_key(ord("2"))
        self.assertEqual(g.selected, {"pushup", "pullup"})
        g._handle_key(ord("1"))
        self.assertEqual(g.selected, {"pullup"})

    def test_numero_fuera_de_la_lista_no_hace_nada(self):
        g = make_game()
        g._handle_key(ord("9"))
        self.assertEqual(g.selected, {"pushup"})

    def test_r_vuelve_al_menu_y_borra_la_partida(self):
        g = make_game()
        g._do("start")
        g._handle_key(ord("r"))
        self.assertIs(g.phase, Phase.MENU)
        self.assertEqual(g.sessions, [])

    def test_f_alterna_pantalla_completa(self):
        g = make_game()
        g._handle_key(ord("f"))
        self.assertTrue(g._fullscreen)
        g._handle_key(ord("f"))
        self.assertFalse(g._fullscreen)

    def test_g_empieza_y_para_la_grabacion(self):
        g = make_game()
        g._handle_key(ord("g"))
        self.assertIsNotNone(g.recorder)
        g._handle_key(ord("g"))
        self.assertIsNone(g.recorder)

    def test_sin_tecla_pulsada_no_hace_nada(self):
        g = make_game()
        for valor in (-1, 255):                           # Lo que devuelve waitKey sin pulsar nada
            g._handle_key(valor)
        self.assertIs(g.phase, Phase.MENU)
        self.assertTrue(g._running)


# =============================================================================
# Selector de vídeo
# =============================================================================
class TestTeclasDelSelectorDeVideo(AisladoDeResultados):
    def setUp(self):
        super().setUp()
        self.videos = [Path("/videos/a.mp4"), Path("/videos/b.mp4")]
        self.fuente = FuenteFalsa(VideoConSalida())
        self.g = juego_con_fuente(NoSource(320, 240), self.fuente)
        self.g._picker = PickerInfo(self.videos, Path("/videos"), False)
        self.g._go(Phase.PICK_VIDEO)

    def test_escape_vuelve_al_menu_sin_salir(self):
        self.g._handle_key(27)
        self.assertIs(self.g.phase, Phase.MENU)
        self.assertTrue(self.g._running)

    def test_numero_elige_el_video_de_la_lista(self):
        self.g._handle_key(ord("2"))
        self.assertEqual(self.fuente.llamadas, [str(self.videos[1])])
        self.assertIs(self.g.phase, Phase.MENU)
        self.assertIs(self.g.camera, self.fuente.fuente)

    def test_numero_fuera_de_la_lista_no_abre_nada(self):
        self.g._handle_key(ord("5"))
        self.assertEqual(self.fuente.llamadas, [])
        self.assertIs(self.g.phase, Phase.PICK_VIDEO)

    def test_b_busca_en_el_ordenador_y_usa_el_archivo_elegido(self):
        with mock.patch("game.game.ask_video_file", return_value=Path("/x/c.mp4")):
            self.g._handle_key(ord("b"))
        self.assertEqual(self.fuente.llamadas, [Path("/x/c.mp4")])
        self.assertIs(self.g.phase, Phase.MENU)

    def test_b_cancelado_sigue_en_el_selector(self):
        with mock.patch("game.game.ask_video_file", return_value=None):
            self.g._handle_key(ord("b"))
        self.assertEqual(self.fuente.llamadas, [])
        self.assertIs(self.g.phase, Phase.PICK_VIDEO)

    def test_v_en_el_menu_abre_el_selector_con_los_videos_encontrados(self):
        g = make_game()
        with mock.patch("game.game.list_videos", return_value=self.videos), \
                mock.patch("game.game.tk_available", return_value=False):
            g._handle_key(ord("v"))
        self.assertIs(g.phase, Phase.PICK_VIDEO)
        self.assertEqual(g._picker.videos, self.videos)
        self.assertFalse(g._picker.can_browse)


# =============================================================================
# Historial
# =============================================================================
class TestTeclasDelHistorial(AisladoDeResultados):
    def test_h_en_el_menu_abre_el_historial(self):
        g = make_game()
        g._handle_key(ord("h"))
        self.assertIs(g.phase, Phase.HISTORY)

    def test_escape_y_enter_vuelven_al_menu(self):
        for tecla in (27, 13):
            with self.subTest(tecla=tecla):
                g = make_game()
                g._handle_key(ord("h"))
                g._handle_key(tecla)
                self.assertIs(g.phase, Phase.MENU)
                self.assertTrue(g._running)

    def test_h_desde_el_resumen_vuelve_al_resumen(self):
        g = make_game()
        g._go(Phase.SUMMARY)
        g._handle_key(ord("h"))
        self.assertIs(g.phase, Phase.HISTORY)
        g._handle_key(27)
        self.assertIs(g.phase, Phase.SUMMARY)

    def test_h_durante_el_ejercicio_no_hace_nada(self):
        g = make_game()
        g._do("start")
        g._handle_key(ord("h"))
        self.assertIs(g.phase, Phase.INSTRUCTIONS)


# =============================================================================
# Pausa, salto y reinicio
# =============================================================================
class TestPausaSaltoYReinicio(AisladoDeResultados):
    def test_pausa_no_actua_en_el_menu(self):
        cam = FakeVideo()
        g = make_game(camera=cam)
        g._handle_key(ord("p"))
        self.assertTrue(cam.paused)

    def test_pausa_alterna_el_video_durante_el_ejercicio(self):
        cam = FakeVideo()
        g = make_game(camera=cam)
        g._do("start")
        self.assertFalse(cam.paused)
        g._handle_key(ord("p"))
        self.assertTrue(cam.paused)
        g._handle_key(ord("p"))
        self.assertFalse(cam.paused)

    def test_pausa_no_hace_nada_con_camara(self):
        # FakeCamera no tiene toggle_pause: si el juego intentara pausar, fallaría
        g = make_game(camera=FakeCamera())
        g._do("start")
        g._handle_key(ord("p"))
        self.assertIs(g.phase, Phase.INSTRUCTIONS)

    def test_r_con_video_lo_rebobina(self):
        cam = FakeVideo()
        g = make_game(camera=cam)
        g._do("start")
        cam.position = 12.0
        g._handle_key(ord("r"))
        self.assertIs(g.phase, Phase.MENU)
        self.assertEqual(cam.position, 0.0)
        self.assertTrue(cam.paused)

    def test_n_salta_el_ejercicio_en_instrucciones(self):
        g = make_game(keys=("pushup", "pullup"))
        g._do("start")
        g._handle_key(ord("n"))
        self.assertIs(g.phase, Phase.EXERCISE_RESULT)
        self.assertTrue(g.sessions[0].skipped)
        g._handle_key(13)                                 # Pasa al siguiente ejercicio
        self.assertIs(g.phase, Phase.INSTRUCTIONS)
        self.assertEqual(g.index, 1)

    def test_n_salta_el_ejercicio_en_calibracion(self):
        g = make_game()
        g._do("start")
        g._go(Phase.CALIBRATION)
        g._handle_key(ord("n"))
        self.assertIs(g.phase, Phase.EXERCISE_RESULT)
        self.assertTrue(g.session.skipped)


# =============================================================================
# Cambio de fuente
# =============================================================================
class TestCambioDeFuente(AisladoDeResultados):
    def test_boton_camara_sin_camara_la_abre(self):
        fuente = FuenteFalsa(FakeCamera())
        g = juego_con_fuente(NoSource(320, 240), fuente)
        g._do("camera")
        self.assertEqual(fuente.llamadas, [0])
        self.assertIs(g.camera, fuente.fuente)
        self.assertEqual(g._toast[0], "Fuente: Cámara 0")

    def test_si_falla_la_fuente_no_cambia(self):
        for error in (RuntimeError("No se pudo abrir la cámara 0"), ValueError("Fuente inválida")):
            with self.subTest(error=type(error).__name__):
                camara = NoSource(320, 240)
                g = juego_con_fuente(camara, FuenteFalsa(error=error))
                g._do("camera")
                self.assertIs(g.camera, camara)
                self.assertEqual(g._toast[0], str(error))
                self.assertEqual(g._toast[1], RED)

    def test_boton_camara_ya_activa_no_vuelve_a_abrirla(self):
        fuente = FuenteFalsa(FakeCamera())
        g = juego_con_fuente(FakeCamera(), fuente)
        g._do("camera")
        self.assertEqual(fuente.llamadas, [])
        self.assertEqual(g._toast[0], "Ya estás usando Cámara 0")

    def test_boton_camara_con_video_vuelve_a_la_camara(self):
        video = VideoConSalida()
        fuente = FuenteFalsa(FakeCamera())
        g = juego_con_fuente(video, fuente)
        g._do("camera")
        self.assertEqual(fuente.llamadas, [0])
        self.assertIs(g.camera, fuente.fuente)
        self.assertTrue(video.liberado)                   # El vídeo anterior se libera


# =============================================================================
# Fin del vídeo
# =============================================================================
class TestFinDelVideo(AisladoDeResultados):
    def test_fin_del_video_en_instrucciones_espera_a_que_se_salte(self):
        # Como dice el README: si se acaba durante las instrucciones, hay que pulsar N
        cam = FakeVideo()
        g = make_game(camera=cam)
        g._do("start")
        cam.ended = True
        with contextlib.redirect_stdout(io.StringIO()):
            g._on_video_end()
        self.assertIs(g.phase, Phase.INSTRUCTIONS)
        g._handle_key(ord("n"))
        self.assertIs(g.phase, Phase.EXERCISE_RESULT)
        self.assertTrue(g.session.skipped)

    def test_fin_del_video_cierra_el_ejercicio_en_cada_fase(self):
        for fase in (Phase.COUNTDOWN, Phase.CALIBRATION, Phase.ACTIVE):
            with self.subTest(fase=fase.name):
                cam = FakeVideo()
                g = make_game(camera=cam)
                g._do("start")
                g._go(fase)
                cam.ended = True
                with contextlib.redirect_stdout(io.StringIO()):
                    g._on_video_end()
                self.assertIs(g.phase, Phase.EXERCISE_RESULT)
                self.assertTrue(g.session.skipped)        # Sin repeticiones: cuenta como saltado

    def test_fin_del_video_con_repeticiones_no_salta_el_ejercicio(self):
        cam = FakeVideo()
        g = make_game(camera=cam)
        g._do("start")
        g._go(Phase.ACTIVE)
        g.session.reps.append(repeticion(90.0))
        cam.ended = True
        with contextlib.redirect_stdout(io.StringIO()):
            g._on_video_end()
        self.assertIs(g.phase, Phase.EXERCISE_RESULT)
        self.assertFalse(g.session.skipped)

    def test_fin_del_video_fuera_de_un_ejercicio_no_cambia_nada(self):
        for fase in (Phase.MENU, Phase.EXERCISE_RESULT, Phase.SUMMARY):
            with self.subTest(fase=fase.name):
                cam = FakeVideo()
                g = make_game(camera=cam)
                g._go(fase)
                cam.ended = True
                with contextlib.redirect_stdout(io.StringIO()):
                    g._on_video_end()
                self.assertIs(g.phase, fase)


# =============================================================================
# Ratón
# =============================================================================
class TestRaton(AisladoDeResultados):
    def test_clic_fuera_de_los_botones_no_hace_nada(self):
        g = make_game()
        g._draw(FRAME.copy(), None, 0.0)
        g._click = (-50, -50)
        g._handle_click()
        self.assertIsNone(g._click)
        self.assertIs(g.phase, Phase.MENU)

    def test_clic_en_el_indicador_del_video_pausa_y_reanuda(self):
        cam = FakeVideo()
        g = make_game(camera=cam)
        g._do("start")
        g._draw(FRAME.copy(), pushup_pose(165), 0.0)
        _, (x1, y1, x2, y2) = next(b for b in g._buttons if b[0] == "pause")
        g._click = ((x1 + x2) // 2, (y1 + y2) // 2)
        g._handle_click()
        self.assertTrue(cam.paused)
        g._click = ((x1 + x2) // 2, (y1 + y2) // 2)
        g._handle_click()
        self.assertFalse(cam.paused)


# =============================================================================
# Guardado de resultados y resumen
# =============================================================================
class TestGuardadoDeResultados(AisladoDeResultados):
    def test_al_terminar_se_guarda_el_json_en_results(self):
        g = make_game()
        g.settings.game.save_results = True
        g._do("start")
        g._handle_key(ord("n"))
        with contextlib.redirect_stdout(io.StringIO()):
            g._handle_key(13)
        self.assertIs(g.phase, Phase.SUMMARY)
        self.assertEqual(g.saved_path.parent, self.resultados.resolve())
        datos = json.loads(g.saved_path.read_text(encoding="utf-8"))
        self.assertTrue(datos["exercises"][0]["skipped"])

    def test_el_resumen_compara_con_la_partida_anterior(self):
        anterior = self.resultados / "session_20260101_100000.json"
        anterior.write_text(json.dumps({"date": "2026-01-01T10:00:00", "total_points": 250,
                                        "exercises": []}), encoding="utf-8")
        g = make_game()
        g.settings.game.save_results = True
        g._do("start")
        g._handle_key(ord("n"))
        with contextlib.redirect_stdout(io.StringIO()):
            g._handle_key(13)
        self.assertEqual(g._previous.total_points, 250)

    def test_sin_guardar_no_escribe_nada(self):
        g = make_game()                                   # make_game desactiva el guardado
        g._do("start")
        g._handle_key(ord("n"))
        g._handle_key(13)
        self.assertIsNone(g.saved_path)
        self.assertEqual(list(self.resultados.iterdir()), [])


# =============================================================================
# Sesión de un ejercicio y guardado en JSON
# =============================================================================
class TestSesion(unittest.TestCase):
    def setUp(self):
        self.flexion = build_exercises(["pushup"])[0]

    def test_sesion_sin_repeticiones(self):
        s = ExerciseSession(self.flexion)
        self.assertEqual(s.count, 0)
        self.assertEqual(s.points, 0)
        self.assertEqual(s.average, 0.0)
        self.assertEqual(s.best, 0.0)
        self.assertIsNone(s.best_rep)
        self.assertEqual(s.top_feedback(), [])
        self.assertEqual(s.component_average("form"), 0.0)

    def test_puntos_media_y_mejor_repeticion(self):
        s = ExerciseSession(self.flexion, reps=[repeticion(80), repeticion(90), repeticion(100)])
        self.assertEqual(s.count, 3)
        self.assertEqual(s.points, 270)
        self.assertAlmostEqual(s.average, 90.0)
        self.assertEqual(s.best, 100)
        self.assertEqual(s.best_rep.total, 100)

    def test_los_puntos_se_redondean_al_entero(self):
        s = ExerciseSession(self.flexion, reps=[repeticion(90.4), repeticion(90.4)])
        self.assertEqual(s.points, 181)                   # 180,8 -> 181

    def test_media_por_componente(self):
        s = ExerciseSession(self.flexion, reps=[repeticion(80, tecnica=1.0), repeticion(90, tecnica=0.5)])
        self.assertAlmostEqual(s.component_average("form"), 0.75)
        self.assertAlmostEqual(s.component_average("rom"), 1.0)

    def test_consejos_mas_repetidos(self):
        s = ExerciseSession(self.flexion, reps=[
            repeticion(80, ["a", "b"]), repeticion(80, ["a"]),
            repeticion(80, ["b", "c"]), repeticion(80, ["a"])])
        self.assertEqual(s.top_feedback(), ["a", "b"])
        self.assertEqual(s.top_feedback(1), ["a"])

    def test_to_dict_tiene_los_campos_del_readme(self):
        s = ExerciseSession(self.flexion, reps=[repeticion(100)], partial_reps=1, duration=12.4,
                            calibrated=True, reference={"rest": {"codo": 171.2}, "peak": {"codo": 98.4}})
        d = s.to_dict()
        self.assertEqual(set(d), {"exercise", "name", "skipped", "reps", "partial_reps", "points",
                                  "average_score", "best_score", "duration_s", "calibrated",
                                  "reference", "repetitions"})
        self.assertEqual(d["exercise"], "pushup")
        self.assertEqual(d["name"], "Flexión")
        self.assertEqual(d["reps"], 1)
        self.assertEqual(d["partial_reps"], 1)
        self.assertTrue(d["calibrated"])
        self.assertEqual(d["repetitions"][0]["total"], 100)

    def test_save_sessions_escribe_un_json_valido(self):
        pullup = build_exercises(["pullup"])[0]
        s1 = ExerciseSession(self.flexion, reps=[repeticion(90), repeticion(100)])
        s2 = ExerciseSession(pullup, skipped=True)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = save_sessions([s1, s2], Path(tmp))
            texto = ruta.read_text(encoding="utf-8")
        datos = json.loads(texto)
        self.assertEqual(set(datos), {"date", "total_points", "exercises"})
        self.assertEqual(datos["total_points"], 190)
        self.assertEqual([e["exercise"] for e in datos["exercises"]], ["pushup", "pullup"])
        self.assertIn("Flexión", texto)                   # Las tildes se guardan sin escapar
        self.assertRegex(ruta.name, r"^session_\d{8}_\d{6}\.json$")
        datetime.fromisoformat(datos["date"])

    def test_save_sessions_crea_la_carpeta_si_no_existe(self):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp) / "nueva" / "resultados"
            ruta = save_sessions([ExerciseSession(self.flexion)], carpeta)
            self.assertTrue(ruta.is_file())
            self.assertEqual(ruta.parent, carpeta.resolve())


# =============================================================================
# Historial de partidas
# =============================================================================
class TestHistorial(unittest.TestCase):
    def setUp(self):
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.carpeta = Path(carpeta.name)

    def test_ordena_por_fecha_y_no_por_nombre(self):
        guardar_partida(self.carpeta, "session_zz.json", "2026-01-01T10:00:00", 100)
        guardar_partida(self.carpeta, "session_aa.json", "2026-01-02T10:00:00", 200)
        self.assertEqual([e.total_points for e in load_history(self.carpeta)], [100, 200])

    def test_limite_devuelve_las_ultimas(self):
        for dia in (1, 2, 3):
            guardar_partida(self.carpeta, f"session_{dia}.json", f"2026-01-0{dia}T10:00:00", dia)
        self.assertEqual([e.total_points for e in load_history(self.carpeta, limit=2)], [2, 3])
        self.assertEqual(len(load_history(self.carpeta)), 3)

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

    def test_carpeta_vacia_o_inexistente(self):
        self.assertEqual(load_history(self.carpeta), [])
        self.assertEqual(load_history(self.carpeta / "no_existe"), [])


# =============================================================================
# Argumentos de línea de comandos (main.py)
# =============================================================================
class TestArgumentosDeMain(unittest.TestCase):
    def setUp(self):
        # apply_args modifica config.settings: se guarda y se restaura tras cada test
        self.guardada = copy.deepcopy(config.settings)
        self.addCleanup(self.restaurar)
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.tmp = Path(carpeta.name)

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

    def test_main_con_video_inexistente_devuelve_1(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            codigo = juego_main.main(["--video", str(self.tmp / "no_existe.mp4")])
        self.assertEqual(codigo, 1)
        self.assertIn("No existe el archivo de vídeo", error.getvalue())


# =============================================================================
# Configuración (config.py)
# =============================================================================
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

    def test_url_de_cada_variante(self):
        for variante in config.MODEL_VARIANTS:
            with self.subTest(variante=variante):
                esperada = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                            f"pose_landmarker_{variante}/float16/latest/pose_landmarker_{variante}.task")
                self.assertEqual(config.MODEL_URLS[variante], esperada)

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
        a, b = Settings(), Settings()
        a.camera.width = 640
        self.assertEqual(b.camera.width, 1280)

    def test_umbral_de_reposo_menor_que_el_de_maximo(self):
        ajustes = Settings()
        self.assertLess(ajustes.scoring.rest_threshold, ajustes.scoring.peak_threshold)


# =============================================================================
# Scripts (scripts/)
# =============================================================================
@unittest.skipUnless(shutil.which("bash"), "Hace falta bash")
class TestScriptsSh(unittest.TestCase):
    def test_run_y_dependencies_tienen_sintaxis_bash_correcta(self):
        for nombre in ("run.sh", "dependencies.sh"):
            with self.subTest(script=nombre):
                r = subprocess.run(["bash", "-n", str(SCRIPTS_DIR / nombre)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)


class TestMakeZip(unittest.TestCase):
    def setUp(self):
        self.zip = cargar_script("make_zip.py")

    def test_slug_quita_tildes_y_espacios(self):
        self.assertEqual(self.zip.slug("José Pérez Ñúñez"), "jose_perez_nunez")
        self.assertEqual(self.zip.slug("  Ana   María  "), "ana_maria")
        self.assertEqual(self.zip.slug("¡¿?"), "")

    def test_iter_files_excluye_lo_que_no_va_en_la_entrega(self):
        archivos = ["README.md", "src/main.py", "src/__pycache__/main.cpython-312.pyc",
                    "src/vision/modelo.task", ".venv/bin/python", "models/pose_landmarker_full.task",
                    "results/session_1.json", "mp_env/x.py", ".git/config", "build.zip",
                    "videos/demo.mp4", "videos/LEEME.md", "videos/sub/otro.mp4"]
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            for nombre in archivos:
                ruta = raiz / nombre
                ruta.parent.mkdir(parents=True, exist_ok=True)
                ruta.write_bytes(b"x")
            incluidos = {rel.as_posix() for _, rel in self.zip.iter_files(raiz)}
        self.assertEqual(incluidos, {"README.md", "src/main.py", "videos/LEEME.md"})

    def test_sin_nombre_valido_no_genera_nada(self):
        error = io.StringIO()
        with mock.patch.object(sys, "argv", ["make_zip.py", "¡¿?"]), \
                contextlib.redirect_stderr(error):
            codigo = self.zip.main()
        self.assertEqual(codigo, 1)
        self.assertIn("vacío", error.getvalue())


class TestDownloadModels(unittest.TestCase):
    def setUp(self):
        self.dm = cargar_script("download_models.py")
        parche = mock.patch.object(self.dm, "download_model")   # Sin red: la descarga se simula
        self.descarga = parche.start()
        self.addCleanup(parche.stop)

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


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
