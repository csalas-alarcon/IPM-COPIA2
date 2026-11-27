"""
tests/test_interfaz.py  ·  Pruebas de la interfaz sin ventana.

Lienzo, botones, gráficas, pantallas, sonido, grabación y lista de vídeos.
Todo se dibuja en frames en memoria: no se abre ninguna ventana ni tkinter.

Ejecutar:  .venv/bin/python -m unittest tests.test_interfaz -v
(Windows:  .venv\\Scripts\\python -m unittest tests.test_interfaz -v)
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import wave
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for ruta in (SRC_DIR, TESTS_DIR):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from exercises.catalog import build_exercises  # noqa: E402
from game.history import HistoryEntry  # noqa: E402
from ui.canvas import ACCENT, GREEN, RED, WHITE, Canvas, TextRenderer, lerp_color, score_color  # noqa: E402
from ui.dialogs import ask_video_file, list_videos  # noqa: E402
from ui.recorder import Recorder  # noqa: E402
from ui.screens import (Feedback, MenuInfo, PickerInfo, _mmss, _nice_ceil, draw_feedback,  # noqa: E402
                        draw_history, draw_menu, draw_toast, draw_video_picker, draw_video_status)
from ui.sound import SAMPLE_RATE, Sound, synthesize, to_wav_bytes  # noqa: E402
from ui.widgets import button, checkbox, draw_line_chart, draw_rep_chart  # noqa: E402
from utilidades import RelojFalso  # noqa: E402


# =============================================================================
# Utilidades
# =============================================================================
def nuevo_lienzo(alto=720, ancho=1280):
    """Lienzo negro de un frame del tamaño dado."""
    return Canvas(np.zeros((alto, ancho, 3), np.uint8), TextRenderer())


def acciones(c):
    """Acciones de los botones registrados en el lienzo."""
    return [accion for accion, _ in c.buttons]


def menu(selected=("pushup",), source_kind="camera"):
    """MenuInfo de prueba con dos ejercicios."""
    label = "Cámara 0" if source_kind == "camera" else "Sin cámara"
    return MenuInfo(exercises=build_exercises(["pushup", "pullup"]), selected=set(selected),
                    source_label=label, source_kind=source_kind, calibrate=True,
                    duration=60, target=10)


def entrada(i, exercises=("Flexión",)):
    """Partida guardada de prueba; i = días desde el 1/11/2026."""
    return HistoryEntry(Path("x"), datetime(2026, 11, 1) + timedelta(days=i), 300 + 50 * i,
                        8, 70 + i, list(exercises))


# =============================================================================
# Canvas: colores y escala
# =============================================================================
class TestColores(unittest.TestCase):
    def test_lerp_interpola_y_recorta(self):
        self.assertEqual(lerp_color((0, 0, 0), (100, 200, 250), 0.5), (50, 100, 125))
        self.assertEqual(lerp_color((0, 0, 0), (100, 200, 250), -1), (0, 0, 0))
        self.assertEqual(lerp_color((0, 0, 0), (100, 200, 250), 2), (100, 200, 250))

    def test_color_segun_puntuacion(self):
        self.assertEqual(score_color(80), GREEN)
        self.assertEqual(score_color(79.9), ACCENT)
        self.assertEqual(score_color(60), ACCENT)
        self.assertEqual(score_color(59.9), RED)


class TestEscala(unittest.TestCase):
    def test_frame_720_es_1_a_1(self):
        c = nuevo_lienzo(720, 1280)
        self.assertEqual(c.s, 1.0)
        self.assertEqual(c.W, 1280)
        self.assertEqual(c.pt(100, 200), (100, 200))

    def test_frame_480_escala_dos_tercios(self):
        c = nuevo_lienzo(480, 640)
        self.assertAlmostEqual(c.s, 2 / 3)
        self.assertAlmostEqual(c.W, 960)              # 4:3 sigue teniendo ancho de diseño 960
        self.assertEqual(c.pt(720, 720), (480, 480))

    def test_frame_1080_escala_uno_y_medio(self):
        c = nuevo_lienzo(1080, 1920)
        self.assertAlmostEqual(c.W, 1280)             # 16:9 siempre mide 1280 de diseño
        self.assertEqual(c.px(360), 540)

    def test_rectangulo_en_unidades_de_diseno(self):
        c = nuevo_lienzo(480, 640)
        c.rect(0, 0, 360, 360, WHITE)                 # En píxeles: 0..240
        self.assertEqual(tuple(c.frame[100, 100]), WHITE)
        self.assertEqual(tuple(c.frame[400, 400]), (0, 0, 0))

    def test_dim_oscurece(self):
        c = nuevo_lienzo()
        c.frame[:] = 200
        c.dim(0.5)
        self.assertEqual(int(c.frame[0, 0, 0]), 100)


# =============================================================================
# Canvas: formas, ratón y texto
# =============================================================================
class TestFormas(unittest.TestCase):
    def test_panel_fuera_de_la_pantalla_no_cambia_nada(self):
        c = nuevo_lienzo()
        c.frame[:] = 50
        c.panel(2000, 2000, 2100, 2100)
        self.assertTrue(np.all(c.frame == 50))

    def test_panel_que_sale_de_la_pantalla_se_recorta(self):
        c = nuevo_lienzo()
        c.frame[:] = 50
        c.panel(-100, -100, 100, 100)                 # Solo se pinta la parte visible
        self.assertNotEqual(int(c.frame[10, 10, 0]), 50)
        c.panel(-500, -500, 5000, 5000)               # Más grande que el frame
        self.assertEqual(c.frame.shape, (720, 1280, 3))

    def test_polilinea_discontinua_deja_huecos(self):
        c = nuevo_lienzo()
        c.polyline([(0, 10), (100, 10), (200, 10), (300, 10)], WHITE, 2, dashed=True)
        self.assertGreater(int(c.frame[10, 50].sum()), 0)     # Primer trazo
        self.assertEqual(int(c.frame[10, 150].sum()), 0)      # Hueco entre trazos
        self.assertGreater(int(c.frame[10, 250].sum()), 0)    # Segundo trazo

    def test_polilinea_con_un_solo_punto_no_falla(self):
        c = nuevo_lienzo()
        c.polyline([(5, 5)], WHITE)
        self.assertEqual(int(c.frame.sum()), 0)

    def test_hovered_detecta_el_raton(self):
        c = nuevo_lienzo()
        self.assertFalse(c.hovered(100, 100, 200, 200))       # Sin ratón
        c.mouse = (150, 150)
        self.assertTrue(c.hovered(100, 100, 200, 200))
        c.mouse = (300, 300)
        self.assertFalse(c.hovered(100, 100, 200, 200))


class TestTexto(unittest.TestCase):
    def test_texto_se_encola_y_se_pinta_al_terminar(self):
        c = nuevo_lienzo()
        c.text("Hola", 100, 100, 40, WHITE)
        self.assertEqual(int(c.frame.sum()), 0)               # Aún no se ha pintado
        self.assertIs(c.finish(), c.frame)
        self.assertGreater(int(c.frame.sum()), 0)
        self.assertEqual(c._texts, [])                        # La cola queda vacía

    def test_fuente_se_cachea(self):
        r = TextRenderer()
        self.assertIs(r.font(20), r.font(20))

    def test_tamanos_pequenos_no_fallan(self):
        r = TextRenderer()
        for size in (0, 1, 3):
            self.assertIsNotNone(r.font(size))


# =============================================================================
# Widgets: botones, casillas y gráficas
# =============================================================================
class TestBotones(unittest.TestCase):
    def test_boton_registra_accion_y_rectangulo_en_pixeles(self):
        c = nuevo_lienzo(480, 640)                            # Escala 2/3
        button(c, "start", "Empezar", 100, 100, 200, 150)
        self.assertEqual(c.buttons, [("start", (67, 67, 133, 100))])

    def test_boton_deshabilitado_no_se_registra(self):
        c = nuevo_lienzo()
        button(c, "start", "Empezar", 100, 100, 200, 150, enabled=False)
        self.assertEqual(c.buttons, [])

    def test_casilla_registra_toda_la_fila(self):
        c = nuevo_lienzo()
        checkbox(c, "toggle:pushup", "Flexión", True, 100, 100, 500, 144)
        self.assertEqual(c.buttons, [("toggle:pushup", (100, 100, 500, 144))])

    def test_casilla_marcada_se_ve_distinta(self):
        marcada, vacia = nuevo_lienzo(), nuevo_lienzo()
        checkbox(marcada, "toggle:pushup", "Flexión", True, 100, 100, 500, 144)
        checkbox(vacia, "toggle:pushup", "Flexión", False, 100, 100, 500, 144)
        self.assertFalse(np.array_equal(marcada.frame, vacia.frame))


class TestGraficas(unittest.TestCase):
    def test_grafica_de_repeticion_con_un_punto_no_dibuja_curva(self):
        vacia, un_punto = nuevo_lienzo(), nuevo_lienzo()
        draw_rep_chart(vacia, 20, 20, 600, 400, [], [])
        draw_rep_chart(un_punto, 20, 20, 600, 400, [0.5], [0.5])
        self.assertTrue(np.array_equal(vacia.frame, un_punto.frame))   # Una curva necesita 2 puntos

    def test_grafica_de_repeticion_con_datos_cambia_pixeles(self):
        vacia, normal = nuevo_lienzo(), nuevo_lienzo()
        draw_rep_chart(vacia, 20, 20, 600, 400, [], [])
        draw_rep_chart(normal, 20, 20, 600, 400, [0, 0.5, 1, 0.5, 0], [0, 0.4, 1, 0.4, 0])
        self.assertFalse(np.array_equal(vacia.frame, normal.frame))

    def test_grafica_de_lineas_con_un_punto_dibuja_el_marcador(self):
        vacia, un_punto = nuevo_lienzo(), nuevo_lienzo()
        draw_line_chart(vacia, 20, 20, 600, 400, [], 100)
        draw_line_chart(un_punto, 20, 20, 600, 400, [("Puntos", [50], GREEN)], 100)
        self.assertFalse(np.array_equal(vacia.frame, un_punto.frame))

    def test_grafica_de_lineas_con_series_cambia_pixeles(self):
        vacia, normal = nuevo_lienzo(), nuevo_lienzo()
        draw_line_chart(vacia, 20, 20, 600, 400, [], 100)
        draw_line_chart(normal, 20, 20, 600, 400, [("Puntos", [10, 50, 80], GREEN)], 100,
                        "Título", ["1/11", "2/11", "3/11"])
        self.assertFalse(np.array_equal(vacia.frame, normal.frame))


# =============================================================================
# Pantallas: formatos y elementos comunes
# =============================================================================
class TestFormatos(unittest.TestCase):
    def test_mmss(self):
        self.assertEqual(_mmss(0), "0:00")
        self.assertEqual(_mmss(59.9), "0:59")
        self.assertEqual(_mmss(65), "1:05")
        self.assertEqual(_mmss(600), "10:00")
        self.assertEqual(_mmss(-3), "0:00")

    def test_nice_ceil_redondea_a_1_2_5(self):
        self.assertEqual(_nice_ceil(0), 1.0)
        self.assertAlmostEqual(_nice_ceil(3), 5.0)
        self.assertAlmostEqual(_nice_ceil(5), 5.0)
        self.assertAlmostEqual(_nice_ceil(7), 10.0)
        self.assertAlmostEqual(_nice_ceil(0.3), 0.5)
        self.assertAlmostEqual(_nice_ceil(120), 200.0)


class TestIndicadorVideo(unittest.TestCase):
    def test_boton_pausa_registrado_mientras_el_video_sigue(self):
        c = nuevo_lienzo()
        draw_video_status(c, 10, 60, paused=False, ended=False)
        self.assertEqual(acciones(c), ["pause"])
        self.assertEqual(c.buttons[0][1][2], c.W - 14)        # Ocupa el indicador de la esquina

    def test_sin_boton_pausa_cuando_el_video_acaba(self):
        c = nuevo_lienzo()
        draw_video_status(c, 60, 60, paused=False, ended=True)
        self.assertEqual(c.buttons, [])


class TestAvisos(unittest.TestCase):
    def test_toast_se_dibuja(self):
        c = nuevo_lienzo()
        draw_toast(c, "No hay cámara")
        c.finish()
        self.assertGreater(int(c.frame.sum()), 0)

    def test_feedback_se_dibuja_antes_de_caducar(self):
        c = nuevo_lienzo()
        draw_feedback(c, Feedback(["¡Bien!"], GREEN, until=5.0), now=1.0)
        c.finish()
        self.assertGreater(int(c.frame.sum()), 0)

    def test_feedback_caducado_no_se_dibuja(self):
        c = nuevo_lienzo()
        draw_feedback(c, Feedback(["¡Bien!"], GREEN, until=5.0), now=6.0)
        c.finish()
        self.assertEqual(int(c.frame.sum()), 0)

    def test_feedback_vacio_no_falla(self):
        c = nuevo_lienzo()
        draw_feedback(c, None, now=1.0)
        self.assertEqual(int(c.frame.sum()), 0)


# =============================================================================
# Pantallas: menú, selector de vídeo e historial
# =============================================================================
class TestMenu(unittest.TestCase):
    def test_empezar_solo_con_algun_ejercicio_y_una_fuente(self):
        casos = (("sin ejercicios", menu(selected=()), False),
                 ("sin fuente", menu(source_kind="none"), False),
                 ("con ejercicio y fuente", menu(), True))
        for nombre, info, activo in casos:
            with self.subTest(nombre):
                c = nuevo_lienzo()
                draw_menu(c, info)
                self.assertEqual("start" in acciones(c), activo)

    def test_menu_tiene_los_botones_basicos(self):
        c = nuevo_lienzo()
        draw_menu(c, menu())
        for accion in ("camera", "pick_video", "history", "quit"):
            self.assertIn(accion, acciones(c))


class TestSelectorDeVideo(unittest.TestCase):
    def test_un_boton_por_video(self):
        c = nuevo_lienzo()
        videos = [Path(f"video_{i}.mp4") for i in range(3)]
        draw_video_picker(c, PickerInfo(videos=videos, can_browse=True))
        self.assertEqual([a for a in acciones(c) if a.startswith("video:")],
                         ["video:0", "video:1", "video:2"])

    def test_como_maximo_ocho_videos(self):
        c = nuevo_lienzo()
        videos = [Path(f"video_{i}.mp4") for i in range(10)]
        draw_video_picker(c, PickerInfo(videos=videos, can_browse=True))
        self.assertEqual(len([a for a in acciones(c) if a.startswith("video:")]), 8)

    def test_sin_videos_no_tiene_botones_de_video(self):
        c = nuevo_lienzo()
        draw_video_picker(c, PickerInfo())
        self.assertEqual([a for a in acciones(c) if a.startswith("video:")], [])
        self.assertIn("back", acciones(c))

    def test_buscar_en_el_ordenador_solo_con_tkinter(self):
        for hay_tkinter in (False, True):
            with self.subTest(tkinter=hay_tkinter):
                c = nuevo_lienzo()
                draw_video_picker(c, PickerInfo(can_browse=hay_tkinter))
                self.assertEqual("browse" in acciones(c), hay_tkinter)


class TestHistorial(unittest.TestCase):
    def test_historial_vacio_y_con_datos_se_dibujan_distinto(self):
        vacio, con_datos = nuevo_lienzo(), nuevo_lienzo()
        draw_history(vacio, [])
        vacio.finish()
        draw_history(con_datos, [entrada(i) for i in range(5)])
        con_datos.finish()
        self.assertFalse(np.array_equal(vacio.frame, con_datos.frame))
        self.assertIn("back", acciones(vacio))
        self.assertIn("back", acciones(con_datos))

    def test_entrada_sin_ejercicios_no_falla(self):
        c = nuevo_lienzo()
        draw_history(c, [entrada(0, exercises=())])
        c.finish()


# =============================================================================
# Sonido (sin dispositivo de audio: backend simulado)
# =============================================================================
class TestSintesis(unittest.TestCase):
    def test_duracion_es_la_suma_de_las_notas(self):
        senal = synthesize([(440, 0.1), (660, 0.05)])
        self.assertEqual(len(senal), int(SAMPLE_RATE * 0.1) + int(SAMPLE_RATE * 0.05))

    def test_frecuencia_cero_es_silencio(self):
        self.assertTrue(np.all(synthesize([(0, 0.1)]) == 0))

    def test_la_amplitud_no_pasa_del_volumen(self):
        senal = synthesize([(440, 0.1)], volume=0.35)
        self.assertLessEqual(float(np.abs(senal).max()), 0.35)
        self.assertGreater(float(np.abs(senal).max()), 0.3)

    def test_wav_tiene_cabecera_y_numero_de_muestras(self):
        senal = synthesize([(440, 0.1)])
        datos = to_wav_bytes(senal)
        self.assertTrue(datos.startswith(b"RIFF"))
        self.assertEqual(datos[8:12], b"WAVE")
        with wave.open(io.BytesIO(datos)) as wav:
            self.assertEqual(wav.getnframes(), len(senal))
            self.assertEqual(wav.getframerate(), SAMPLE_RATE)
            self.assertEqual(wav.getnchannels(), 1)


class TestSonido(unittest.TestCase):
    def setUp(self):
        # Sound(enabled=False) no abre ningún dispositivo; se le pone un backend simulado
        self.sonido = Sound(enabled=False)
        self.sonido.backend = "sounddevice"
        self.sonido._sd = mock.Mock()

    def test_sin_backend_no_hace_nada(self):
        silencioso = Sound(enabled=False)
        self.assertFalse(silencioso.enabled)
        silencioso.play("rep")

    def test_nombre_desconocido_no_suena(self):
        self.sonido.play("no_existe")
        self.sonido._sd.play.assert_not_called()

    def test_reproduce_con_el_backend(self):
        self.sonido.play("rep")
        self.sonido._sd.play.assert_called_once()

    def test_error_del_backend_desactiva_el_sonido(self):
        self.sonido._sd.play.side_effect = OSError("sin dispositivo")
        with contextlib.redirect_stdout(io.StringIO()):
            self.sonido.play("rep")
        self.assertFalse(self.sonido.enabled)
        self.assertIsNone(self.sonido.backend)


# =============================================================================
# Grabación de la ventana
# =============================================================================
class TestGrabadora(unittest.TestCase):
    def setUp(self):
        self.carpeta = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ruta = self.carpeta / "rec.mp4"
        self.reloj = RelojFalso()
        self.enterContext(mock.patch("ui.recorder.time", self.reloj))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))   # La grabadora avisa por consola

    def leer_video(self):
        """(ancho, alto, número de frames) del vídeo grabado."""
        captura = cv2.VideoCapture(str(self.ruta))
        ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = 0
        while captura.read()[0]:
            frames += 1
        captura.release()
        return ancho, alto, frames

    def test_se_activa_al_escribir_y_se_puede_cerrar_dos_veces(self):
        grabadora = Recorder(self.ruta, fps=10)
        self.assertFalse(grabadora.active)
        grabadora.write(np.zeros((240, 320, 3), np.uint8))
        self.assertTrue(grabadora.active)
        grabadora.close()
        self.assertFalse(grabadora.active)
        grabadora.close()

    def test_repite_frames_segun_el_reloj(self):
        grabadora = Recorder(self.ruta, fps=10)
        frame = np.full((240, 320, 3), 120, np.uint8)
        grabadora.write(frame)                                # t = 0 s: 1 frame
        self.reloj.avanzar(0.5)
        grabadora.write(frame)                                # t = 0,5 s: 6 frames en total
        grabadora.close()
        self.assertEqual(self.leer_video()[2], 6)
        self.assertAlmostEqual(grabadora.elapsed, 0.6)

    def test_el_tamano_se_fija_con_el_primer_frame_y_se_hace_par(self):
        grabadora = Recorder(self.ruta, fps=10)
        grabadora.write(np.zeros((241, 321, 3), np.uint8))    # Ancho y alto impares: se ajustan
        self.reloj.avanzar(0.2)
        grabadora.write(np.zeros((480, 640, 3), np.uint8))    # Otro tamaño: se reescala
        grabadora.close()
        self.assertEqual(self.leer_video()[:2], (320, 240))


# =============================================================================
# Lista de vídeos y explorador de archivos
# =============================================================================
class TestListaDeVideos(unittest.TestCase):
    def setUp(self):
        self.carpeta = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_solo_videos_sin_repetir_y_ordenados(self):
        (self.carpeta / "b.mp4").write_bytes(b"")
        (self.carpeta / "A.MOV").write_bytes(b"")             # Extensión en mayúsculas
        (self.carpeta / "notas.txt").write_text("x")
        encontrados = list_videos([self.carpeta, self.carpeta])    # La misma carpeta dos veces
        self.assertEqual([p.name for p in encontrados], ["A.MOV", "b.mp4"])

    def test_carpeta_inexistente_se_ignora(self):
        self.assertEqual(list_videos([self.carpeta / "no_existe"]), [])

    def test_limite(self):
        for nombre in ("a.mp4", "b.mp4", "c.mp4"):
            (self.carpeta / nombre).write_bytes(b"")
        self.assertEqual(len(list_videos([self.carpeta], limit=2)), 2)


class TestExplorador(unittest.TestCase):
    def test_sin_tkinter_devuelve_none_sin_abrir_nada(self):
        with mock.patch("ui.dialogs.tk_available", return_value=False), \
                mock.patch("ui.dialogs.subprocess.run") as ejecutar:
            self.assertIsNone(ask_video_file(Path(".")))
        ejecutar.assert_not_called()

    def test_devuelve_la_ruta_elegida_o_none_si_se_cancela(self):
        casos = (("elegido\n", "elegido"), ("", None))        # El proceso escribe la ruta en su salida
        for salida, esperada in casos:
            with self.subTest(salida=salida), \
                    mock.patch("ui.dialogs.tk_available", return_value=True), \
                    mock.patch("ui.dialogs.subprocess.run",
                               return_value=mock.Mock(stdout=salida)):
                self.assertEqual(ask_video_file(Path(".")), esperada)


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
