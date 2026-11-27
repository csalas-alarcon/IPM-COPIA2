"""
tests/test_vision.py  ·  Pruebas de src/vision sin cámara ni descargas.

Vídeo sintético diminuto (0,5 s) con un reloj simulado, geometría de poses,
timestamps del detector y lógica de los modelos. No abre ventanas, no usa la
red y no escribe en models/, results/ ni videos/.

Ejecutar:  .venv/bin/python -m unittest tests.test_vision -v
(Windows:  .venv\\Scripts\\python -m unittest tests.test_vision -v)
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CameraConfig  # noqa: E402
from vision import models  # noqa: E402
from vision.camera import Camera, NoSource, fit_frame, is_video_file, parse_source  # noqa: E402
from vision.detector import PoseDetector  # noqa: E402
from vision.pose import Landmark, PoseFrame, angle_to_vertical, distance, joint_angle, side_landmark  # noqa: E402


# =============================================================================
# Constantes y utilidades
# =============================================================================
FPS = 10                 # Fotogramas por segundo del vídeo de prueba
N_FRAMES = 5             # 5 frames a 10 fps = 0,5 s de vídeo
TAMANO = (32, 24)        # (ancho, alto) del vídeo de prueba: pequeño y rápido


def _escribir_video(ruta: Path) -> None:
    """Crea un vídeo AVI de N_FRAMES fotogramas, cada uno de un gris distinto (0, 40, 80...)."""
    writer = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), FPS, TAMANO)
    for i in range(N_FRAMES):
        writer.write(np.full((TAMANO[1], TAMANO[0], 3), 40 * i, np.uint8))
    writer.release()


class RelojFalso:
    """Sustituye al módulo time dentro de vision.camera.

    Camera reproduce el vídeo a velocidad real con time.monotonic() y
    time.sleep(). Con este reloj el tiempo solo avanza cuando el código duerme
    o cuando el test lo avanza, así que los tests no dependen de la velocidad
    de la máquina y no esperan de verdad.
    """

    def __init__(self):
        self.ahora = 0.0

    def monotonic(self) -> float:
        return self.ahora

    def sleep(self, segundos: float) -> None:
        self.ahora += segundos


# =============================================================================
# Fuentes y frames: parse_source, is_video_file, fit_frame
# =============================================================================
class TestFuentes(unittest.TestCase):
    def test_parse_source_convierte_solo_los_indices_de_camara(self):
        url = "http://192.168.1.20:8080/video"
        casos = [("0", 0), (" 2 ", 2), (1, 1), (url, url), ("videos/saque.mp4", "videos/saque.mp4")]
        for entrada, esperado in casos:
            with self.subTest(entrada=entrada):
                self.assertEqual(parse_source(entrada), esperado)

    def test_is_video_file_solo_acepta_archivos_locales_existentes(self):
        with tempfile.TemporaryDirectory() as tmp:
            existente = Path(tmp) / "clip.mp4"
            existente.write_bytes(b"")
            self.assertTrue(is_video_file(str(existente)))
        self.assertFalse(is_video_file("/no/existe/clip.mp4"))
        self.assertFalse(is_video_file("http://localhost/clip.mp4"))
        self.assertFalse(is_video_file(0))


class TestFitFrame(unittest.TestCase):
    def test_frames_que_no_hace_falta_adaptar_se_dejan_igual(self):
        casos = [
            ("16:9 con la altura pedida", (720, 1280), 720),
            ("4:3, no es lo bastante vertical", (480, 640), 720),
            ("5 % por encima del máximo se tolera (750 <= 756)", (750, 1000), 720),
            ("sin máximo (0)", (1000, 2000), 0),
        ]
        for nombre, (alto, ancho), maximo in casos:
            with self.subTest(nombre):
                frame = np.zeros((alto, ancho, 3), np.uint8)
                self.assertEqual(fit_frame(frame, maximo).shape, (alto, ancho, 3))

    def test_frame_mas_alto_que_el_maximo_se_reduce_manteniendo_la_proporcion(self):
        frame = np.zeros((1000, 2000, 3), np.uint8)
        self.assertEqual(fit_frame(frame, 720).shape, (720, 1440, 3))

    def test_frame_vertical_se_centra_sobre_fondo_16_9(self):
        frame = np.full((960, 480, 3), 255, np.uint8)
        resultado = fit_frame(frame, 720)               # Se reduce a 360x720 y se centra en 1280x720
        self.assertEqual(resultado.shape, (720, 1280, 3))
        self.assertTrue(np.all(resultado[0, 0] == [24, 20, 18]))         # Esquina: color de relleno
        self.assertTrue(np.all(resultado[360, 640] == [255, 255, 255]))  # Centro: el vídeo


# =============================================================================
# Camera con un vídeo sintético y un reloj simulado
# =============================================================================
class TestCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.video = Path(cls._tmp.name) / "video_corto.avi"
        _escribir_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.reloj = RelojFalso()
        parche = patch("vision.camera.time", self.reloj)
        parche.start()
        self.addCleanup(parche.stop)

    def abrir(self, **cfg) -> Camera:
        cam = Camera(CameraConfig(source=str(self.video), **cfg))
        self.addCleanup(cam.release)
        return cam

    def test_video_recien_abierto_esta_en_pausa_en_el_primer_frame(self):
        cam = self.abrir()
        self.assertTrue(cam.is_video)
        self.assertTrue(cam.paused)
        self.assertFalse(cam.ended)
        self.assertEqual(cam.position, 0.0)
        self.assertAlmostEqual(cam.duration, 0.5, places=2)
        self.assertEqual(cam.frame_size, TAMANO)
        self.assertEqual(cam.label, "video_corto.avi")

    def test_leer_en_pausa_repite_el_mismo_frame_sin_avanzar(self):
        cam = self.abrir()
        id_inicial = cam.frame_id
        primero = cam.read()
        segundo = cam.read()
        self.assertTrue(np.array_equal(primero, segundo))
        self.assertEqual(cam.frame_id, id_inicial)
        self.assertEqual(cam.position, 0.0)

    def test_play_avanza_un_frame_por_cada_decima_de_segundo(self):
        cam = self.abrir()
        cam.play()
        frame = cam.read()
        self.assertFalse(cam.paused)
        self.assertAlmostEqual(cam.position, 0.1)
        self.assertAlmostEqual(float(frame.mean()), 40.0, delta=3)     # Frame 1: gris 40
        frame = cam.read()
        self.assertAlmostEqual(cam.position, 0.2)
        self.assertAlmostEqual(float(frame.mean()), 80.0, delta=3)     # Frame 2: gris 80

    def test_pause_congela_el_video_y_toggle_pause_alterna(self):
        cam = self.abrir()
        cam.toggle_pause()
        cam.read()
        self.assertFalse(cam.paused)
        cam.toggle_pause()
        congelada = cam.position
        cam.read()
        self.assertTrue(cam.paused)
        self.assertEqual(cam.position, congelada)

    def test_si_el_equipo_va_retrasado_salta_frames_para_alcanzar_el_reloj(self):
        cam = self.abrir()
        cam.play()
        self.reloj.ahora += 0.35          # Pasan 0,35 s sin leer: toca el frame 3
        cam.read()
        self.assertAlmostEqual(cam.position, 0.3)

    def test_rewind_vuelve_al_principio_en_pausa(self):
        cam = self.abrir()
        cam.play()
        cam.read()
        cam.rewind()
        self.assertEqual(cam.position, 0.0)
        self.assertTrue(cam.paused)
        self.assertFalse(cam.ended)

    def test_start_seconds_empieza_en_ese_segundo_y_fija_el_reloj_del_juego(self):
        cam = self.abrir(start_seconds=0.3)   # 0,3 s a 10 fps = frame 3
        self.assertAlmostEqual(cam.position, 0.3)
        self.assertAlmostEqual(cam.clock(), cam.position)
        self.assertEqual(cam.timestamp_ms(), 300)

    def test_al_llegar_al_final_se_marca_ended_y_se_pausa(self):
        cam = self.abrir()
        cam.play()
        for _ in range(N_FRAMES + 2):
            cam.read()
        self.assertTrue(cam.ended)
        self.assertTrue(cam.paused)
        id_final = cam.frame_id
        cam.read()
        self.assertEqual(cam.frame_id, id_final)
        cam.play()                            # Terminado: play no lo reanuda
        self.assertTrue(cam.paused)

    def test_fuentes_que_no_se_pueden_abrir_lanzan_error_claro(self):
        with tempfile.TemporaryDirectory() as tmp:
            texto = Path(tmp) / "no_es_video.txt"
            texto.write_text("esto no es un vídeo", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "No se pudo abrir el vídeo"):
                Camera(CameraConfig(source=str(texto)))
        with self.assertRaisesRegex(RuntimeError, "No existe el archivo"):
            Camera(CameraConfig(source="/no/existe/clip.mp4"))


class TestCamaraEnVivo(unittest.TestCase):
    """Cámara en vivo con un VideoCapture simulado (no se abre ninguna cámara)."""

    def abrir(self, capturas: list, **cfg) -> Camera:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = capturas
        with patch("vision.camera.cv2.VideoCapture", return_value=cap):
            cam = Camera(CameraConfig(source=0, **cfg))
        self.addCleanup(cam.release)
        return cam

    def test_con_la_camara_en_vivo_la_imagen_sale_espejada(self):
        original = np.zeros((24, 32, 3), np.uint8)
        original[:, :4] = 255                 # Franja blanca a la izquierda
        cam = self.abrir([(True, original)])
        frame = cam.read()
        self.assertEqual(cam.frame_id, 1)
        self.assertTrue(np.all(frame[:, -4:] == 255))   # Ahora está a la derecha
        self.assertTrue(np.all(frame[:, :4] == 0))

    def test_tras_demasiadas_lecturas_fallidas_lanza_error(self):
        cam = self.abrir([(False, None)] * 3, max_failed_reads=3)
        self.assertIsNone(cam.read())
        self.assertIsNone(cam.read())
        with self.assertRaisesRegex(RuntimeError, "ha dejado de enviar imágenes"):
            cam.read()


class TestNoSource(unittest.TestCase):
    def test_fuente_vacia_devuelve_pantalla_oscura_sin_video_ni_timestamp(self):
        fuente = NoSource(width=64, height=48)
        frame = fuente.read()
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(fuente.frame_size, (64, 48))
        self.assertEqual(fuente.frame_id, 1)
        self.assertFalse(fuente.is_video)
        self.assertIsNone(fuente.timestamp_ms())
        self.assertEqual(fuente.label, "Sin cámara")


# =============================================================================
# Geometría: joint_angle, angle_to_vertical, distance, side_landmark
# =============================================================================
class TestGeometria(unittest.TestCase):
    def test_joint_angle_en_2d_y_3d(self):
        origen = np.array([0, 0])
        casos = [
            ("recto", np.array([1, 0]), np.array([0, 1]), 90.0),
            ("línea recta", np.array([-1, 0]), np.array([1, 0]), 180.0),
            ("mismo sentido", np.array([1, 0]), np.array([5, 0]), 0.0),
            ("diagonal", np.array([1, 0]), np.array([1, 1]), 45.0),
        ]
        for nombre, a, c, esperado in casos:
            with self.subTest(nombre):
                self.assertAlmostEqual(joint_angle(a, origen, c), esperado)
        self.assertAlmostEqual(joint_angle(np.array([0, 1, 0]), np.zeros(3), np.array([0, 0, 2])), 90.0)

    def test_joint_angle_no_depende_del_orden_de_los_extremos(self):
        a, b, c = np.array([1, 0]), np.array([0, 0]), np.array([1, 1])
        self.assertAlmostEqual(joint_angle(a, b, c), joint_angle(c, b, a))

    def test_joint_angle_con_punto_en_el_vertice_devuelve_180(self):
        self.assertAlmostEqual(joint_angle(np.array([0, 0]), np.array([0, 0]), np.array([0, 1])), 180.0)

    def test_angle_to_vertical_ignora_el_sentido_y_la_profundidad(self):
        casos = [
            ("hacia abajo", [0, 5], 0.0),
            ("hacia arriba", [0, -5], 0.0),
            ("horizontal", [3, 0], 90.0),
            ("diagonal", [1, 1], 45.0),
            ("con profundidad", [0, 2, 7], 0.0),
            ("vector nulo", [0, 0], 0.0),
        ]
        for nombre, vector, esperado in casos:
            with self.subTest(nombre):
                self.assertAlmostEqual(angle_to_vertical(np.array(vector)), esperado)

    def test_distance_en_2d_y_3d(self):
        self.assertAlmostEqual(distance(np.array([0, 0]), np.array([3, 4])), 5.0)
        self.assertAlmostEqual(distance(np.array([0, 0, 0]), np.array([1, 2, 2])), 3.0)
        self.assertAlmostEqual(distance(np.array([2, 2]), np.array([2, 2])), 0.0)

    def test_side_landmark_da_el_indice_entero_del_lado_pedido(self):
        self.assertEqual(side_landmark("LEFT", "ELBOW"), Landmark.LEFT_ELBOW)
        self.assertEqual(side_landmark("RIGHT", "ANKLE"), 28)
        self.assertIs(type(side_landmark("LEFT", "HIP")), int)
        with self.assertRaises(KeyError):
            side_landmark("CENTER", "ELBOW")


# =============================================================================
# PoseFrame
# =============================================================================
class TestPoseFrame(unittest.TestCase):
    def setUp(self):
        # Hombro (11), codo (13) y muñeca (15) izquierdos. En la imagen están en
        # línea recta (180º); en 3D forman un ángulo recto (90º).
        self.image = np.zeros((33, 2))
        self.image[11], self.image[13], self.image[15] = [0, -1], [0, 0], [0, 1]
        self.world = np.zeros((33, 3))
        self.world[11], self.world[13], self.world[15] = [0, -1, 0], [0, 0, 0], [1, 0, 0]
        self.visibilidad = np.zeros(33)

    def crear(self, world, use_world=True) -> PoseFrame:
        return PoseFrame(self.image, self.visibilidad, world, timestamp_ms=0, use_world=use_world)

    def test_angle_usa_3d_solo_si_hay_world_y_use_world_esta_activo(self):
        self.assertAlmostEqual(self.crear(self.world, use_world=True).angle(11, 13, 15), 90.0)
        self.assertAlmostEqual(self.crear(self.world, use_world=False).angle(11, 13, 15), 180.0)
        self.assertAlmostEqual(self.crear(None, use_world=True).angle(11, 13, 15), 180.0)

    def test_accesores_de_visibilidad_y_punto(self):
        self.visibilidad[11], self.visibilidad[12] = 0.8, 0.4
        self.image[11] = [3, 4]
        pose = self.crear(self.world)
        self.assertAlmostEqual(pose.vis(11), 0.8)
        self.assertAlmostEqual(pose.mean_visibility([11, 12]), 0.6)
        self.assertEqual(pose.mean_visibility([]), 0.0)
        np.testing.assert_array_equal(pose.point(11), [3, 4])


# =============================================================================
# Detector: _next_timestamp (sin cargar MediaPipe)
# =============================================================================
def _detector_sin_mediapipe() -> PoseDetector:
    """PoseDetector sin modelo (se salta __init__): solo sirve para probar _next_timestamp."""
    det = PoseDetector.__new__(PoseDetector)
    det._last_ts = -1
    det._offset = 0
    return det


class TestNextTimestamp(unittest.TestCase):
    def test_en_vivo_el_timestamp_crece_aunque_el_reloj_no_avance(self):
        det = _detector_sin_mediapipe()
        with patch("vision.detector.time.monotonic", return_value=5.0):
            primero = det._next_timestamp(None)
            segundo = det._next_timestamp(None)
        self.assertEqual(primero, 5000)
        self.assertGreater(segundo, primero)

    def test_con_video_usa_su_timestamp_y_si_se_repite_lo_incrementa(self):
        det = _detector_sin_mediapipe()
        self.assertEqual([det._next_timestamp(t) for t in (0, 33, 66)], [0, 33, 66])
        self.assertEqual(det._next_timestamp(66), 67)

    def test_al_rebobinar_el_video_los_timestamps_siguen_creciendo(self):
        det = _detector_sin_mediapipe()
        sucesion = [det._next_timestamp(t) for t in (0, 100, 500, 0, 33, 33, 200)]
        for anterior, actual in zip(sucesion, sucesion[1:]):
            self.assertGreater(actual, anterior)


# =============================================================================
# Modelos: _is_valid, ensure_model, download_model
# =============================================================================
class TestModelos(unittest.TestCase):
    def test_un_modelo_es_valido_si_existe_y_pesa_al_menos_1_mega(self):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            pequeno = carpeta / "pequeno.task"
            pequeno.write_bytes(b"x" * 999_999)
            grande = carpeta / "grande.task"
            grande.write_bytes(b"x" * 1_000_000)
            self.assertFalse(models._is_valid(carpeta / "falta.task"))
            self.assertFalse(models._is_valid(pequeno))
            self.assertTrue(models._is_valid(grande))

    def test_ensure_model_devuelve_el_modelo_si_ya_existe(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pose_landmarker_full.task"
            ruta.write_bytes(b"x" * 1_000_000)
            with patch("vision.models.model_path", return_value=ruta), \
                    patch("vision.models.download_model") as descargar:
                self.assertEqual(models.ensure_model("full", download=False), ruta)
                descargar.assert_not_called()

    def test_ensure_model_sin_descarga_avisa_de_como_conseguirlo(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pose_landmarker_full.task"
            with patch("vision.models.model_path", return_value=ruta), \
                    patch("vision.models.download_model") as descargar:
                with self.assertRaisesRegex(FileNotFoundError, "download_models.py"):
                    models.ensure_model("full", download=False)
                descargar.assert_not_called()

    def test_ensure_model_con_descarga_delega_en_download_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pose_landmarker_full.task"
            descargada = Path(tmp) / "descargada.task"
            with patch("vision.models.model_path", return_value=ruta), \
                    patch("vision.models.download_model", return_value=descargada) as descargar:
                self.assertEqual(models.ensure_model("full", download=True), descargada)
                descargar.assert_called_once_with("full")

    def test_variante_desconocida_falla_antes_de_tocar_la_red(self):
        with patch("vision.models.requests.get") as peticion:
            with self.assertRaisesRegex(ValueError, "Modelo desconocido"):
                models.download_model("ultra")
            peticion.assert_not_called()


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
