"""
tests/test_vision.py  ·  Pruebas de src/vision sin cámara ni descargas.

Vídeos sintéticos diminutos (0,5 s), geometría de poses, timestamps del
detector y lógica de los modelos. No abre ventanas, no usa la red y no
escribe en models/, results/ ni videos/.

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
from unittest.mock import patch

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
    """Crea un vídeo AVI sintético de N_FRAMES fotogramas, cada uno de un gris distinto."""
    writer = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), FPS, TAMANO)
    for i in range(N_FRAMES):
        writer.write(np.full((TAMANO[1], TAMANO[0], 3), 40 * i, np.uint8))
    writer.release()


# =============================================================================
# Fuentes: parse_source, is_video_file
# =============================================================================
class TestParseSource(unittest.TestCase):
    def test_numero_en_texto_pasa_a_entero(self):
        self.assertEqual(parse_source("0"), 0)

    def test_numero_con_espacios_pasa_a_entero(self):
        self.assertEqual(parse_source(" 2 "), 2)

    def test_entero_se_deja_igual(self):
        self.assertEqual(parse_source(1), 1)

    def test_url_se_deja_igual(self):
        url = "http://192.168.1.20:8080/video"
        self.assertEqual(parse_source(url), url)

    def test_ruta_de_archivo_se_deja_igual(self):
        self.assertEqual(parse_source("videos/saque.mp4"), "videos/saque.mp4")


class TestIsVideoFile(unittest.TestCase):
    def test_archivo_existente_es_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "clip.mp4"
            ruta.write_bytes(b"")
            self.assertTrue(is_video_file(str(ruta)))

    def test_archivo_inexistente_no_es_video(self):
        self.assertFalse(is_video_file("/no/existe/clip.mp4"))

    def test_url_no_es_archivo_aunque_termine_en_mp4(self):
        self.assertFalse(is_video_file("http://localhost/clip.mp4"))

    def test_indice_de_webcam_no_es_video(self):
        self.assertFalse(is_video_file(0))


# =============================================================================
# Frames: fit_frame
# =============================================================================
class TestFitFrame(unittest.TestCase):
    def test_frame_16_9_pequeno_no_se_toca(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(fit_frame(frame, 720).shape, (720, 1280, 3))

    def test_frame_mas_alto_que_el_maximo_se_reduce(self):
        frame = np.zeros((1000, 2000, 3), np.uint8)
        self.assertEqual(fit_frame(frame, 720).shape, (720, 1440, 3))

    def test_sin_maximo_no_se_reduce(self):
        frame = np.zeros((1000, 2000, 3), np.uint8)
        self.assertEqual(fit_frame(frame, 0).shape, (1000, 2000, 3))

    def test_margen_del_5_por_ciento_no_reduce(self):
        # 750 px no supera 720 * 1,05 = 756 px, así que no se reduce
        frame = np.zeros((750, 1000, 3), np.uint8)
        self.assertEqual(fit_frame(frame, 720).shape, (750, 1000, 3))

    def test_vertical_se_centra_en_fondo_16_9(self):
        frame = np.full((960, 480, 3), 255, np.uint8)   # Vertical: 480 de ancho, 960 de alto
        resultado = fit_frame(frame, 720)               # Se reduce a 360x720 y se centra en 1280x720
        self.assertEqual(resultado.shape, (720, 1280, 3))
        self.assertTrue(np.all(resultado[0, 0] == [24, 20, 18]))      # Esquina: color de relleno
        self.assertTrue(np.all(resultado[360, 640] == [255, 255, 255]))  # Centro: el vídeo

    def test_4_3_no_se_toca(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(fit_frame(frame, 720).shape, (480, 640, 3))


# =============================================================================
# Camera con un vídeo sintético
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

    def _abrir(self, **cfg) -> Camera:
        cam = Camera(CameraConfig(source=str(self.video), **cfg))
        self.addCleanup(cam.release)
        return cam

    def test_video_empieza_en_pausa(self):
        cam = self._abrir()
        self.assertTrue(cam.paused)
        self.assertFalse(cam.ended)
        self.assertEqual(cam.position, 0.0)

    def test_leer_en_pausa_repite_el_mismo_frame(self):
        cam = self._abrir()
        antes = cam.frame_id
        primero = cam.read()
        segundo = cam.read()
        self.assertEqual(cam.frame_id, antes)
        self.assertTrue(np.array_equal(primero, segundo))
        self.assertEqual(cam.position, 0.0)

    def test_play_avanza_el_video(self):
        cam = self._abrir()
        cam.play()
        cam.read()
        self.assertFalse(cam.paused)
        self.assertGreater(cam.position, 0.0)

    def test_pause_detiene_el_avance(self):
        cam = self._abrir()
        cam.play()
        cam.read()
        cam.pause()
        congelada = cam.position
        cam.read()
        self.assertTrue(cam.paused)
        self.assertEqual(cam.position, congelada)

    def test_toggle_pause_alterna(self):
        cam = self._abrir()
        cam.toggle_pause()
        self.assertFalse(cam.paused)
        cam.toggle_pause()
        self.assertTrue(cam.paused)

    def test_rewind_vuelve_al_principio_en_pausa(self):
        cam = self._abrir()
        cam.play()
        cam.read()
        cam.pause()
        cam.rewind()
        self.assertEqual(cam.position, 0.0)
        self.assertTrue(cam.paused)
        self.assertFalse(cam.ended)

    def test_start_seconds_empieza_en_ese_segundo(self):
        cam = self._abrir(start_seconds=0.3)   # 0,3 s a 10 fps = frame 3
        self.assertAlmostEqual(cam.position, 0.3, places=2)

    def test_duracion_del_video(self):
        cam = self._abrir()
        self.assertAlmostEqual(cam.duration, 0.5, places=2)

    def test_tamano_del_frame(self):
        cam = self._abrir()
        self.assertEqual(cam.frame_size, TAMANO)

    def test_etiqueta_es_el_nombre_del_archivo(self):
        cam = self._abrir()
        self.assertEqual(cam.label, "video_corto.avi")

    def test_timestamp_ms_es_la_posicion_en_milisegundos(self):
        cam = self._abrir(start_seconds=0.3)
        self.assertEqual(cam.timestamp_ms(), 300)

    def test_clock_de_video_es_la_posicion(self):
        cam = self._abrir(start_seconds=0.3)
        self.assertAlmostEqual(cam.clock(), cam.position)

    def test_fin_del_video_marca_ended_y_pausa(self):
        cam = self._abrir()
        cam.play()
        for _ in range(100):                 # Límite de seguridad: el vídeo dura 0,5 s
            cam.read()
            if cam.ended:
                break
        self.assertTrue(cam.ended)
        self.assertTrue(cam.paused)
        antes = cam.frame_id
        cam.read()
        self.assertEqual(cam.frame_id, antes)

    def test_archivo_que_no_es_video_lanza_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            texto = Path(tmp) / "no_es_video.txt"
            texto.write_text("esto no es un vídeo", encoding="utf-8")
            with self.assertRaises(RuntimeError) as contexto:
                Camera(CameraConfig(source=str(texto)))
            self.assertIn("No se pudo abrir el vídeo", str(contexto.exception))

    def test_archivo_inexistente_lanza_error(self):
        with self.assertRaises(RuntimeError) as contexto:
            Camera(CameraConfig(source="/no/existe/clip.mp4"))
        self.assertIn("No existe el archivo", str(contexto.exception))


# =============================================================================
# Fuente vacía: NoSource
# =============================================================================
class TestNoSource(unittest.TestCase):
    def test_devuelve_pantalla_oscura_del_tamano_pedido(self):
        fuente = NoSource(width=64, height=48)
        frame = fuente.read()
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(fuente.frame_size, (64, 48))

    def test_read_cuenta_frames(self):
        fuente = NoSource(width=64, height=48)
        antes = fuente.frame_id
        fuente.read()
        self.assertEqual(fuente.frame_id, antes + 1)

    def test_no_es_video_y_no_tiene_timestamp(self):
        fuente = NoSource(width=64, height=48)
        self.assertFalse(fuente.is_video)
        self.assertIsNone(fuente.timestamp_ms())
        self.assertEqual(fuente.label, "Sin cámara")


# =============================================================================
# Geometría: joint_angle, angle_to_vertical, distance, side_landmark
# =============================================================================
class TestJointAngle(unittest.TestCase):
    def test_angulo_recto(self):
        angulo = joint_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1]))
        self.assertAlmostEqual(angulo, 90.0)

    def test_linea_recta_es_180(self):
        angulo = joint_angle(np.array([-1, 0]), np.array([0, 0]), np.array([1, 0]))
        self.assertAlmostEqual(angulo, 180.0)

    def test_mismo_sentido_es_0(self):
        angulo = joint_angle(np.array([1, 0]), np.array([0, 0]), np.array([5, 0]))
        self.assertAlmostEqual(angulo, 0.0)

    def test_angulo_de_45_grados(self):
        angulo = joint_angle(np.array([1, 0]), np.array([0, 0]), np.array([1, 1]))
        self.assertAlmostEqual(angulo, 45.0)

    def test_funciona_en_3d(self):
        angulo = joint_angle(np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([0, 0, 2]))
        self.assertAlmostEqual(angulo, 90.0)

    def test_es_simetrico_en_los_extremos(self):
        a, b, c = np.array([1, 0]), np.array([0, 0]), np.array([1, 1])
        self.assertAlmostEqual(joint_angle(a, b, c), joint_angle(c, b, a))

    def test_vector_degenerado_devuelve_180(self):
        # Si un punto coincide con el vértice no hay ángulo: se devuelve 180
        angulo = joint_angle(np.array([0, 0]), np.array([0, 0]), np.array([0, 1]))
        self.assertAlmostEqual(angulo, 180.0)


class TestAngleToVertical(unittest.TestCase):
    def test_vertical_es_0(self):
        self.assertAlmostEqual(angle_to_vertical(np.array([0, 5])), 0.0)

    def test_vertical_hacia_abajo_tambien_es_0(self):
        self.assertAlmostEqual(angle_to_vertical(np.array([0, -5])), 0.0)

    def test_horizontal_es_90(self):
        self.assertAlmostEqual(angle_to_vertical(np.array([3, 0])), 90.0)

    def test_diagonal_es_45(self):
        self.assertAlmostEqual(angle_to_vertical(np.array([1, 1])), 45.0)

    def test_ignora_la_profundidad(self):
        self.assertAlmostEqual(angle_to_vertical(np.array([0, 2, 7])), 0.0)

    def test_vector_nulo_es_0(self):
        self.assertAlmostEqual(angle_to_vertical(np.array([0, 0])), 0.0)


class TestDistance(unittest.TestCase):
    def test_distancia_en_2d(self):
        self.assertAlmostEqual(distance(np.array([0, 0]), np.array([3, 4])), 5.0)

    def test_distancia_en_3d(self):
        self.assertAlmostEqual(distance(np.array([0, 0, 0]), np.array([1, 2, 2])), 3.0)

    def test_mismo_punto_es_0(self):
        self.assertAlmostEqual(distance(np.array([2, 2]), np.array([2, 2])), 0.0)


class TestSideLandmark(unittest.TestCase):
    def test_codo_izquierdo(self):
        self.assertEqual(side_landmark("LEFT", "ELBOW"), 13)
        self.assertEqual(side_landmark("LEFT", "ELBOW"), Landmark.LEFT_ELBOW)

    def test_tobillo_derecho(self):
        self.assertEqual(side_landmark("RIGHT", "ANKLE"), 28)

    def test_devuelve_un_entero(self):
        self.assertIs(type(side_landmark("LEFT", "HIP")), int)

    def test_lado_desconocido_lanza_error(self):
        with self.assertRaises(KeyError):
            side_landmark("CENTER", "ELBOW")


# =============================================================================
# PoseFrame
# =============================================================================
def _pose(image: np.ndarray, world: np.ndarray | None = None,
          visibility: np.ndarray | None = None, use_world: bool = True) -> PoseFrame:
    """PoseFrame de prueba. Por defecto visibilidad 1 en todos los puntos."""
    vis = np.ones(33) if visibility is None else visibility
    return PoseFrame(image, vis, world, timestamp_ms=0, use_world=use_world)


class TestPoseFrame(unittest.TestCase):
    def setUp(self):
        # Hombro (11), codo (13) y muñeca (15) izquierdos. En la imagen están en
        # línea recta (180º); en 3D forman un ángulo recto (90º).
        self.image = np.zeros((33, 2))
        self.image[11], self.image[13], self.image[15] = [0, -1], [0, 0], [0, 1]
        self.world = np.zeros((33, 3))
        self.world[11], self.world[13], self.world[15] = [0, -1, 0], [0, 0, 0], [1, 0, 0]

    def test_angulo_usa_coordenadas_3d_si_estan_disponibles(self):
        pose = _pose(self.image, self.world, use_world=True)
        self.assertAlmostEqual(pose.angle(11, 13, 15), 90.0)

    def test_angulo_usa_la_imagen_si_use_world_es_falso(self):
        pose = _pose(self.image, self.world, use_world=False)
        self.assertAlmostEqual(pose.angle(11, 13, 15), 180.0)

    def test_angulo_usa_la_imagen_si_no_hay_world(self):
        pose = _pose(self.image, None, use_world=True)
        self.assertAlmostEqual(pose.angle(11, 13, 15), 180.0)

    def test_visibilidad_media_de_varios_puntos(self):
        vis = np.zeros(33)
        vis[11], vis[12] = 0.8, 0.4
        pose = _pose(self.image, visibility=vis)
        self.assertAlmostEqual(pose.mean_visibility([11, 12]), 0.6)

    def test_visibilidad_media_sin_puntos_es_0(self):
        pose = _pose(self.image)
        self.assertEqual(pose.mean_visibility([]), 0.0)

    def test_vis_devuelve_la_visibilidad_de_un_punto(self):
        vis = np.zeros(33)
        vis[11] = 0.8
        pose = _pose(self.image, visibility=vis)
        self.assertAlmostEqual(pose.vis(11), 0.8)

    def test_point_devuelve_las_coordenadas_de_imagen(self):
        self.image[11] = [3, 4]
        pose = _pose(self.image)
        np.testing.assert_array_equal(pose.point(11), [3, 4])


# =============================================================================
# Detector: _next_timestamp (sin cargar MediaPipe)
# =============================================================================
def _detector_sin_mediapipe() -> PoseDetector:
    """Objeto PoseDetector sin modelo: solo sirve para probar _next_timestamp."""
    det = PoseDetector.__new__(PoseDetector)
    det._last_ts = -1
    det._offset = 0
    return det


class TestNextTimestamp(unittest.TestCase):
    def test_en_vivo_el_timestamp_crece(self):
        det = _detector_sin_mediapipe()
        primero = det._next_timestamp(None)
        segundo = det._next_timestamp(None)
        self.assertGreater(segundo, primero)

    def test_video_usa_su_propio_timestamp(self):
        det = _detector_sin_mediapipe()
        self.assertEqual(det._next_timestamp(0), 0)
        self.assertEqual(det._next_timestamp(33), 33)
        self.assertEqual(det._next_timestamp(66), 66)

    def test_timestamp_repetido_se_incrementa(self):
        det = _detector_sin_mediapipe()
        det._next_timestamp(100)
        self.assertEqual(det._next_timestamp(100), 101)

    def test_timestamp_hacia_atras_nunca_baja(self):
        det = _detector_sin_mediapipe()
        sucesion = [det._next_timestamp(t) for t in [0, 100, 50, 0, 0, 200]]
        for anterior, actual in zip(sucesion, sucesion[1:]):
            self.assertGreater(actual, anterior)

    def test_rebobinar_un_video_sigue_creciendo(self):
        det = _detector_sin_mediapipe()
        det._next_timestamp(0)
        det._next_timestamp(500)           # Fin del vídeo
        tras_rebobinar = det._next_timestamp(0)   # Se vuelve a empezar desde 0
        self.assertGreater(tras_rebobinar, 500)
        self.assertGreater(det._next_timestamp(33), tras_rebobinar)


# =============================================================================
# Modelos: _is_valid, ensure_model, download_model
# =============================================================================
class TestModelos(unittest.TestCase):
    def test_archivo_inexistente_no_es_valido(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(models._is_valid(Path(tmp) / "falta.task"))

    def test_archivo_demasiado_pequeno_no_es_valido(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pequeno.task"
            ruta.write_bytes(b"x" * 999_999)
            self.assertFalse(models._is_valid(ruta))

    def test_archivo_de_1_mega_es_valido(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "modelo.task"
            ruta.write_bytes(b"x" * 1_000_000)
            self.assertTrue(models._is_valid(ruta))

    def test_ensure_model_devuelve_el_modelo_si_ya_existe(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pose_landmarker_full.task"
            ruta.write_bytes(b"x" * 1_000_000)
            with patch("vision.models.model_path", return_value=ruta):
                self.assertEqual(models.ensure_model("full", download=False), ruta)

    def test_ensure_model_sin_descarga_avisa_si_falta(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pose_landmarker_full.task"
            with patch("vision.models.model_path", return_value=ruta), \
                    patch("vision.models.download_model") as descargar:
                with self.assertRaises(FileNotFoundError) as contexto:
                    models.ensure_model("full", download=False)
                descargar.assert_not_called()
                self.assertIn("download_models.py", str(contexto.exception))

    def test_ensure_model_con_descarga_delega_en_download_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "pose_landmarker_full.task"
            descargada = Path(tmp) / "descargada.task"
            with patch("vision.models.model_path", return_value=ruta), \
                    patch("vision.models.download_model", return_value=descargada) as descargar:
                self.assertEqual(models.ensure_model("full", download=True), descargada)
                descargar.assert_called_once_with("full")

    def test_variante_desconocida_falla_antes_de_descargar(self):
        with patch("vision.models.requests.get") as peticion:
            with self.assertRaises(ValueError) as contexto:
                models.download_model("ultra")
            peticion.assert_not_called()
            self.assertIn("Modelo desconocido", str(contexto.exception))


if __name__ == "__main__":
    unittest.main()
