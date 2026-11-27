"""
tests/test_analisis.py  ·  Pruebas de ejercicios, conteo, puntuación y calibración.

Sin cámara: poses sintéticas construidas a mano y progresos simulados.
Complementa a tests/test_logic.py (que ya cubre el juego completo).

Ejecutar:  .venv/bin/python -m unittest tests.test_analisis -v
(Windows:  .venv\\Scripts\\python -m unittest tests.test_analisis -v)
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.calibration import CalibrationResult, Calibrator  # noqa: E402
from analysis.counter import ExerciseTracker, RepCounter, RepState  # noqa: E402
from analysis.scoring import (  # noqa: E402
    RepScore,
    _ease_from,
    _tempo_factor,
    build_profile,
    reference_trajectory,
    resample,
    score_repetition,
)
from config import Settings  # noqa: E402
from exercises.base import Exercise  # noqa: E402
from exercises.bulgarian_squat import BulgarianSquat  # noqa: E402
from exercises.catalog import EXERCISES, build_exercises  # noqa: E402
from exercises.pullup import PullUp  # noqa: E402
from exercises.pushup import PushUp  # noqa: E402
from vision.pose import Landmark, PoseFrame  # noqa: E402


# =============================================================================
# Utilidades: poses sintéticas y progresos simulados
# =============================================================================
def esqueleto(puntos: dict, vis: float = 0.95) -> PoseFrame:
    """Pose 2D en la que sólo son visibles las articulaciones de `puntos`.

    `puntos` usa los nombres de Landmark ("LEFT_SHOULDER", ...) y coordenadas de
    imagen en píxeles. Las demás articulaciones quedan en (0, 0) con visibilidad 0.
    """
    xy = np.zeros((33, 2))
    visibilidad = np.zeros(33)
    for nombre, punto in puntos.items():
        idx = int(Landmark[nombre])
        xy[idx] = punto
        visibilidad[idx] = vis
    mundo = np.hstack([xy, np.zeros((33, 1))])
    return PoseFrame(xy, visibilidad, mundo, 0, True)


def flexion(codo_deg: float, lado: str = "LEFT", dx: float = 0.0, hundido: float = 0.0,
            de_pie: bool = False) -> dict:
    """Puntos de un lado de una flexión vista de lado.

    codo_deg: ángulo del codo (hombro-codo-muñeca).
    hundido: cuánto baja la cadera, en píxeles (para simular mala técnica).
    de_pie: cuerpo vertical en lugar de plancha.
    """
    hombro = np.array([300.0 + dx, 300.0])
    codo = np.array([300.0 + dx, 380.0])
    a = np.radians(codo_deg)
    muneca = codo + 80.0 * np.array([np.sin(a), -np.cos(a)])
    if de_pie:
        cadera = np.array([300.0 + dx, 600.0])
        rodilla = np.array([300.0 + dx, 750.0])
        tobillo = np.array([300.0 + dx, 900.0])
    else:
        tobillo = np.array([900.0 + dx, 300.0])
        cadera = np.array([600.0 + dx, 300.0 + hundido])
        rodilla = (cadera + tobillo) / 2
    return {
        f"{lado}_SHOULDER": hombro,
        f"{lado}_ELBOW": codo,
        f"{lado}_WRIST": muneca,
        f"{lado}_HIP": cadera,
        f"{lado}_KNEE": rodilla,
        f"{lado}_ANKLE": tobillo,
    }


def flexion_en(p: float) -> np.ndarray:
    """Ángulo del codo de la flexión para un progreso p (0 = reposo 165º, 1 = máximo 80º)."""
    return np.array([165.0 + p * (80.0 - 165.0)])


def ciclo(n: int = 30) -> list:
    """Curva 0 -> 1 -> 0 con perfil coseno (2n + 1 valores; a 30 fps son 2 s)."""
    sube = (1 - np.cos(np.pi * np.arange(n + 1) / n)) / 2
    return list(sube) + list(sube[-2::-1])


def repeticion(p_max: float) -> list:
    """Progresos de una repetición que llega a p_max, con reposo antes y después."""
    return [0.0] * 5 + [p_max * x for x in ciclo()] + [0.0] * 5


def alimentar(contador: RepCounter, progresos: list) -> list:
    """Pasa una lista de progresos a 30 fps y devuelve los eventos que genera el contador."""
    eventos, t = [], 0.0
    for p in progresos:
        evento = contador.update(t, flexion_en(p), p, {})
        if evento:
            eventos.append(evento)
        t += 1 / 30
    return eventos


def poses_repeticion(**kwargs) -> list:
    """Poses de flexión (un lado) de una repetición completa; kwargs van a flexion()."""
    return [esqueleto(flexion(165 + p * (80 - 165), "LEFT", **kwargs)) for p in repeticion(1.0)]


def seguir(tracker: ExerciseTracker, poses: list, counting: bool = True) -> list:
    """Pasa las poses al tracker a 30 fps y devuelve sus estados."""
    return [tracker.update(pose, i / 30, counting) for i, pose in enumerate(poses)]


def rep_ideal(pico: float = 80.0):
    """Repetición ideal de 2 s en el ángulo del codo: (tiempos, features (N, 1))."""
    t = np.linspace(0, 2, 61)
    s = (1 - np.cos(2 * np.pi * t / 2)) / 2
    return t, (165.0 + s * (pico - 165.0))[:, None]


# =============================================================================
# Ejercicios: base (normalización, progreso, referencias, medición)
# =============================================================================
class TestEjercicioBase(unittest.TestCase):
    def setUp(self):
        self.ex = PushUp()

    def test_ejercicio_sin_features_no_se_puede_crear(self):
        class SinFeatures(Exercise):
            key = "sin_features"

        with self.assertRaises(ValueError):
            SinFeatures()

    def test_normalize_vale_0_en_reposo_1_en_maximo_y_0_5_en_medio(self):
        for angulo, esperado in [(165.0, 0.0), (80.0, 1.0), (122.5, 0.5)]:
            with self.subTest(angulo=angulo):
                self.assertAlmostEqual(float(self.ex.normalize(np.array([angulo]))[0]), esperado)

    def test_progreso_vale_0_en_reposo_1_en_maximo_y_es_simetrico(self):
        self.assertAlmostEqual(self.ex.progress(np.array([165.0])), 0.0)
        self.assertAlmostEqual(self.ex.progress(np.array([80.0])), 1.0)
        # Dos poses espejo (a 20 % y a 80 % del camino) suman 1 en progreso
        cerca_reposo = self.ex.progress(np.array([148.0]))
        cerca_maximo = self.ex.progress(np.array([97.0]))
        self.assertLess(cerca_reposo, 0.5)
        self.assertAlmostEqual(cerca_reposo + cerca_maximo, 1.0)

    def test_set_reference_cambia_las_referencias(self):
        self.ex.set_reference(np.array([170.0]), np.array([100.0]))
        np.testing.assert_allclose(self.ex.rest_vector, [170.0])
        np.testing.assert_allclose(self.ex.peak_vector, [100.0])
        self.assertTrue(self.ex.calibrated)
        self.assertAlmostEqual(self.ex.progress(np.array([100.0])), 1.0)

    def test_set_reference_rechaza_reposo_igual_a_maximo(self):
        with self.assertRaises(ValueError):
            self.ex.set_reference(np.array([120.0]), np.array([120.0]))

    def test_reset_reference_vuelve_a_las_referencias_estandar(self):
        self.ex.set_reference(np.array([170.0]), np.array([100.0]), (np.zeros(1), np.zeros(1)))
        self.ex.reset_reference()
        np.testing.assert_allclose(self.ex.rest_vector, [165.0])
        np.testing.assert_allclose(self.ex.peak_vector, [80.0])
        self.assertFalse(self.ex.calibrated)
        self.assertIsNone(self.ex.reference_profile)

    def test_ramp_es_lineal_se_limita_entre_0_y_1_y_admite_bad_menor_que_ok(self):
        casos = [
            ("a mitad", 5, 0, 10, 0.5),
            ("por debajo de ok no penaliza", -3, 0, 10, 0.0),
            ("por encima de bad satura", 20, 0, 10, 1.0),
            ("al revés (bad < ok)", 7, 10, 0, 0.3),
            ("ok igual a bad no divide por cero", 1, 5, 5, 0.0),
        ]
        for nombre, valor, ok, bad, esperado in casos:
            with self.subTest(nombre):
                self.assertAlmostEqual(self.ex._ramp(valor, ok, bad), esperado)

    def test_asimetria_entre_codos(self):
        # Tolerancia 15º, máximo 40º. Con un solo lado no hay nada que comparar.
        joints = ("SHOULDER", "ELBOW", "WRIST")
        casos = [(120, 120, 0.0), (90, 130, 1.0), (100, 127.5, 0.5)]
        for izquierdo, derecho, esperado in casos:
            with self.subTest(izquierdo=izquierdo, derecho=derecho):
                pose = esqueleto({**flexion(izquierdo, "LEFT"), **flexion(derecho, "RIGHT", dx=10)})
                self.assertAlmostEqual(self.ex._asymmetry(pose, ("LEFT", "RIGHT"), joints), esperado)
        pose = esqueleto(flexion(120, "LEFT"))
        self.assertIsNone(self.ex._asymmetry(pose, ("LEFT",), joints))

    def test_active_sides_elige_los_lados_visibles(self):
        dos_lados = esqueleto({**flexion(165, "LEFT"), **flexion(165, "RIGHT", dx=10)})
        self.assertEqual(sorted(self.ex.active_sides(dos_lados, 0.5)), ["LEFT", "RIGHT"])
        un_lado = esqueleto(flexion(165, "LEFT"))
        self.assertEqual(self.ex.active_sides(un_lado, 0.5), ["LEFT"])
        poco_visible = esqueleto(flexion(165, "LEFT"), vis=0.4)
        self.assertEqual(self.ex.active_sides(poco_visible, 0.5), [])
        self.assertEqual(self.ex.active_sides(poco_visible, 0.3), ["LEFT"])

    def test_measure_devuelve_el_angulo_del_codo_o_none_si_no_se_ve_nada(self):
        self.assertIsNone(self.ex.measure(esqueleto({}), 0.5))
        medida = self.ex.measure(esqueleto(flexion(120, "LEFT")), 0.5)
        self.assertEqual(medida.sides, ("LEFT",))
        self.assertAlmostEqual(float(medida.features[0]), 120.0, places=4)


# =============================================================================
# Posturas y técnica de cada ejercicio
# =============================================================================
class TestFlexion(unittest.TestCase):
    def setUp(self):
        self.ex = PushUp()

    def test_postura_correcta_solo_en_plancha(self):
        self.assertEqual(self.ex.check_posture(esqueleto(flexion(165)), ("LEFT",)), (True, ""))
        ok, aviso = self.ex.check_posture(esqueleto(flexion(165, de_pie=True)), ("LEFT",))
        self.assertFalse(ok)
        self.assertEqual(aviso, "Colócate en plancha: cuerpo horizontal")

    def test_cuerpo_recto_no_penaliza_y_cadera_hundida_penaliza_al_maximo(self):
        recto = self.ex.form_penalties(esqueleto(flexion(165)), ("LEFT",), 0.9)
        self.assertAlmostEqual(recto["cuerpo"], 0.0)
        self.assertAlmostEqual(recto["rodillas"], 0.0)
        hundido = self.ex.form_penalties(esqueleto(flexion(165, hundido=124)), ("LEFT",), 0.9)
        self.assertAlmostEqual(hundido["cuerpo"], 1.0)


class TestDominada(unittest.TestCase):
    """Torso de 200 px (hombro en y=200, cadera en y=400)."""

    def setUp(self):
        self.ex = PullUp()

    def dominada(self, muneca_y: float, nariz_y: float | None = None):
        puntos = {
            "LEFT_SHOULDER": (100.0, 200.0),
            "LEFT_ELBOW": (60.0, 200.0),
            "LEFT_WRIST": (100.0, muneca_y),
            "LEFT_HIP": (100.0, 400.0),
        }
        if nariz_y is not None:
            puntos["NOSE"] = (100.0, nariz_y)
        return esqueleto(puntos)

    def test_las_manos_tienen_que_estar_por_encima_del_hombro_con_algo_de_tolerancia(self):
        self.assertTrue(self.ex.check_posture(self.dominada(100.0), ("LEFT",))[0])
        # Tolerancia: 0,15 x torso = 30 px por debajo del hombro
        self.assertTrue(self.ex.check_posture(self.dominada(220.0), ("LEFT",))[0])
        ok, aviso = self.ex.check_posture(self.dominada(250.0), ("LEFT",))
        self.assertFalse(ok)
        self.assertEqual(aviso, "Agárrate a la barra: manos por encima de la cabeza")

    def test_barbilla_penaliza_solo_si_queda_por_debajo_de_la_barra(self):
        # Nariz 60 px (0,3 x torso) por debajo de la muñeca -> penalización máxima
        debajo = self.ex.form_penalties(self.dominada(100.0, nariz_y=160.0), ("LEFT",), 0.9)
        encima = self.ex.form_penalties(self.dominada(100.0, nariz_y=90.0), ("LEFT",), 0.9)
        self.assertAlmostEqual(debajo["barbilla"], 1.0)
        self.assertAlmostEqual(encima["barbilla"], 0.0)

    def test_barbilla_solo_se_evalua_cerca_de_arriba(self):
        penal = self.ex.form_penalties(self.dominada(100.0, nariz_y=160.0), ("LEFT",), 0.5)
        self.assertNotIn("barbilla", penal)


class TestSentadillaBulgara(unittest.TestCase):
    def setUp(self):
        self.ex = BulgarianSquat()

    @staticmethod
    def tronco(inclinacion_deg: float):
        """Cuerpo de pie (lado izquierdo) con el tronco inclinado `inclinacion_deg`."""
        a = np.radians(inclinacion_deg)
        cadera = np.array([300.0, 400.0])
        hombro = cadera + 300.0 * np.array([np.sin(a), -np.cos(a)])
        return esqueleto({"LEFT_HIP": cadera, "LEFT_SHOULDER": hombro})

    @staticmethod
    def piernas(tobillo_izq: float, tobillo_der: float, vis: float = 0.95):
        """Vista de lado: cadera en y=100, rodilla en y=300 y tobillos en las y dadas."""
        return esqueleto({
            "LEFT_SHOULDER": (300.0, 0.0), "LEFT_HIP": (300.0, 100.0),
            "LEFT_KNEE": (300.0, 300.0), "LEFT_ANKLE": (300.0, tobillo_izq),
            "RIGHT_SHOULDER": (340.0, 0.0), "RIGHT_HIP": (340.0, 100.0),
            "RIGHT_KNEE": (340.0, 300.0), "RIGHT_ANKLE": (340.0, tobillo_der),
        }, vis)

    def test_el_torso_tiene_que_estar_erguido(self):
        self.assertTrue(self.ex.check_posture(self.tronco(10.0), ("LEFT",))[0])
        ok, aviso = self.ex.check_posture(self.tronco(70.0), ("LEFT",))
        self.assertFalse(ok)
        self.assertEqual(aviso, "Ponte de pie, con el torso erguido")

    def test_la_penalizacion_del_torso_crece_entre_30_y_50_grados(self):
        for inclinacion, esperado in [(0.0, 0.0), (40.0, 0.5), (60.0, 1.0)]:
            with self.subTest(inclinacion=inclinacion):
                penal = self.ex.form_penalties(self.tronco(inclinacion), ("LEFT",), 0.5)
                self.assertAlmostEqual(penal["torso"], esperado)

    def test_mide_solo_la_pierna_delantera_la_del_tobillo_mas_bajo(self):
        # Tobillo izquierdo más abajo en la imagen: pierna delantera izquierda
        medida = self.ex.measure(self.piernas(500.0, 300.0), 0.5)
        self.assertEqual(medida.sides, ("LEFT",))

    def test_la_pierna_delantera_solo_cambia_si_la_diferencia_es_clara(self):
        self.ex.measure(self.piernas(500.0, 300.0), 0.5)           # Izquierda delantera
        self.assertEqual(self.ex.active_sides(self.piernas(500.0, 520.0), 0.5), ["LEFT"])
        self.assertEqual(self.ex.active_sides(self.piernas(500.0, 600.0), 0.5), ["RIGHT"])

    def test_sin_pierna_delantera_visible_no_hay_medida(self):
        self.assertIsNone(self.ex.measure(self.piernas(500.0, 300.0, vis=0.2), 0.5))

    def test_reset_olvida_la_pierna_delantera(self):
        self.ex.measure(self.piernas(500.0, 300.0), 0.5)
        self.ex.reset()
        self.assertIsNone(self.ex._front)


class TestPenalizacionesGenerales(unittest.TestCase):
    def test_penalizaciones_entre_0_y_1_con_claves_de_form_messages(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            puntos = {nombre: tuple(rng.uniform(0, 400, 2)) for nombre in Landmark.__members__}
            pose = esqueleto(puntos)
            for ex in build_exercises(EXERCISES):
                penal = ex.form_penalties(pose, ("LEFT", "RIGHT"), 0.9)
                for regla, valor in penal.items():
                    self.assertIn(regla, ex.form_messages, ex.key)
                    self.assertTrue(0.0 <= valor <= 1.0, f"{ex.key}/{regla}: {valor}")


# =============================================================================
# Conteo de repeticiones (RepCounter)
# =============================================================================
class TestRepCounter(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings().scoring

    def test_estados_segun_el_progreso(self):
        # Umbrales: reposo 0,25 y pico 0,75
        c = RepCounter(PushUp(), self.cfg)
        self.assertEqual(alimentar(c, [0.0] * 10), [])
        self.assertIs(c.state, RepState.REST)
        alimentar(c, [0.5])
        self.assertIs(c.state, RepState.MOVING)
        alimentar(c, [0.8])
        self.assertIs(c.state, RepState.PEAK)
        eventos = alimentar(c, [0.1])
        self.assertIs(c.state, RepState.REST)
        self.assertEqual([e.kind for e in eventos], ["rep"])

    def test_repeticion_completa_cuenta_y_se_puntua(self):
        eventos = alimentar(RepCounter(PushUp(), self.cfg), repeticion(1.0))
        self.assertEqual([e.kind for e in eventos], ["rep"])
        self.assertAlmostEqual(eventos[0].max_progress, 1.0, places=6)
        self.assertIsInstance(eventos[0].score, RepScore)
        self.assertGreater(eventos[0].score.total, 90)

    def test_dos_repeticiones_seguidas_cuentan_dos(self):
        eventos = alimentar(RepCounter(PushUp(), self.cfg), repeticion(1.0) + repeticion(1.0))
        self.assertEqual([e.kind for e in eventos], ["rep", "rep"])

    def test_repeticion_incompleta_es_parcial_sin_puntuacion(self):
        # Llega al 60 % (>= 45 % y < 75 %): cuenta como parcial
        eventos = alimentar(RepCounter(PushUp(), self.cfg), repeticion(0.6))
        self.assertEqual([e.kind for e in eventos], ["partial"])
        self.assertIsNone(eventos[0].score)

    def test_no_hay_evento_si_el_movimiento_es_pequeno_o_no_vuelve_al_reposo(self):
        pequeno = repeticion(0.3)                      # Por debajo del 45 % de una parcial
        sin_volver = [0.0] * 5 + ciclo()[:31]          # De 0 a 1, sin bajar
        for nombre, progresos in [("pequeño", pequeno), ("sin volver", sin_volver)]:
            with self.subTest(nombre):
                self.assertEqual(alimentar(RepCounter(PushUp(), self.cfg), progresos), [])

    def test_repeticion_abandonada_tras_15_segundos_se_descarta(self):
        c = RepCounter(PushUp(), self.cfg)
        c.update(0.0, flexion_en(0.0), 0.0, {})
        c.update(0.1, flexion_en(0.5), 0.5, {})
        evento = c.update(0.1 + self.cfg.abandon_rep_seconds + 0.5, flexion_en(0.5), 0.5, {})
        self.assertIsNone(evento)
        self.assertIs(c.state, RepState.REST)


# =============================================================================
# ExerciseTracker: medición, postura, avisos y contador
# =============================================================================
class TestExerciseTracker(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.ex = PushUp()
        self.tracker = ExerciseTracker(self.ex, self.settings.game, self.settings.scoring)

    def test_sin_pose_o_sin_articulaciones_visibles_avisa_al_usuario(self):
        sin_pose = self.tracker.update(None, 0.0)
        self.assertFalse(sin_pose.detected)
        self.assertIn("No te detecto", sin_pose.issue)
        sin_articulaciones = self.tracker.update(esqueleto({}), 0.1)
        self.assertTrue(sin_articulaciones.detected)
        self.assertFalse(sin_articulaciones.measurable)
        self.assertEqual(sin_articulaciones.issue, self.ex.visibility_hint)

    def test_postura_incorrecta_no_cuenta_repeticiones(self):
        estados = seguir(self.tracker, poses_repeticion(de_pie=True))
        self.assertFalse(estados[-1].posture_ok)
        self.assertEqual(estados[0].issue, "Colócate en plancha: cuerpo horizontal")
        self.assertEqual([e.event for e in estados if e.event], [])
        self.assertIs(self.tracker.counter.state, RepState.REST)

    def test_con_counting_false_mide_pero_no_cuenta(self):
        estados = seguir(self.tracker, poses_repeticion(), counting=False)
        self.assertGreater(max(e.progress for e in estados), 0.75)
        self.assertEqual([e.event for e in estados if e.event], [])

    def test_con_buena_postura_cuenta_una_repeticion(self):
        estados = seguir(self.tracker, poses_repeticion())
        eventos = [e.event for e in estados if e.event]
        self.assertEqual([e.kind for e in eventos], ["rep"])

    def test_aviso_en_vivo_solo_si_la_tecnica_es_mala(self):
        hundida = self.tracker.update(esqueleto(flexion(165, hundido=124)), 0.0)
        self.assertEqual(hundida.issue, self.ex.form_messages["cuerpo"])
        buena = self.tracker.update(esqueleto(flexion(165)), 0.1)
        self.assertIsNone(buena.issue)


# =============================================================================
# Puntuación (scoring.py)
# =============================================================================
class TestPuntuacion(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings().scoring
        self.ex = PushUp()

    def test_resample_interpola_linealmente_cada_columna(self):
        tiempos = np.array([0.0, 1.0, 2.0])
        valores = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 40.0]])
        r = resample(tiempos, valores, 5)
        np.testing.assert_allclose(r[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(r[:, 1], [10.0, 15.0, 20.0, 30.0, 40.0])

    def test_tempo_penaliza_lo_muy_rapido_y_lo_muy_lento_con_un_minimo(self):
        # Rango sin penalización: de 0,8 s a 8 s
        casos = [(2.0, 1.0), (0.4, 0.5), (0.1, 0.4), (16.0, 0.5)]
        for duracion, esperado in casos:
            with self.subTest(duracion=duracion):
                self.assertAlmostEqual(_tempo_factor(duracion, self.cfg), esperado)

    def test_referencia_sin_perfil_sube_de_0_a_1_y_baja_a_0(self):
        ref = reference_trajectory(np.array([0.0]), np.array([0.0]), 20)
        self.assertEqual(ref.shape, (40, 1))
        self.assertAlmostEqual(ref[0, 0], 0.0, places=6)
        self.assertAlmostEqual(ref[19, 0], 1.0, places=6)
        self.assertAlmostEqual(ref[-1, 0], 0.0, places=6)
        self.assertTrue(np.all(np.diff(ref[:20, 0]) >= 0))
        self.assertTrue(np.all(np.diff(ref[20:, 0]) <= 0))

    def test_referencia_con_perfil_medido_sigue_ese_perfil(self):
        rampa = np.linspace(0.0, 1.0, 20)[:, None]
        ref = reference_trajectory(np.array([0.0]), np.array([0.0]), 20, (rampa, rampa))
        esperado = np.concatenate([np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20)])
        np.testing.assert_allclose(ref[:, 0], esperado, atol=1e-9)

    def test_perfil_de_una_repeticion_va_de_0_a_1_en_ida_y_vuelta(self):
        t, features = rep_ideal()
        u = (features - 165.0) / (80.0 - 165.0)
        up, down = build_profile(t, u, np.array([1.0]), 20)
        self.assertEqual(up.shape, (20, 1))
        self.assertEqual(down.shape, (20, 1))
        self.assertAlmostEqual(up[0, 0], 0.0, places=6)
        self.assertAlmostEqual(up[-1, 0], 1.0, places=6)
        self.assertAlmostEqual(down[0, 0], 0.0, places=6)
        self.assertAlmostEqual(down[-1, 0], 1.0, places=6)

    def test_fase_sin_movimiento_usa_el_coseno_ideal(self):
        t = np.linspace(0, 2, 61)
        cos = (1 - np.cos(2 * np.pi * t / 2)) / 2
        u = np.stack([cos, np.zeros_like(cos)], axis=1)     # Segundo ángulo quieto
        up, down = build_profile(t, u, np.array([0.5, 0.5]), 20)
        ideal = _ease_from(np.zeros(2), 20)
        np.testing.assert_allclose(up[:, 1], ideal[:, 1], atol=1e-9)
        np.testing.assert_allclose(down[:, 1], ideal[:, 1], atol=1e-9)

    def test_repeticion_ideal_puntua_alto_y_rellena_las_dos_curvas_para_la_grafica(self):
        t, features = rep_ideal()
        score = score_repetition(self.ex, t, features, [{} for _ in t], self.cfg)
        self.assertEqual(len(score.curve), 40)
        self.assertEqual(len(score.reference_curve), 40)
        self.assertGreater(score.total, 90)

    def test_pasarse_de_la_referencia_no_penaliza_el_rango(self):
        t, features = rep_ideal(pico=60.0)                  # Más profundo que el máximo (80º)
        score = score_repetition(self.ex, t, features, [{} for _ in t], self.cfg)
        self.assertAlmostEqual(score.rom, 1.0)

    def test_tecnica_es_uno_menos_la_media_de_las_penalizaciones(self):
        t, features = rep_ideal()
        penalizaciones = [{"cuerpo": 0.2 if i % 2 else 0.4} for i in range(len(t))]
        media = np.mean([p["cuerpo"] for p in penalizaciones])
        score = score_repetition(self.ex, t, features, penalizaciones, self.cfg)
        self.assertAlmostEqual(score.form, 1.0 - media, places=6)

    def test_con_menos_de_tres_muestras_no_hay_trayectoria(self):
        score = score_repetition(self.ex, np.array([0.0, 0.5]), np.array([[165.0], [80.0]]),
                                 [{}, {}], self.cfg)
        self.assertEqual(score.trajectory, 0.0)
        self.assertEqual(score.curve, [])
        self.assertEqual(score.reference_curve, [])

    def test_to_dict_redondea_a_tres_decimales(self):
        score = RepScore(total=99.87654, rom=1 / 3, trajectory=0.5, form=1.0, tempo=1.0,
                         duration=1.23456, feedback=["Aviso"], curve=[0.123456, 1.0])
        datos = score.to_dict()
        self.assertEqual(datos["total"], 99.877)
        self.assertEqual(datos["rom"], 0.333)
        self.assertEqual(datos["duration"], 1.235)
        self.assertEqual(datos["curve"], [0.123, 1.0])
        self.assertEqual(datos["feedback"], ["Aviso"])


# =============================================================================
# Calibración (analysis/calibration.py)
# =============================================================================
class TestCalibracion(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.ex = PushUp()

    def nuevo_calibrador(self) -> Calibrator:
        return Calibrator(self.ex, self.settings.game, self.settings.scoring)

    def test_apply_fija_las_referencias_y_el_perfil_del_resultado(self):
        perfil = (np.linspace(0.0, 1.0, 40)[:, None], np.linspace(0.0, 1.0, 40)[:, None])
        resultado = CalibrationResult(np.array([170.0]), np.array([100.0]), perfil, ())
        resultado.apply(self.ex)
        self.assertTrue(self.ex.calibrated)
        np.testing.assert_allclose(self.ex.rest_vector, [170.0])
        np.testing.assert_allclose(self.ex.peak_vector, [100.0])
        self.assertIs(self.ex.reference_profile, perfil)

    def test_sin_datos_de_reposo_usa_el_frame_mas_cercano_al_reposo(self):
        # Sin add_rest() y con una subida lenta, el último frame bajo el umbral ya va
        # a mitad de recorrido (progreso ~0,2): el reposo debe salir del frame de
        # menor progreso, que aquí es el de los 165º quietos del principio.
        cal = self.nuevo_calibrador()
        reposo, maximo = np.array([165.0]), np.array([100.0])
        estado = None
        for i, p in enumerate(repeticion(1.0)):
            f = reposo + p * (maximo - reposo)
            estado = cal.update(i / 30, f, self.ex.progress(f)) or estado
        self.assertEqual(estado, "done")
        self.assertAlmostEqual(float(cal.result.rest[0]), 165.0, places=6)
        self.assertAlmostEqual(float(cal.result.peak[0]), 100.0, delta=2.0)

    def test_repeticion_que_no_llega_al_minimo_pide_repetir(self):
        cal = self.nuevo_calibrador()
        reposo, maximo = np.array([165.0]), np.array([140.0])   # Solo el 30 % del recorrido estándar
        estado = None
        for i, p in enumerate(repeticion(1.0)):
            f = reposo + p * (maximo - reposo)
            estado = cal.update(i / 30, f, self.ex.progress(f)) or estado
        self.assertEqual(estado, "retry")
        self.assertIsNone(cal.result)

    def test_abandono_por_tiempo_descarta_la_repeticion_de_prueba(self):
        cal = self.nuevo_calibrador()
        reposo, medio = np.array([172.0]), np.array([122.5])        # Progreso 0,5
        cal.update(0.0, reposo, self.ex.progress(reposo))
        cal.update(0.1, medio, self.ex.progress(medio))            # Empieza la repetición
        estado = cal.update(15.6, medio, self.ex.progress(medio))  # Sigue ahí 15 s después
        self.assertIsNone(estado)
        self.assertFalse(cal.recording)
        self.assertIsNone(cal.result)


# =============================================================================
# Catálogo de ejercicios
# =============================================================================
class TestCatalogo(unittest.TestCase):
    def test_catalogo_tiene_los_tres_ejercicios_con_su_propia_clave(self):
        self.assertEqual(set(EXERCISES), {"pullup", "pushup", "bulgarian_squat"})
        for clave, clase in EXERCISES.items():
            self.assertEqual(clase.key, clave)

    def test_build_exercises_crea_instancias_nuevas_en_el_orden_pedido(self):
        ejercicios = build_exercises(["bulgarian_squat", "pushup"])
        self.assertEqual([e.key for e in ejercicios], ["bulgarian_squat", "pushup"])
        self.assertIsInstance(ejercicios[0], BulgarianSquat)
        # Las referencias calibradas son de cada instancia: no deben compartirse
        self.assertIsNot(build_exercises(["pushup"])[0], build_exercises(["pushup"])[0])

    def test_clave_desconocida_lanza_error_con_el_nombre(self):
        with self.assertRaisesRegex(ValueError, "remo"):
            build_exercises(["pushup", "remo"])


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
