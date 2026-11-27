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


def calibrar(cal: Calibrator, ex: Exercise, reposo: np.ndarray, maximo: np.ndarray):
    """Repetición de prueba (reposo -> máximo -> reposo) sin ruido. Devuelve el último estado."""
    estado, t = None, 0.0
    for p in repeticion(1.0):
        f = reposo + p * (maximo - reposo)
        estado = cal.update(t, f, ex.progress(f)) or estado
        t += 1 / 30
    return estado


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
        self.assertAlmostEqual(float(self.ex.normalize(np.array([165.0]))[0]), 0.0)
        self.assertAlmostEqual(float(self.ex.normalize(np.array([80.0]))[0]), 1.0)
        self.assertAlmostEqual(float(self.ex.normalize(np.array([122.5]))[0]), 0.5)

    def test_progreso_es_simetrico_entre_reposo_y_maximo(self):
        # Dos poses espejo (a 20 % y a 80 % del camino) suman 1 en progreso
        p_cerca_reposo = self.ex.progress(np.array([148.0]))
        p_cerca_maximo = self.ex.progress(np.array([97.0]))
        self.assertLess(p_cerca_reposo, 0.5)
        self.assertAlmostEqual(p_cerca_reposo + p_cerca_maximo, 1.0)

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

    def test_ramp_es_lineal_y_se_limita_entre_0_y_1(self):
        self.assertAlmostEqual(self.ex._ramp(5, 0, 10), 0.5)
        self.assertEqual(self.ex._ramp(-3, 0, 10), 0.0)     # Por debajo de 'ok' no penaliza
        self.assertEqual(self.ex._ramp(20, 0, 10), 1.0)     # Por encima de 'bad' satura

    def test_ramp_funciona_al_reves_cuando_bad_es_menor_que_ok(self):
        self.assertAlmostEqual(self.ex._ramp(7, 10, 0), 0.3)

    def test_ramp_con_ok_igual_a_bad_no_divide_por_cero(self):
        self.assertEqual(self.ex._ramp(1, 5, 5), 0.0)

    def test_asimetria_con_un_solo_lado_es_none(self):
        pose = esqueleto(flexion(120, "LEFT"))
        joints = ("SHOULDER", "ELBOW", "WRIST")
        self.assertIsNone(self.ex._asymmetry(pose, ("LEFT",), joints))

    def test_asimetria_con_el_mismo_angulo_es_cero(self):
        pose = esqueleto({**flexion(120, "LEFT"), **flexion(120, "RIGHT", dx=10)})
        joints = ("SHOULDER", "ELBOW", "WRIST")
        self.assertAlmostEqual(self.ex._asymmetry(pose, ("LEFT", "RIGHT"), joints), 0.0)

    def test_asimetria_de_40_grados_es_penalizacion_maxima(self):
        pose = esqueleto({**flexion(90, "LEFT"), **flexion(130, "RIGHT", dx=10)})
        joints = ("SHOULDER", "ELBOW", "WRIST")
        self.assertAlmostEqual(self.ex._asymmetry(pose, ("LEFT", "RIGHT"), joints), 1.0)

    def test_asimetria_de_27_5_grados_es_media_penalizacion(self):
        # Tolerancia 15º, máximo 40º: 27,5º de diferencia -> 0,5
        pose = esqueleto({**flexion(100, "LEFT"), **flexion(127.5, "RIGHT", dx=10)})
        joints = ("SHOULDER", "ELBOW", "WRIST")
        self.assertAlmostEqual(self.ex._asymmetry(pose, ("LEFT", "RIGHT"), joints), 0.5)

    def test_active_sides_bilateral_usa_los_dos_lados_visibles(self):
        pose = esqueleto({**flexion(165, "LEFT"), **flexion(165, "RIGHT", dx=10)})
        self.assertEqual(sorted(self.ex.active_sides(pose, 0.5)), ["LEFT", "RIGHT"])

    def test_active_sides_con_un_solo_lado_visible_usa_ese(self):
        pose = esqueleto(flexion(165, "LEFT"))
        self.assertEqual(self.ex.active_sides(pose, 0.5), ["LEFT"])

    def test_active_sides_descarta_un_lado_con_poca_visibilidad(self):
        pose = esqueleto(flexion(165, "LEFT"), vis=0.4)
        self.assertEqual(self.ex.active_sides(pose, 0.5), [])
        self.assertEqual(self.ex.active_sides(pose, 0.3), ["LEFT"])

    def test_measure_sin_lados_visibles_devuelve_none(self):
        self.assertIsNone(self.ex.measure(esqueleto({}), 0.5))

    def test_measure_devuelve_el_angulo_del_codo(self):
        medida = self.ex.measure(esqueleto(flexion(120, "LEFT")), 0.5)
        self.assertEqual(medida.sides, ("LEFT",))
        self.assertAlmostEqual(float(medida.features[0]), 120.0, places=4)


# =============================================================================
# Posturas y técnica de cada ejercicio
# =============================================================================
class TestFlexion(unittest.TestCase):
    def setUp(self):
        self.ex = PushUp()

    def test_plancha_es_postura_correcta(self):
        self.assertEqual(self.ex.check_posture(esqueleto(flexion(165)), ("LEFT",)), (True, ""))

    def test_de_pie_no_es_postura_de_flexion(self):
        ok, aviso = self.ex.check_posture(esqueleto(flexion(165, de_pie=True)), ("LEFT",))
        self.assertFalse(ok)
        self.assertEqual(aviso, "Colócate en plancha: cuerpo horizontal")

    def test_cadera_hundida_penaliza_el_cuerpo(self):
        penal = self.ex.form_penalties(esqueleto(flexion(165, hundido=124)), ("LEFT",), 0.9)
        self.assertAlmostEqual(penal["cuerpo"], 1.0)

    def test_cuerpo_recto_no_penaliza(self):
        penal = self.ex.form_penalties(esqueleto(flexion(165)), ("LEFT",), 0.9)
        self.assertAlmostEqual(penal["cuerpo"], 0.0)
        self.assertAlmostEqual(penal["rodillas"], 0.0)


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

    def test_manos_por_encima_de_los_hombros_es_correcto(self):
        self.assertEqual(self.ex.check_posture(self.dominada(100.0), ("LEFT",))[0], True)

    def test_manos_por_debajo_de_los_hombros_no_cuenta(self):
        ok, aviso = self.ex.check_posture(self.dominada(400.0), ("LEFT",))
        self.assertFalse(ok)
        self.assertEqual(aviso, "Agárrate a la barra: manos por encima de la cabeza")

    def test_tolera_manos_algo_por_debajo_del_hombro(self):
        # Tolerancia: 0,15 x torso = 30 px por debajo del hombro
        self.assertTrue(self.ex.check_posture(self.dominada(220.0), ("LEFT",))[0])
        self.assertFalse(self.ex.check_posture(self.dominada(250.0), ("LEFT",))[0])

    def test_barbilla_por_debajo_de_la_barra_penaliza(self):
        # Nariz 60 px (0,3 x torso) por debajo de la muñeca -> penalización máxima
        penal = self.ex.form_penalties(self.dominada(100.0, nariz_y=160.0), ("LEFT",), 0.9)
        self.assertAlmostEqual(penal["barbilla"], 1.0)

    def test_barbilla_por_encima_de_la_barra_no_penaliza(self):
        penal = self.ex.form_penalties(self.dominada(100.0, nariz_y=90.0), ("LEFT",), 0.9)
        self.assertAlmostEqual(penal["barbilla"], 0.0)

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

    def test_torso_erguido_es_postura_correcta(self):
        self.assertTrue(self.ex.check_posture(self.tronco(10.0), ("LEFT",))[0])

    def test_torso_muy_inclinado_no_cuenta(self):
        ok, aviso = self.ex.check_posture(self.tronco(70.0), ("LEFT",))
        self.assertFalse(ok)
        self.assertEqual(aviso, "Ponte de pie, con el torso erguido")

    def test_torso_inclinado_penaliza_a_mitad_de_escala(self):
        # Entre 30º (ok) y 50º (mal): 40º -> 0,5
        penal = self.ex.form_penalties(self.tronco(40.0), ("LEFT",), 0.5)
        self.assertAlmostEqual(penal["torso"], 0.5)

    def test_torso_erguido_no_penaliza(self):
        penal = self.ex.form_penalties(self.tronco(0.0), ("LEFT",), 0.5)
        self.assertAlmostEqual(penal["torso"], 0.0)

    def test_mide_solo_la_pierna_delantera_del_tobillo_mas_bajo(self):
        # El tobillo izquierdo está más abajo en la imagen: pierna delantera izquierda
        medida = self.ex.measure(self.piernas(500.0, 300.0), 0.5)
        self.assertEqual(medida.sides, ("LEFT",))

    def test_histeresis_no_cambia_de_pierna_por_diferencias_pequenas(self):
        self.ex.measure(self.piernas(500.0, 300.0), 0.5)           # Izquierda delantera
        self.assertEqual(self.ex.active_sides(self.piernas(500.0, 520.0), 0.5), ["LEFT"])

    def test_cambia_de_pierna_cuando_la_diferencia_es_clara(self):
        self.ex.measure(self.piernas(500.0, 300.0), 0.5)
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

    def test_reposo_no_genera_eventos(self):
        c = RepCounter(PushUp(), self.cfg)
        self.assertEqual(alimentar(c, [0.0] * 10), [])
        self.assertIs(c.state, RepState.REST)

    def test_superar_el_reposo_pasa_a_movimiento(self):
        c = RepCounter(PushUp(), self.cfg)
        alimentar(c, [0.0, 0.5])
        self.assertIs(c.state, RepState.MOVING)

    def test_superar_el_umbral_de_pico_pasa_a_pico(self):
        c = RepCounter(PushUp(), self.cfg)
        alimentar(c, [0.0, 0.8])
        self.assertIs(c.state, RepState.PEAK)

    def test_volver_al_reposo_cierra_la_repeticion(self):
        c = RepCounter(PushUp(), self.cfg)
        eventos = alimentar(c, [0.0, 0.8, 0.1])
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

    def test_movimiento_pequeno_se_ignora(self):
        # Llega al 30 %: por debajo del mínimo para una parcial
        self.assertEqual(alimentar(RepCounter(PushUp(), self.cfg), repeticion(0.3)), [])

    def test_sin_volver_al_reposo_no_hay_repeticion(self):
        subida = ciclo()[:31]                       # De 0 a 1, sin bajar
        self.assertEqual(alimentar(RepCounter(PushUp(), self.cfg), [0.0] * 5 + subida), [])

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

    def test_sin_pose_avisa_de_que_no_se_detecta(self):
        estado = self.tracker.update(None, 0.0)
        self.assertFalse(estado.detected)
        self.assertIn("No te detecto", estado.issue)

    def test_persona_sin_articulaciones_visibles_pide_colocarse(self):
        estado = self.tracker.update(esqueleto({}), 0.0)
        self.assertTrue(estado.detected)
        self.assertFalse(estado.measurable)
        self.assertEqual(estado.issue, self.ex.visibility_hint)

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

    def test_aviso_en_vivo_por_cadera_hundida(self):
        estado = self.tracker.update(esqueleto(flexion(165, hundido=124)), 0.0)
        self.assertEqual(estado.issue, self.ex.form_messages["cuerpo"])

    def test_sin_aviso_con_buena_tecnica(self):
        estado = self.tracker.update(esqueleto(flexion(165)), 0.0)
        self.assertIsNone(estado.issue)


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

    def test_tempo_normal_no_penaliza(self):
        self.assertEqual(_tempo_factor(2.0, self.cfg), 1.0)

    def test_tempo_rapido_se_penaliza_en_proporcion(self):
        self.assertAlmostEqual(_tempo_factor(0.4, self.cfg), 0.5)

    def test_tempo_muy_rapido_tiene_un_minimo(self):
        self.assertAlmostEqual(_tempo_factor(0.1, self.cfg), 0.4)

    def test_tempo_lento_se_penaliza_en_proporcion(self):
        self.assertAlmostEqual(_tempo_factor(16.0, self.cfg), 0.5)

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

    def test_repeticion_ideal_rellena_las_dos_curvas_para_la_grafica(self):
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

    def test_sin_datos_de_reposo_usa_el_primer_frame_de_la_repeticion(self):
        # Sin add_rest() el reposo es el primer frame grabado. Con una subida brusca,
        # ese frame es el de reposo de verdad (172º). (Ver informe: con una subida lenta
        # el primer frame grabado ya va a mitad de recorrido.)
        cal = self.nuevo_calibrador()
        reposo, maximo = np.array([172.0]), np.array([100.0])
        progresos = [0.0] * 5 + [0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.0]
        estado = None
        for i, p in enumerate(progresos):
            f = reposo + p * (maximo - reposo)
            estado = cal.update(i / 30, f, self.ex.progress(f)) or estado
        self.assertEqual(estado, "done")
        self.assertAlmostEqual(float(cal.result.rest[0]), 172.0, places=6)
        self.assertAlmostEqual(float(cal.result.peak[0]), 100.0, delta=2.0)

    def test_repeticion_que_no_llega_al_minimo_pide_repetir(self):
        cal = self.nuevo_calibrador()
        # Llega al 30 % del recorrido estándar: ni siquiera al 50 % exigido
        estado = calibrar(cal, self.ex, np.array([165.0]), np.array([140.0]))
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
    def test_catalogo_tiene_los_tres_ejercicios(self):
        self.assertEqual(set(EXERCISES), {"pullup", "pushup", "bulgarian_squat"})

    def test_la_clave_de_cada_clase_coincide_con_la_del_catalogo(self):
        for clave, clase in EXERCISES.items():
            self.assertEqual(clase.key, clave)

    def test_build_exercises_crea_las_instancias_en_el_orden_pedido(self):
        ejercicios = build_exercises(["bulgarian_squat", "pushup"])
        self.assertEqual([e.key for e in ejercicios], ["bulgarian_squat", "pushup"])
        self.assertIsInstance(ejercicios[0], BulgarianSquat)

    def test_cada_llamada_crea_instancias_nuevas(self):
        # Las referencias calibradas son de cada instancia: no deben compartirse
        self.assertIsNot(build_exercises(["pushup"])[0], build_exercises(["pushup"])[0])

    def test_sin_claves_devuelve_lista_vacia(self):
        self.assertEqual(build_exercises([]), [])

    def test_clave_desconocida_lanza_error_con_el_nombre(self):
        with self.assertRaisesRegex(ValueError, "remo"):
            build_exercises(["pushup", "remo"])


# =============================================================================
# Punto de entrada
# =============================================================================
if __name__ == "__main__":
    unittest.main()
