# MiEntrenadorPersonal

Práctica 2 de IPM (Interacción Persona-Máquina), hecha con visión por computador.

MiEntrenadorPersonal es un juego serio del ámbito de la salud. Con la cámara del portátil, la
del móvil o un vídeo grabado, detecta los 33 puntos del cuerpo con **MediaPipe Pose Landmarker**
y guía al usuario en tres ejercicios con su propio peso: **dominada**, **flexión** y
**sentadilla búlgara**. Cuenta las repeticiones solo y puntúa la técnica de cada una de 0 a 100,
comparando el movimiento con una pose de reposo, una de máximo esfuerzo y una trayectoria de
referencia entre las dos. Como en los juegos de "dibuja un círculo", cuanto más se parece tu
movimiento a la referencia, más puntos.

Lo más destacado:

- **Repetición de prueba:** antes de cada ejercicio haces una repetición lenta y completa, y de
  ella salen *tus* poses de reposo y de máximo y *tu* trayectoria. Las demás se comparan con eso.
- Gráfica de cada repetición frente a la de referencia, consejos en vivo, sonidos e historial
  de partidas.
- Botones con el ratón además del teclado; fuente (cámara o vídeo) y ejercicios elegibles desde
  el menú; grabación de la ventana en `.mp4`.

La cámara debe estar quieta mientras haces el ejercicio. El código parte del ejemplo de la
asignatura (<https://github.com/CarloHSUA/IPM/>), un juego de tocar círculos con las manos.

## Índice

1. [Para qué sirve](#1-para-qué-sirve)
2. [Requisitos e instalación](#2-requisitos-e-instalación)
3. [Ejecución](#3-ejecución)
4. [Cómo se juega](#4-cómo-se-juega)
5. [Cómo funciona](#5-cómo-funciona)
6. [Arquitectura del código](#6-arquitectura-del-código)
7. [Configuración](#7-configuración)
8. [Resultados guardados](#8-resultados-guardados)
9. [Pruebas](#9-pruebas)
10. [Generar el .zip de entrega](#10-generar-el-zip-de-entrega)
11. [Añadir un ejercicio nuevo](#11-añadir-un-ejercicio-nuevo)
12. [Problemas frecuentes](#12-problemas-frecuentes)
13. [Notas para desarrolladores](#13-notas-para-desarrolladores)

---

## 1. Para qué sirve

- **Corregir la técnica y prevenir lesiones:** cada repetición se compara con la referencia y se
  dan consejos concretos durante y después del ejercicio.
- **Motivación:** puntos, objetivo de repeticiones, tiempo límite y resumen final.
- **Seguimiento:** cada partida se guarda, así que el historial muestra la evolución.
- **Entrenar sin supervisión:** solo hace falta una cámara.
- **Adaptable a rehabilitación:** los ángulos de reposo y máximo, los umbrales y las reglas son
  parámetros que se pueden ajustar a rangos de movimiento reducidos.

---

## 2. Requisitos e instalación

Necesitas:

- **Python 3.11 o 3.12.** Las versiones fijadas de `jax` y `scipy` exigen 3.11 o más, y
  `mediapipe 0.10.21` no tiene ruedas para 3.13. Los scripts buscan `python3.12` o `python3.11`.
- El módulo `venv` (en Ubuntu: `sudo apt install python3.12-venv`).
- Una webcam, un móvil que emita su cámara por red, o un vídeo grabado.
- Internet solo para instalar (paquetes pip y los tres modelos `.task`, unos 46 MB).
- Opcional: sonido (en Linux, `sudo apt install libportaudio2` o tener `paplay`/`aplay`) y el
  explorador de archivos del menú (en Linux, `sudo apt install python3.12-tk`). Sin ellos el
  juego funciona igual.

Funciona en Linux, macOS (`scripts/*.sh`) y Windows (`scripts\*.bat`). Se ha probado en Ubuntu
24.04 con Python 3.12.

Instalación en un equipo limpio (crea `.venv/` dentro del proyecto; se puede lanzar desde
cualquier carpeta):

```bash
source scripts/dependencies.sh      # Linux / macOS (o: bash scripts/dependencies.sh)
scripts\dependencies.bat            # Windows
```

El script busca un Python compatible (se puede forzar con `PYTHON=/ruta/python3.12`), crea el
entorno (`RECREATE_VENV=1` lo recrea), instala `requirements.txt` y descarga los modelos en
`models/`. Con `source`, deja el entorno activado.

A mano sería:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
.venv/bin/python scripts/download_models.py             # --model full para bajar solo uno
```

---

## 3. Ejecución

```bash
bash scripts/run.sh          # Linux / macOS
scripts\run.bat              # Windows
python src/main.py           # con el entorno activado
```

Si falta el modelo, `main.py` lo descarga. Casi todo se puede elegir también desde el menú
(fuente, ejercicios, repetición de prueba). Sin cámara, el juego arranca igual y pide abrir un
vídeo.

### Opciones

| Opción | Por defecto | Qué hace |
|--------|-------------|----------|
| `--source X` | `0` | Índice de webcam, URL de stream (móvil) o ruta de un vídeo |
| `--video ARCHIVO` | | Juega con un vídeo grabado en vez de la cámara |
| `--model {lite,full,heavy}` | `full` | `lite` va más rápido y `heavy` es más preciso |
| `--exercises ...` | `pullup pushup bulgarian_squat` | Ejercicios y su orden |
| `--reps N` | `10` | Repeticiones objetivo por ejercicio |
| `--duration S` | `60` | Segundos máximos por ejercicio |
| `--width W --height H` | `1280 720` | Resolución pedida a la webcam |
| `--mirror` / `--no-mirror` | | Con cámara se espeja siempre salvo `--no-mirror`; un vídeo no se espeja salvo `--mirror` |
| `--start S` | `0` | Con un vídeo, empieza en el segundo `S` |
| `--no-calibrate` | | Sin repetición de prueba (referencias estándar) |
| `--record [ARCHIVO]` | | Graba la ventana en `.mp4` (por defecto `results/grabacion_*.mp4`); también con `G` |
| `--no-sound` | | Sin sonidos |
| `--no-download` | | No descarga el modelo si falta |
| `--no-save` | | No guarda resultados en `results/` |

```bash
bash scripts/run.sh --model lite
bash scripts/run.sh --exercises pushup bulgarian_squat --reps 8 --duration 45
bash scripts/run.sh --video videos/flexiones_playa.mp4 --exercises pushup --start 3 --record
```

### Cámara del móvil

Instala una app que emita la cámara por la red local (IP Webcam en Android, DroidCam en
Android/iOS), conecta móvil y portátil a la misma Wi-Fi y pasa la URL del vídeo:

```bash
bash scripts/run.sh --source http://192.168.1.20:8080/video     # IP Webcam
bash scripts/run.sh --source http://192.168.1.20:4747/video     # DroidCam
```

### Jugar con un vídeo grabado

El juego funciona igual que en vivo: pulsas `ENTER` y la persona del vídeo sigue las
indicaciones de la pantalla (reposo, cuenta atrás, repeticiones).

```bash
bash scripts/run.sh --video demo.mp4 --exercises pushup
```

Sin terminal: lanza el juego sin opciones, pulsa **Abrir vídeo… (V)** y elige uno de la carpeta
`videos/` o con **Buscar en el ordenador…**.

En [`videos/`](videos/) hay cuatro vídeos de demostración de Wikimedia Commons, ya probados con
el juego. Al elegir uno, marca en el menú solo su ejercicio:

| Vídeo | Ejercicio |
|---|---|
| `flexiones_playa.mp4` | Flexión |
| `flexiones_kettlebell.mp4` | Flexión (pulsa **Empezar ya** en las instrucciones) |
| `sentadilla_bulgara_lateral.mp4` | Sentadilla búlgara |
| `dominadas_espalda.mp4` | Dominada |

Su origen, autoría y licencia están en [`videos/CREDITOS.md`](videos/CREDITOS.md).

Cómo se comporta con un vídeo:

- Se queda en pausa en el menú, con el primer frame, y empieza al pulsar EMPEZAR.
- Se reproduce a velocidad real; si el equipo no llega, se saltan frames. El reloj del juego es
  el del vídeo, así que cuenta atrás, tiempos y tempo son exactos aunque vaya lento.
- `P` pausa y reanuda (el juego se congela también). `R`, o "Jugar otra vez", lo rebobina y
  vuelve al menú.
- Los vídeos verticales se centran sobre un fondo oscuro y los muy grandes se reducen.
- Cuando se acaba, se cierra el ejercicio en curso con lo conseguido y los que faltan se marcan
  como saltados. Si se acaba durante las instrucciones, `ENTER` o `N` saltan el ejercicio.

Para grabar un vídeo: cámara quieta y colocada como en la [tabla de colocación](#colocación),
cuerpo entero en el encuadre y buena luz. Empieza a grabar unos segundos antes de colocarte,
quédate en reposo unos 2 s (el juego necesita 1,5 s), espera la cuenta atrás de 3 s, haz una
repetición de prueba lenta y completa y luego las repeticiones. Si grabas varios ejercicios,
el vídeo sigue corriendo en la pantalla de resultado hasta que pulses `ENTER`, así que es más
cómodo grabar un vídeo por ejercicio.

Para la presentación, `--record` (o `G`) graba la ventana del juego en `results/grabacion_*.mp4`;
sirve de plan B si el equipo va lento al grabar la pantalla.

---

## 4. Cómo se juega

```
           ┌── SELECTOR DE VÍDEO (V)
MENÚ ──────┤
  │        └── HISTORIAL (H)
  │ EMPEZAR
  ▼
┌─ por cada ejercicio ───────────────────────────────────────────────────────────┐
│ INSTRUCCIONES ──(reposo 1,5 s o ENTER)──▶ CUENTA ATRÁS (3 s)                  │
│      │                                         │                              │
│      N (saltar)               REPETICIÓN DE PRUEBA (si la calibración está    │
│      │                        activa; ENTER = usar la referencia estándar)    │
│      │                                         ▼                              │
│      ▼                                  ACTIVO (hasta 10 reps o 60 s)         │
│ RESULTADO (con la gráfica de tu mejor repetición) ◀────────┘                  │
└───────┬────────────────────────────────────────────────────────────────────────┘
        ▼
    RESUMEN (puntos totales, comparación con la partida anterior, se guarda el JSON)
```

1. **Menú:** elige la fuente (cámara o vídeo), los ejercicios y si hay repetición de prueba, y
   pulsa EMPEZAR.
2. **Instrucciones:** se explica el ejercicio y dónde poner la cámara. Colócate en reposo; cuando
   el sistema te detecta, un anillo se llena durante 1,5 s y empieza solo (o pulsa `ENTER`).
3. **Cuenta atrás** de 3 s, con pitidos.
4. **Repetición de prueba:** haz una repetición lenta y completa. El juego guarda tu reposo, tu
   máximo y cómo te mueves, y te avisa con "¡Referencia guardada!". Si no llegas a la mitad del
   recorrido estándar, te pide repetirla. Tienes 20 s; pasado ese tiempo, o con `ENTER`, usa la
   referencia estándar.
5. **Ejercicio:** repeticiones y objetivo, tiempo, puntos, una barra de progreso con los umbrales,
   el esqueleto con las articulaciones medidas resaltadas, la puntuación desglosada de la última
   repetición con su gráfica, avisos en vivo y un mensaje tras cada repetición ("¡Perfecta!",
   "+82", "Incompleta").
6. **Resultado del ejercicio:** repeticiones, puntos, medias por componente, consejos más
   repetidos y la gráfica de tu mejor repetición (las demás, en gris) frente a la referencia.
7. **Resumen:** tabla de ejercicios, puntuación total y diferencia con la partida anterior.
8. **Historial:** gráficas de puntos y puntuación media de las últimas partidas y tabla con las
   más recientes.

### Controles

Todas las pantallas tienen botones con ratón; cada uno indica su tecla.

| Tecla | Acción |
|-------|--------|
| `ENTER` / `ESPACIO` | Empezar, continuar, terminar el ejercicio o usar la referencia estándar |
| `1`, `2`, `3` (menú) | Activar o desactivar cada ejercicio |
| `C` / `V` (menú) | Usar la cámara / abrir un vídeo |
| `K` (menú) | Activar o desactivar la repetición de prueba |
| `H` (menú y resumen) | Historial |
| `1`…`8` / `B` (selector de vídeo) | Elegir un vídeo de la lista / buscar en el ordenador |
| `P` | Pausar o reanudar el vídeo (también con clic en su indicador) |
| `G` | Empezar o parar la grabación de la ventana |
| `N` | Saltar el ejercicio |
| `F` | Pantalla completa |
| `R` | Volver al menú (reinicia la partida; con un vídeo, lo rebobina) |
| `ESC` | Volver (en el selector de vídeo y el historial) o salir |
| `Q` / cerrar ventana | Salir |

### Colocación

| Ejercicio | Cámara | Reposo | Máximo esfuerzo |
|-----------|--------|--------|-----------------|
| Dominada | De frente o de lado, de las manos a las rodillas | Colgado, brazos estirados | Barbilla por encima de la barra |
| Flexión | **De lado**, a ras de suelo, cuerpo entero | Plancha con brazos estirados | Codos a unos 80-90º |
| Sentadilla búlgara | **De lado**, a la altura de la cadera, cuerpo entero | De pie, pie trasero en un banco | Rodilla delantera a unos 90º |

---

## 5. Cómo funciona

### Detección

`PoseDetector` usa el Pose Landmarker en modo `VIDEO` (síncrono, con timestamps estrictamente
crecientes) y devuelve por frame una `PoseFrame` con los puntos en píxeles (`image_xy`), su
visibilidad y los puntos 3D en metros (`world_xyz`). Los ángulos se calculan en 3D, que no
depende de la distancia a la cámara ni de que esté perfectamente de lado. La imagen se espeja
antes de detectar, así que los puntos coinciden con lo que se ve. Se elige el lado por
visibilidad, no por etiqueta izquierda/derecha.

### Poses de referencia como ángulos

Cada ejercicio define ángulos articulares (el vértice de un trío de puntos) con un valor en
reposo y otro en máximo esfuerzo. Los ángulos no dependen de la posición ni de la estatura.

| Ejercicio | Ángulo | Reposo | Máximo | Peso |
|-----------|--------|--------|--------|------|
| Dominada | codo (hombro-**codo**-muñeca) | 165º | 55º | 1.0 |
| Dominada | hombro (cadera-**hombro**-codo) | 165º | 45º | 0.7 |
| Flexión | codo | 165º | 80º | 1.0 |
| Sentadilla búlgara | rodilla (cadera-**rodilla**-tobillo) | 170º | 90º | 1.0 |
| Sentadilla búlgara | cadera (hombro-**cadera**-rodilla) | 170º | 100º | 0.6 |

En la dominada y la flexión se promedian los lados visibles (visibilidad media ≥ 0.5). En la
sentadilla búlgara solo se mide la pierna adelantada, la del tobillo más bajo en la imagen (con
histéresis para que no parpadee). Los ángulos se suavizan con una media móvil exponencial
(α = 0.45).

### Progreso y conteo

Cada ángulo se normaliza como `u = (ángulo − reposo) / (máximo − reposo)`. El progreso `p` vale
0 en reposo y 1 en el máximo, y se calcula con las distancias ponderadas de `u` a ambas poses:
`p = d_reposo / (d_reposo + d_máximo)`.

El contador es una máquina de estados con histéresis:

```
REPOSO ──(p > 0.25)──▶ MOVIENDO ──(p ≥ 0.75)──▶ PICO ──(p ≤ 0.25)──▶ REPETICIÓN (se puntúa)
                          └──(vuelve a p ≤ 0.25 sin llegar a 0.75)──▶ "incompleta" si p_max ≥ 0.45
```

Una repetición que dura más de 15 s se descarta. Si no se cumple la postura del ejercicio, no se
cuenta nada y se avisa: en la dominada las muñecas deben estar por encima de los hombros; en la
flexión el tronco debe estar a 45º o más de la vertical, y en la sentadilla búlgara a 55º o menos.

### Puntuación de cada repetición

```
total = 100 · (0.35 · rango + 0.35 · trayectoria · tempo + 0.30 · técnica)
```

- **Rango:** lo cerca que se llegó al máximo. Pasarse no penaliza.
- **Trayectoria:** parecido con la referencia, `exp(−6 · max(error − 0.015, 0))`. La repetición
  se parte en el pico en ida y vuelta, cada fase se remuestrea a 20 muestras y se compara con el
  tramo correspondiente del perfil coseno ideal `(1 − cos πτ) / 2` (arranque y frenado suaves),
  tras normalizar por el pico alcanzado: la trayectoria mide la forma y el rango, la profundidad.
  Alinear por fases deja que la ida y la vuelta duren distinto, pero penaliza parones, tirones y
  rebotes. Un DTW libre se descartó porque deformaba el tiempo y absorbía justo esos defectos.
- **Tempo:** 1 si dura entre 0,8 y 8 s; si dura menos, `duración / 0,8` (mínimo 0,4); si dura
  más, `8 / duración` (mínimo 0,5).
- **Técnica:** 1 menos la media de las penalizaciones de las reglas del ejercicio.

Con datos sintéticos, una repetición ideal da 96-100; muy rápida (0,5 s), a tirones o con mala
técnica, entre 73 y 80; y con un 55-70 % del recorrido no cuenta (incompleta).

Reglas de técnica (penalización lineal entre el valor ok y el valor malo):

| Ejercicio | Regla | Medida | Ok → mal |
|-----------|-------|--------|----------|
| Dominada | `barbilla` | Nariz respecto a la muñeca más alta (solo con p ≥ 0.8) | a la altura → 0.3 × torso por debajo |
| Dominada | `simetria` | Diferencia entre los codos | 15º → 40º |
| Dominada | `balanceo` | Flexión de cadera (kipping) | 150º → 115º |
| Flexión | `cuerpo` | Ángulo hombro-cadera-tobillo | 165º → 140º |
| Flexión | `rodillas` | Ángulo cadera-rodilla-tobillo | 160º → 130º |
| Flexión | `simetria` | Diferencia entre los codos | 15º → 40º |
| Sentadilla búlgara | `torso` | Inclinación del tronco respecto a la vertical | 30º → 50º |

Tras cada repetición se dan consejos si el rango es < 0.75, una regla tiene penalización media
≥ 0.3, el tempo se sale de rango o la trayectoria es < 0.6. En vivo se avisa cuando una regla
supera 0.6 en el frame actual. Los puntos del juego son la suma de las puntuaciones de las
repeticiones.

### Repetición de prueba

Tras la cuenta atrás, el usuario hace una repetición lenta y completa
([`src/analysis/calibration.py`](src/analysis/calibration.py)):

1. **Reposo:** mediana de los ángulos mientras esperaba quieto en las instrucciones y la cuenta
   atrás. Con menos de 3 muestras se usa el frame más cercano al reposo que se haya visto.
2. **Repetición:** se detecta con las referencias estándar, como el contador. Tiene que llegar a
   `p ≥ 0.5`; si se queda corta, se pide repetirla.
3. **Máximo:** media de los ángulos en los frames de máximo progreso.
4. **Comprobación por ángulo:** cada ángulo debe moverse en el mismo sentido que el estándar y
   al menos un 40 % de su recorrido; si no, se queda con la referencia estándar y se avisa. Si
   falla el ángulo principal, se pide repetir.
5. **Trayectoria:** la ida y la vuelta se remuestrean, se fuerzan monótonas para quitar ruido y
   se escalan de 0 a 1. Sustituyen al coseno ideal en la comparación.

Con la calibración desactivada (menú o `--no-calibrate`) se usan las referencias estándar.

Cada repetición guarda dos curvas de 40 puntos: `curve` (el progreso real) y `reference_curve`
(con la que se comparó). Son las que se dibujan en el HUD y en el resultado.

---

## 6. Arquitectura del código

```
IPM_Prac_2/
├── README.md
├── requirements.txt
├── models/                     # Modelos .task (no versionado)
├── results/                    # Partidas en JSON y grabaciones (no versionado)
├── videos/                     # Vídeos para el selector del menú
├── scripts/                    # dependencies, run, download_models, make_zip
├── tests/
│   ├── utilidades.py           # Poses, fuentes y reloj falsos para las pruebas del juego
│   ├── test_vision.py          # Cámara, geometría, detector y modelos
│   ├── test_analisis.py        # Ejercicios, contador, puntuación y calibración
│   ├── test_interfaz.py        # Lienzo, widgets, pantallas, sonido y grabación
│   ├── test_juego.py           # Game, sesiones, historial, main.py, config y scripts
│   └── test_logic.py           # Escenarios completos: partidas del menú al resumen
└── src/
    ├── main.py                 # Argumentos, modelo, cámara, detector y juego
    ├── config.py               # Rutas absolutas y parámetros
    ├── vision/                 # camera, models, detector, pose
    ├── exercises/              # base, pullup, pushup, bulgarian_squat, catalog
    ├── analysis/               # counter (RepCounter y ExerciseTracker), scoring, calibration
    ├── ui/                     # canvas, widgets, screens, dialogs, sound, recorder
    └── game/                   # game (fases y bucle), session, history
```

Por frame, el flujo es:

```
Camera.read() → PoseDetector.detect() → ExerciseTracker.update()
                                          medir ángulos → suavizar → progreso → postura
                                          → reglas de técnica → RepCounter → score_repetition
Game._update()   avanza las fases, calibra y guarda las repeticiones en ExerciseSession
Game._draw()     dibuja con Canvas y las funciones de ui/screens.py
Game._handle_key() / _handle_click()  →  Game._do(acción)
```

Las dependencias entre paquetes no tienen ciclos:

```
config ◀── vision.models, vision.camera
vision.pose ◀── vision.detector
config, vision.pose ◀── exercises ◀── analysis.scoring ◀── analysis.counter, analysis.calibration ◀── ui ◀── game ◀── main
```

Clases principales:

| Clase o función | Dónde | Para qué |
|-----------------|-------|----------|
| `Settings` y sus dataclasses | `config.py` | Parámetros; instancia única `settings` |
| `Camera`, `NoSource` | `vision/camera.py` | Webcam, stream o vídeo (`play`, `pause`, `rewind`, `clock`, `ended`); fuente vacía |
| `PoseDetector`, `PoseFrame` | `vision/` | Detección y landmarks de un frame, con `angle(a, b, c)` en 3D |
| `Exercise`, `FeatureSpec` | `exercises/base.py` | Medida, progreso y reglas; `set_reference()` / `reset_reference()` |
| `RepCounter`, `ExerciseTracker` | `analysis/counter.py` | Máquina de estados y pipeline por frame |
| `score_repetition`, `RepScore` | `analysis/scoring.py` | Puntuación y consejos de una repetición |
| `Calibrator` | `analysis/calibration.py` | Repetición de prueba → referencias del usuario |
| `Canvas`, `TextRenderer` | `ui/canvas.py` | Dibujo en unidades de diseño (alto 720) y texto con Pillow |
| `button`, `checkbox`, gráficas | `ui/widgets.py` | Elementos clicables y gráficas |
| `draw_*` | `ui/screens.py` | Una función por pantalla |
| `Sound`, `Recorder` | `ui/` | Pitidos sin bloquear; grabación de la ventana |
| `Game`, `Phase` | `game/game.py` | Bucle principal y fases |
| `ExerciseSession`, `save_sessions`, `load_history` | `game/` | Resultados, guardado en JSON e historial |

### Formato de los archivos

Todos los `.py` siguen el mismo esquema (solo aparece lo que hace falta):

```python
"""
Qué hay en el archivo, en una a cuatro líneas.
"""

from __future__ import annotations

import os                       # librería estándar
import numpy as np              # terceros
from config import settings     # del proyecto (imports absolutos)

_CONSTANTE = 3                  # constantes en mayúsculas, después de los imports


class Ejemplo:
    ...

# --- Otra parte del archivo ---    # solo en archivos largos
```

Los `__init__.py` son solo un docstring con la lista de archivos de la carpeta.

### Decisiones de diseño

- **Modo `VIDEO` en vez de `LIVE_STREAM`:** cada resultado corresponde exactamente al frame
  dibujado y el código es más simple; el tracking entre frames sigue activo.
- **Histéresis en el conteo** (0.25 / 0.75) para que el ruido no cuente repeticiones dobles.
- **Rutas absolutas** calculadas desde `__file__` en `config.py`; `main.py` añade `src/` a
  `sys.path`, así que los paquetes se importan como módulos de primer nivel.
- **Texto con Pillow**, porque `cv2.putText` no soporta tildes ni eñes.
- **Un único `cv2.waitKey` por frame:** el código base lo llamaba varias veces y se perdían
  pulsaciones.
- Para usar ángulos 2D en vez de 3D: `ModelConfig.use_world_landmarks = False`.

---

## 7. Configuración

Los parámetros están en [`src/config.py`](src/config.py); las referencias y reglas de cada
ejercicio, en su archivo de [`src/exercises/`](src/exercises/).

| Grupo | Parámetro | Valor | Significado |
|-------|-----------|-------|-------------|
| Rutas | `MODELS_DIR`, `RESULTS_DIR`, `VIDEOS_DIR` | `<raíz>/models`, `results`, `videos` | Se cambian con `POSE_GAME_MODELS_DIR`, `POSE_GAME_RESULTS_DIR` y `POSE_GAME_VIDEOS_DIR` |
| Cámara | `source`, `width`, `height` | `0`, 1280, 720 | Fuente y resolución (los frames más altos se reducen) |
| Cámara | `mirror`, `mirror_video`, `start_seconds` | `True`, `False`, 0 | Espejo con cámara y con vídeo; segundo de inicio |
| Modelo | `variant`, `use_world_landmarks` | `full`, `True` | `lite` / `full` / `heavy`; ángulos 3D |
| Juego | `exercise_duration`, `target_reps` | 60 s, 10 | Fin de cada ejercicio |
| Juego | `ready_hold_seconds`, `ready_progress` | 1.5 s, 0.2 | Arranque automático en reposo |
| Juego | `min_visibility`, `smoothing_alpha` | 0.5, 0.45 | Visibilidad mínima y suavizado (1 = sin suavizar) |
| Juego | `calibrate`, `calibration_timeout` | `True`, 20 s | Repetición de prueba y tiempo para hacerla |
| Juego | `calibration_min_progress`, `calibration_min_span` | 0.5, 0.4 | Profundidad mínima de la prueba y recorrido mínimo de cada ángulo |
| Juego | `sound`, `record_path`, `record_fps` | `True`, `None`, 30 | Sonido y grabación |
| Puntuación | `rest_threshold`, `peak_threshold`, `partial_threshold` | 0.25, 0.75, 0.45 | Conteo |
| Puntuación | `weight_rom`, `weight_trajectory`, `weight_form` | 0.35, 0.35, 0.30 | Pesos de la nota |
| Puntuación | `trajectory_sensitivity`, `trajectory_noise_floor` | 6.0, 0.015 | Exigencia de la trayectoria |
| Puntuación | `min_rep_seconds`, `max_rep_seconds` | 0.8, 8.0 | Tempo |

Si sin repetición de prueba un ejercicio no se cuenta bien, ajusta primero `rest` y `peak` de
sus `FeatureSpec`. La consola imprime cada repetición con su desglose, y ayuda a afinar.

---

## 8. Resultados guardados

Cada partida completada se guarda en `results/session_AAAAMMDD_HHMMSS.json` (si dos terminan en
el mismo segundo, la segunda lleva el sufijo `_2`, `_3`…). Ejemplo con valores ilustrativos:

```json
{
  "date": "2026-11-11T11:58:30",
  "total_points": 296,
  "exercises": [
    {
      "exercise": "pushup", "name": "Flexión", "skipped": false,
      "reps": 3, "partial_reps": 1, "points": 296,
      "average_score": 98.5, "best_score": 100.0, "duration_s": 12.4,
      "calibrated": true,
      "reference": {"rest": {"codo": 171.2}, "peak": {"codo": 98.4}},
      "repetitions": [
        {"total": 100.0, "rom": 1.0, "trajectory": 1.0, "form": 1.0, "tempo": 1.0,
         "duration": 1.4, "feedback": [],
         "curve": [0.21, 0.27, "…", 0.24], "reference_curve": [0.21, 0.26, "…", 0.25]}
      ]
    }
  ]
}
```

---

## 9. Pruebas

No necesitan cámara, ventana ni red, y no escriben en `results/`, `videos/` ni `models/`:

```bash
.venv/bin/python -m unittest discover -s tests -v       # Windows: .venv\Scripts\python
.venv/bin/python -m unittest tests.test_juego -v        # un solo archivo
```

| Archivo | Qué comprueba |
|---------|---------------|
| `test_vision.py` | Fuente de vídeo (pausa, fin, rebobinado), `fit_frame`, geometría de poses, timestamps del detector y localización de modelos |
| `test_analisis.py` | Progreso, reglas de cada ejercicio, contador, puntuación y calibración |
| `test_interfaz.py` | Escala del lienzo, botones, gráficas, pantallas, sonido, grabación y lista de vídeos |
| `test_juego.py` | Teclas y botones, cambio de fuente, fin del vídeo, guardado, historial, argumentos de `main.py`, configuración y scripts |
| `test_logic.py` | Repeticiones sintéticas por ejercicio (ideal > 90, corta incompleta, defectos penalizados), calibración, y partidas completas con cámara, con vídeo y con repetición de prueba |

`tests/utilidades.py` reúne lo compartido: poses sintéticas, cámara y vídeo falsos, un reloj
simulado y la clase `PruebaDeJuego`, que da a cada prueba una carpeta de resultados temporal y
un reloj propio. Así ninguna depende de la velocidad de la máquina ni del orden.

---

## 10. Generar el .zip de entrega

```bash
.venv/bin/python scripts/make_zip.py "Nombre Apellido1 Apellido2" ["Nombre2 Apellido1 Apellido2"]
# -> nombre_apellido1_apellido2_nombre2_apellido1_apellido2.zip en la raíz del proyecto
```

Quita tildes y espacios del nombre y excluye `.venv/`, `models/`, `results/`, `.git/`, cachés,
los `.task` y los vídeos propios que hayas copiado en `videos/` (los cuatro de demostración sí
van dentro). El receptor lo despliega
con `scripts/dependencies.sh` o `.bat`.

---

## 11. Añadir un ejercicio nuevo

1. Crea `src/exercises/mi_ejercicio.py` con una subclase de `Exercise`:

   ```python
   from exercises.base import Exercise, FeatureSpec

   class Sentadilla(Exercise):
       key = "squat"                       # identificador para --exercises
       name = "Sentadilla"
       instructions = ("Pies a la anchura de los hombros.", "Baja hasta 90º.", "Sube sin bloquear.")
       camera_tip = "Cámara de lado, cuerpo entero."
       rest_hint = "Ponte de pie"
       visibility_hint = "No veo bien tus piernas"
       rom_feedback = "Baja más"
       features = (FeatureSpec("rodilla", ("HIP", "KNEE", "ANKLE"), rest=170, peak=90),)
       required_joints = ("HIP", "KNEE", "ANKLE")
       bilateral = True
       form_messages = {"torso": "Torso más erguido"}

       def form_penalties(self, pose, sides, progress):
           return {"torso": self._ramp(self._torso_lean(pose, sides), 35, 60)}
   ```

   Opcionalmente, sobreescribe `check_posture()` (impedir el conteo fuera de posición) y
   `active_sides()` (elegir el lado, como hace la sentadilla búlgara).
2. Regístralo en `EXERCISES`, en `src/exercises/catalog.py`.
3. Añádelo a `GameConfig.exercises` o pásalo con `--exercises squat`.
4. Las pruebas de `test_logic.py` recorren `EXERCISES` y lo cubren solas; añade las reglas
   nuevas en `test_analisis.py`.

---

## 12. Problemas frecuentes

| Problema | Solución |
|----------|----------|
| `se necesita Python 3.11 o 3.12` | Instala Python 3.12 o indica uno: `PYTHON=/ruta/python3.12 bash scripts/dependencies.sh` |
| `Error creando el venv` | `sudo apt install python3-venv` (o `python3.12-venv`) |
| `No se pudo abrir la cámara '0'` | Comprueba que existe y que nadie más la usa; prueba `--source 1`. En Linux, el usuario debe estar en el grupo `video` |
| La cámara del móvil no conecta | Misma Wi-Fi; abre la URL en el navegador del portátil y usa la del vídeo (`/video`) |
| Va lento | `--model lite` y/o `--width 640 --height 480` |
| No cuenta repeticiones | Mira el aviso de la pantalla (visibilidad o postura); que se vea el cuerpo necesario y haya buena luz |
| Cuenta repeticiones que no lo son, o no llega al máximo | Ajusta `rest` / `peak` del ejercicio o `peak_threshold` (sección 7) |
| `No se pudo abrir el vídeo` | Comprueba la ruta; si OpenCV no lo lee, conviértelo: `ffmpeg -i entrada.mov salida.mp4` |
| Con un vídeo no empieza el ejercicio | La persona debe estar en reposo unos 1,5 s en las instrucciones; si no, `ENTER` empieza la cuenta atrás |
| La repetición de prueba no se acepta ("Otra vez") | Debe llegar a la mitad del recorrido estándar y volver al reposo. Si no puede ser, `ENTER` usa la referencia estándar o desactívala con `K` |
| Puntúa alto con repeticiones cortas | Se compara con tu repetición de prueba: haz la prueba completa o desactiva la calibración |
| `No hay salida de audio disponible` | En Linux, `sudo apt install libportaudio2`. Funciona igual sin sonido (`--no-sound` quita el aviso) |
| "Buscar en el ordenador…" desactivado | Falta tkinter (en Linux, `sudo apt install python3.12-tk`); copia los vídeos en `videos/` |
| La grabación `.mp4` no se abre | Ábrela con VLC (usa MPEG-4 Part 2) |
| Avisos de Qt / Wayland (Linux) | Son inofensivos; si no aparece la ventana, `QT_QPA_PLATFORM=xcb bash scripts/run.sh` |
| `W0000` / `Created TensorFlow Lite XNNPACK delegate` | Son logs normales de MediaPipe |

---

## 13. Notas para desarrolladores

- **Punto de entrada:** `src/main.py`, siempre con el Python de `.venv` (`scripts/run.sh` o
  `run.bat`).
- **Imports absolutos del módulo concreto** (`from vision.pose import PoseFrame`), porque
  `main.py`, `scripts/download_models.py` y las pruebas añaden `src/` a `sys.path`. Los
  `__init__.py` no reexportan nada y no se usan imports relativos.
- **Rutas:** usa las constantes absolutas de `config` (`PROJECT_ROOT`, `RESULTS_DIR`…), nunca
  rutas relativas al directorio de trabajo.
- **Sin ciclos de importación:** respeta el orden de la sección 6. `config`, `vision/models.py`
  y `vision/pose.py` no importan `mediapipe` ni `cv2`, para que `download_models.py` sea ligero.
- **Coordenadas:** `PoseFrame.image_xy` está en píxeles del frame mostrado (ya espejado), con la
  `y` hacia abajo. Los ángulos usan `world_xyz` si existe.
- **UI:** dibuja siempre a través de `Canvas`, en unidades de diseño (alto 720). El texto se
  encola y se pinta en `Canvas.finish()`; no uses `cv2.putText`.
- **Bucle:** un único `cv2.waitKey` por frame, en `Game.run()`.
- **Reloj:** usa `Game._now()` y no `time.monotonic()`: con un vídeo es el reloj del vídeo. Solo
  se detecta y se llama a `_update()` con frames nuevos (`Camera.frame_id`).
- **Acciones:** teclas y clics acaban en `Game._do(acción)`. Un botón nuevo se dibuja con
  `widgets.button(c, "acción", ...)` y se trata en `_do()`.
- **Referencias:** no toques `rest_vector` / `peak_vector` a mano; usa
  `Exercise.set_reference()` / `reset_reference()`. Cada partida empieza con las estándar.
- **Pruebas sin cámara:** `Game._update()`, `_draw()`, `_handle_key()` y `_do()` se pueden llamar
  con poses sintéticas y un reloj simulado; mira `PruebaDeJuego` en `tests/utilidades.py` y las
  partidas de `tests/test_logic.py`.
- **Dependencias:** compatibles con Python 3.11-3.12, `numpy < 2` y `protobuf < 5` (restricciones
  de mediapipe 0.10.21).
- **No versionar:** `.venv/`, `models/`, `results/`, `*.zip` ni otros vídeos de `videos/` aparte
  de los cuatro de demostración (ya está en `.gitignore`).
