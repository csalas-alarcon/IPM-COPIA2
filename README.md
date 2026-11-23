# MiEntrenadorPersonal: juego de entrenamiento con visión por computador

**Práctica 2 de IPM (Interacción Persona-Máquina) mediante visión por computador.**

MiEntrenadorPersonal es un **juego serio** (*serious game*) del ámbito de la salud. Usa la cámara del
portátil o del móvil y el modelo **MediaPipe Pose Landmarker** para detectar los 33 puntos del
cuerpo, y guía al usuario en tres ejercicios de fuerza con su propio peso:

1. **Dominada**
2. **Flexión** 
3. **Sentadilla búlgara** 

El juego **cuenta las repeticiones automáticamente** y **puntúa la técnica de cada una (0-100)**
comparando el movimiento del usuario con una pose de reposo, una pose de máximo esfuerzo y una
trayectoria de referencia entre ambas. Durante el ejercicio da consejos en tiempo real ("mantén el
cuerpo recto", "baja más", etc.). Al terminar guarda un historial en JSON.

> Requisito de uso: **cámara estática** (apoyada, sin moverse durante el ejercicio).

Basado en el código de ejemplo de la asignatura: <https://github.com/CarloHSUA/IPM/>, un juego
de tocar círculos con las manos usando MediaPipe.

---

## Índice

1. [Por qué es un juego serio (utilidad)](#1-por-qué-es-un-juego-serio-utilidad)
2. [Requisitos](#2-requisitos)
3. [Instalación en un equipo limpio](#3-instalación-en-un-equipo-limpio)
4. [Ejecución](#4-ejecución)
5. [Cómo se juega](#5-cómo-se-juega)
6. [Cómo funciona (algoritmos)](#6-cómo-funciona-algoritmos)
7. [Arquitectura del código](#7-arquitectura-del-código)
8. [Configuración](#8-configuración)
9. [Resultados guardados (formato JSON)](#9-resultados-guardados-formato-json)
10. [Pruebas](#10-pruebas)
11. [Generar el .zip de entrega](#11-generar-el-zip-de-entrega)
12. [Cómo añadir un ejercicio nuevo](#12-cómo-añadir-un-ejercicio-nuevo)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Notas para agentes de IA / desarrolladores](#14-notas-para-agentes-de-ia--desarrolladores)

---

## 1. Por qué es un juego serio (utilidad)

| Objetivo | Cómo lo aborda el juego |
|----------|-------------------------|
| **Prevenir lesiones y corregir la técnica** | Cada repetición se compara con la técnica de referencia y se dan consejos concretos en vivo y al final |
| **Motivación y adherencia** al ejercicio | Puntos, marcador, objetivo de repeticiones, tiempo límite y resumen final (gamificación) |
| **Seguimiento del estado funcional** | Cada sesión se guarda (repeticiones, rango de movimiento, fluidez y técnica), así que se puede ver la evolución a lo largo del tiempo |
| **Entrenamiento autónomo sin supervisión** | Sólo hace falta una cámara: no se necesitan sensores ni un entrenador presente |
| **Adaptable a rehabilitación** | Los ángulos de reposo y máximo, los umbrales y las reglas son parámetros: se pueden ajustar a ejercicios terapéuticos o a rangos de movimiento reducidos |

---

## 2. Requisitos

| Requisito | Detalle |
|-----------|---------|
| **Python 3.11 o 3.12** | **Obligatorio.** `jax`, `scipy` y `contourpy` (versiones fijadas) exigen ≥ 3.11 y `mediapipe 0.10.21` no tiene ruedas para 3.13+. Los scripts buscan automáticamente `python3.12` o `python3.11` |
| Módulo `venv` | En Ubuntu/Debian: `sudo apt install python3.12-venv` (o `python3-venv`) |
| Cámara | Webcam o móvil con una app que emita la cámara por red (ver [Cámara del móvil](#cámara-del-móvil)) |
| Conexión a internet | Sólo en la instalación (paquetes pip y modelos `.task`, ~46 MB los tres) |
| Sistema operativo | Linux y macOS (`scripts/*.sh`) y Windows (`scripts\*.bat`). Probado en Ubuntu 24.04 con Python 3.12 |

---

## 3. Instalación en un equipo limpio

No hace falta tener ningún entorno virtual previo: el script lo crea todo dentro de la carpeta del
proyecto (`.venv/`). Los scripts usan rutas absolutas, así que se pueden lanzar desde cualquier
directorio.

### Linux / macOS

```bash
cd IPM_Prac_2
source scripts/dependencies.sh     # o: bash scripts/dependencies.sh
```

### Windows

```bat
cd IPM_Prac_2
scripts\dependencies.bat
```

### Qué hace el script de instalación

1. Busca un Python compatible (3.12 o 3.11). Si no lo encuentra, se detiene con un mensaje que
   explica cómo instalarlo. Se puede forzar uno con `PYTHON=/ruta/python3.12`.
2. Crea el entorno virtual en `<raíz>/.venv`. Si ya existe con otra versión de Python, lo recrea
   (`RECREATE_VENV=1` fuerza la recreación).
3. Actualiza `pip` e instala `requirements.txt`.
4. Descarga los modelos `pose_landmarker_{lite,full,heavy}.task` en `<raíz>/models/`.
5. Con `source`, deja el entorno activado en la terminal.

### Instalación manual (equivalente)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
.venv/bin/python scripts/download_models.py             # --model full para descargar sólo uno
```

---

## 4. Ejecución

```bash
bash scripts/run.sh                    # Linux/macOS (usa .venv/bin/python src/main.py)
scripts\run.bat                        # Windows
python src/main.py                     # con el venv activado
```

Si el modelo no existe, `main.py` lo descarga automáticamente.

### Opciones de línea de comandos

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--source X` | `0` | Índice de webcam (`0`, `1`…) o URL de stream (cámara del móvil) |
| `--model {lite,full,heavy}` | `full` | `lite` es más rápido (equipos lentos) y `heavy` más preciso |
| `--exercises ...` | `pullup pushup bulgarian_squat` | Ejercicios y orden |
| `--reps N` | `10` | Repeticiones objetivo por ejercicio (termina al alcanzarlas) |
| `--duration S` | `60` | Segundos máximos por ejercicio |
| `--width W --height H` | `1280 720` | Resolución pedida a la webcam |
| `--no-mirror` | — | No mostrar la imagen en espejo |
| `--no-download` | — | No descargar el modelo si falta (da error) |
| `--no-save` | — | No guardar resultados en `results/` |

Ejemplos:

```bash
bash scripts/run.sh --model lite
bash scripts/run.sh --exercises pushup bulgarian_squat --reps 8 --duration 45
```

### Cámara del móvil

Instala en el móvil una app que emita la cámara por la red local, por ejemplo **IP Webcam**
(Android) o **DroidCam** (Android/iOS). Conecta el móvil y el portátil a la misma red Wi-Fi y
pasa la URL del vídeo:

```bash
bash scripts/run.sh --source http://192.168.1.20:8080/video     # IP Webcam
bash scripts/run.sh --source http://192.168.1.20:4747/video     # DroidCam
```

---

## 5. Cómo se juega

### Flujo

```
MENÚ ──ENTER──▶ ┌─ por cada ejercicio ─────────────────────────────────────────────┐
                │ INSTRUCCIONES ──(reposo mantenido 1,5 s o ENTER)──▶ CUENTA ATRÁS │
                │      │                                                3 s         │
                │      N (saltar)                                       ▼          │
                │      ▼                          ACTIVO (hasta 10 reps o 60 s)     │
                │ RESULTADO ◀──────────────────────────────┘                        │
                └───────┬───────────────────────────────────────────────────────────┘
                        ▼
                    RESUMEN (puntos totales, se guarda el JSON) ──ENTER──▶ MENÚ
```

1. **Menú**: pulsa `ENTER`.
2. **Instrucciones**: se explica el ejercicio y dónde poner la cámara. Colócate en la **posición
   de reposo**. Cuando el sistema te detecta en ella, un anillo se llena durante 1,5 s y empieza
   solo (o pulsa `ENTER`).
3. **Cuenta atrás** de 3 s.
4. **Ejercicio**. En pantalla aparecen:
   - repeticiones / objetivo, tiempo restante y puntos;
   - una **barra vertical de progreso** reposo → máximo, con las marcas de los umbrales;
   - el esqueleto, con las articulaciones medidas resaltadas (azul = reposo, verde = máximo);
   - la puntuación de la **última repetición** desglosada (rango, trayectoria, técnica);
   - **avisos en vivo** (técnica, postura incorrecta, persona fuera de cámara);
   - tras cada repetición, un mensaje ("¡Perfecta!", "+82", "Incompleta"…) con consejos.
5. **Resultado del ejercicio**: repeticiones, puntos, medias por componente y los consejos más
   repetidos.
6. **Resumen**: tabla de los tres ejercicios y la puntuación total.

### Controles

| Tecla | Acción |
|-------|--------|
| `ENTER` / `ESPACIO` | Empezar / continuar / terminar el ejercicio actual |
| `N` | Saltar ejercicio |
| `F` | Pantalla completa |
| `R` | Volver al menú (reinicia la partida) |
| `ESC` / `Q` / cerrar ventana | Salir |

### Colocación para cada ejercicio

| Ejercicio | Cámara | Reposo (inicio) | Máximo esfuerzo |
|-----------|--------|-----------------|-----------------|
| Dominada | De frente o de lado; que se vea de las manos a las rodillas | Colgado de la barra, brazos estirados | Barbilla por encima de la barra |
| Flexión | **De lado**, a ras de suelo, cuerpo entero | Plancha con brazos estirados | Codos a ~80-90º |
| Sentadilla búlgara | **De lado**, a la altura de la cadera, cuerpo entero | De pie, pie trasero apoyado en un banco o silla | Rodilla delantera a ~90º |

---

## 6. Cómo funciona (algoritmos)

### 6.1 Detección de la pose

- `PoseDetector` (en [src/vision/detector.py](src/vision/detector.py)) usa el Pose Landmarker en modo
  `VIDEO`, con timestamps monótonos estrictamente crecientes. El frame se convierte de BGR a RGB
  antes de pasarlo al modelo.
- Por cada frame se obtiene una `PoseFrame` con:
  - `image_xy`: (33, 2) en píxeles, para dibujar y para reglas basadas en la vertical de la imagen;
  - `visibility`: (33,);
  - `world_xyz`: (33, 3) en metros, para calcular **ángulos 3D**, que no dependen de la distancia
    a la cámara.
- La imagen se espeja antes de detectar, así que los landmarks coinciden con lo que se ve en
  pantalla. Las etiquetas izquierda/derecha no importan: se elige el lado por visibilidad.

### 6.2 Poses de referencia como ángulos articulares

Cada ejercicio define *features*: ángulos en una articulación (vértice central de un trío de
landmarks), con un valor en **reposo** y otro en **máximo esfuerzo**:

| Ejercicio | Feature (trío de articulaciones) | Reposo | Máximo | Peso |
|-----------|----------------------------------|--------|--------|------|
| Dominada | codo (hombro-**codo**-muñeca) | 165º | 55º | 1.0 |
| Dominada | hombro (cadera-**hombro**-codo) | 165º | 45º | 0.7 |
| Flexión | codo (hombro-**codo**-muñeca) | 165º | 80º | 1.0 |
| Sentadilla búlgara | rodilla delantera (cadera-**rodilla**-tobillo) | 170º | 90º | 1.0 |
| Sentadilla búlgara | cadera delantera (hombro-**cadera**-rodilla) | 170º | 100º | 0.6 |

- **Selección de lado:** en la dominada y la flexión (`bilateral = True`) se promedian los lados
  cuya visibilidad media de las articulaciones necesarias es ≥ 0.5. En la sentadilla búlgara se
  mide sólo la **pierna adelantada**: la del tobillo más bajo en la imagen, porque el pie trasero
  está elevado. Tiene histéresis (8 % de la longitud de la pierna) para que no parpadee.
- **Suavizado:** EMA (media móvil exponencial) sobre los ángulos con α = 0.45 para reducir el
  ruido del detector.

### 6.3 Progreso (0 = reposo, 1 = máximo)

Cada ángulo se normaliza a `u = (ángulo - reposo) / (máximo - reposo)`. El progreso indica a cuál
de las dos poses de referencia se parece más la pose actual:

```
d_reposo = ‖u - 0‖_w        d_máximo = ‖u - 1‖_w        (distancias euclídeas ponderadas)
p = d_reposo / (d_reposo + d_máximo)
```

### 6.4 Conteo de repeticiones (máquina de estados con histéresis)

```
REPOSO ──(p > 0.25)──▶ MOVIENDO ──(p ≥ 0.75)──▶ PICO ──(p ≤ 0.25)──▶ ✔ REPETICIÓN (se puntúa)
                          │
                          └──(p ≤ 0.25 sin llegar a 0.75)──▶ "incompleta" si p_max ≥ 0.45 (si no, se ignora)
```

- La repetición se graba desde el último frame en reposo hasta el primer frame de vuelta al reposo.
- Si una repetición dura más de 15 s, se descarta.
- **Comprobación de postura:** si no se cumple, no se cuentan repeticiones y se avisa.
  - Dominada: las muñecas deben estar por encima de los hombros.
  - Flexión: el tronco debe estar a ≥ 45º de la vertical (horizontal).
  - Sentadilla búlgara: el tronco debe estar a ≤ 55º de la vertical (de pie).

### 6.5 Puntuación de cada repetición (0-100)

```
total = 100 · (0.35 · rango + 0.35 · trayectoria · tempo + 0.30 · técnica)
```

| Componente | Cálculo |
|------------|---------|
| **Rango** | `1 − min_t ‖clip(u_t, 0, 1) − 1‖_w`: lo cerca que se llegó a la pose de máximo esfuerzo. Pasarse (ir más profundo) no penaliza |
| **Trayectoria** | Parecido de la transición con la de referencia (ver abajo): `exp(−6 · max(error − 0.015, 0))` |
| **Tempo** | Factor 1 si la repetición dura entre 0,8 s y 8 s. Si dura menos, `duración / 0,8` (mínimo 0,4); si dura más, `8 / duración` (mínimo 0,5) |
| **Técnica** | `1 − media` de las penalizaciones (0-1) de las reglas del ejercicio, promediadas en los frames de la repetición |

**Trayectoria de referencia (alineada por fases).** La repetición se divide en el pico en dos
fases, **ida** (reposo → máximo) y **vuelta** (máximo → reposo). Cada fase se remuestrea a 20
muestras y se compara con el tramo correspondiente del perfil coseno ideal
`s(τ) = (1 − cos πτ) / 2`, que representa un arranque y un frenado suaves con todas las
articulaciones coordinadas. Como la grabación empieza al cruzar el umbral de reposo, el movimiento
ya lleva velocidad en ese punto, así que se compara con el *mismo tramo* de la curva ideal.
Antes de comparar, la trayectoria se normaliza por el pico alcanzado: la trayectoria mide la
**forma** del movimiento y el rango mide la profundidad.

- Alinear por fases permite que la ida y la vuelta duren distinto (p. ej. bajar lento y subir
  rápido), pero penaliza **parones, tirones y rebotes**.
- Se descartó un DTW (*Dynamic Time Warping*) libre porque deformaba el tiempo y "absorbía" esos
  defectos: daba ~0.86 tanto a movimientos ideales como a movimientos a tirones.

Resultados de referencia con datos sintéticos (ver [tests/](tests/)):

| Repetición | Puntuación |
|------------|-----------|
| Ideal | 96-100 |
| Muy rápida (0,5 s) | ~79 |
| A tirones o con rebotes | ~80 |
| Mala técnica (penalización 0.7) | ~73-76 |
| 55-70 % del recorrido | No cuenta (incompleta) |

**Reglas de técnica** (penalización lineal entre el valor "ok" y el valor "mal"):

| Ejercicio | Regla | Medida | Ok → mal |
|-----------|-------|--------|----------|
| Dominada | `barbilla` | Nariz respecto a la muñeca más alta (sólo con p ≥ 0.8) | a la altura → 0.3 × torso por debajo |
| Dominada | `simetria` | Diferencia entre codos izquierdo y derecho | 15º → 40º |
| Dominada | `balanceo` | Flexión de cadera (kipping / rodillas al pecho) | 150º → 115º |
| Flexión | `cuerpo` | Ángulo hombro-cadera-tobillo (cadera hundida o elevada) | 165º → 140º |
| Flexión | `rodillas` | Ángulo cadera-rodilla-tobillo | 160º → 130º |
| Flexión | `simetria` | Diferencia entre codos | 15º → 40º |
| Sentadilla búlgara | `torso` | Inclinación del tronco respecto a la vertical | 30º → 50º |

**Consejos:** se muestran tras cada repetición si:
- el rango es < 0.75;
- una regla tiene una penalización media ≥ 0.3;
- el tempo está fuera de rango;
- la trayectoria es < 0.6.

En vivo, se avisa cuando una regla supera 0.6 en el frame actual.

**Puntos del juego** = suma de las puntuaciones de las repeticiones: más repeticiones y con mejor
técnica dan más puntos.

---

## 7. Arquitectura del código

### Estructura de carpetas

```
IPM_Prac_2/
├── README.md
├── requirements.txt            # Dependencias fijadas (Python 3.11-3.12)
├── models/                     # Modelos .task descargados (no versionado)
├── results/                    # Sesiones guardadas en JSON (no versionado)
├── scripts/
│   ├── dependencies.sh / .bat  # Despliegue en equipo limpio: venv + requirements + modelos
│   ├── run.sh / .bat           # Lanza src/main.py con el Python del .venv
│   ├── download_models.py      # Descarga de modelos (--model, --force)
│   └── make_zip.py             # Genera el .zip de entrega con el nombre de los integrantes
├── tests/
│   └── test_logic.py           # Pruebas sin cámara (unittest)
└── src/
    ├── main.py                 # Punto de entrada: argumentos, modelo, cámara, detector y juego
    ├── config.py               # Rutas absolutas y todos los parámetros
    ├── vision/                 # Entrada de vídeo y detección de la pose
    │   ├── camera.py           #   Webcam o stream URL, espejo, control de fallos
    │   ├── models.py           #   Descarga atómica y localización de los modelos
    │   ├── detector.py         #   Envoltorio del Pose Landmarker de MediaPipe
    │   └── pose.py             #   Landmarks, geometría y PoseFrame
    ├── exercises/              # Definición de los ejercicios
    │   ├── base.py             #   Clase base Exercise
    │   ├── pullup.py, pushup.py, bulgarian_squat.py
    │   └── catalog.py          #   Registro EXERCISES y build_exercises()
    ├── analysis/               # Conteo y puntuación
    │   ├── counter.py          #   RepCounter (máquina de estados) + ExerciseTracker (pipeline por frame)
    │   └── scoring.py          #   Puntuación de cada rep y trayectoria de referencia
    ├── ui/                     # Interfaz
    │   ├── canvas.py           #   Paleta, lienzo escalable y texto Unicode
    │   └── screens.py          #   Pantallas del juego
    └── game/                   # Lógica del juego
        ├── game.py             #   Fases y bucle principal
        └── session.py          #   Resultados y guardado en JSON
```

### Flujo de datos por frame

```
Camera.read()            frame BGR (espejado)
   │
   ▼
PoseDetector.detect()    PoseFrame (image_xy, visibility, world_xyz)  | None si no hay persona
   │
   ▼
ExerciseTracker.update()
   ├─ Exercise.measure()          ángulos de los lados visibles     → Measurement | None
   ├─ EMA                         suavizado de ángulos
   ├─ Exercise.progress()         p ∈ [0, 1]
   ├─ Exercise.check_posture()    ¿en posición? (si no, no cuenta)
   ├─ Exercise.form_penalties()   {regla: 0..1}
   └─ RepCounter.update()         → RepEvent("rep", RepScore) | RepEvent("partial") | None
                                        └─ score_repetition()  (analysis/scoring.py)
   │
   ▼
Game._update()           avanza las fases y guarda las repeticiones en ExerciseSession
Game._draw()             Canvas + screens.*  → cv2.imshow
Game._handle_key()       un único cv2.waitKey por frame
```

### Dependencias entre paquetes (sin ciclos)

```
config ◀── vision.models, vision.camera
vision.pose ◀── vision.detector
config, vision.pose ◀── exercises ◀── analysis.scoring ◀── analysis.counter ◀── ui ◀── game ◀── main
```

### Clases principales

| Clase / función | Archivo | Responsabilidad |
|-----------------|---------|-----------------|
| `Settings`, `CameraConfig`, `ModelConfig`, `GameConfig`, `ScoringConfig` | `config.py` | Parámetros (dataclasses) y la instancia única `settings` |
| `ensure_model`, `download_model` | `vision/models.py` | Devuelve la ruta absoluta del modelo; descarga a `.part` y renombra (atómico) |
| `Camera` | `vision/camera.py` | `read()` → frame o `None`; lanza `RuntimeError` tras N fallos seguidos |
| `PoseDetector` | `vision/detector.py` | `detect(frame)` → `PoseFrame` o `None` |
| `PoseFrame` | `vision/pose.py` | Landmarks de un frame; `angle(a, b, c)` en 3D |
| `Landmark`, `side_landmark`, `POSE_CONNECTIONS` | `vision/pose.py` | Índices de los 33 puntos y conexiones del esqueleto |
| `Exercise`, `FeatureSpec`, `Measurement` | `exercises/base.py` | Clase base: medición, normalización, progreso, utilidades de reglas |
| `PullUp`, `PushUp`, `BulgarianSquat` | `exercises/*.py` | Referencias, postura y reglas de cada ejercicio |
| `EXERCISES`, `build_exercises` | `exercises/catalog.py` | Registro `clave → clase` |
| `RepCounter`, `RepEvent` | `analysis/counter.py` | Máquina de estados REPOSO/MOVIENDO/PICO |
| `ExerciseTracker`, `TrackerState` | `analysis/counter.py` | Pipeline por frame (medida → suavizado → progreso → postura → reglas → contador) |
| `score_repetition`, `RepScore` | `analysis/scoring.py` | Puntuación y consejos de una repetición |
| `trajectory_error`, `reference_trajectory` | `analysis/scoring.py` | Comparación alineada por fases con el perfil coseno |
| `Canvas` | `ui/canvas.py` | Dibujo en "unidades de diseño" (alto 720) escaladas a cualquier resolución |
| `TextRenderer` | `ui/canvas.py` | Texto con Pillow (tildes y eñes); una conversión BGR↔RGB por frame |
| `draw_*` | `ui/screens.py` | Menú, instrucciones, cuenta atrás, HUD, resultado y resumen |
| `Game`, `Phase` | `game/game.py` | Bucle principal y máquina de fases del juego |
| `ExerciseSession`, `save_sessions` | `game/session.py` | Resultados por ejercicio y guardado en JSON |

### Formato de los archivos

Todos los archivos `.py` siguen el mismo esquema, en este orden (sólo aparecen las secciones
que hacen falta):

```python
"""
carpeta/archivo.py  ·  Título breve.

Descripción de lo que contiene el archivo.
"""

# =============================================================================
# Importaciones
# =============================================================================
from __future__ import annotations

import os                       # 1) librería estándar
import numpy as np              # 2) terceros
from config import settings     # 3) del proyecto (absolutos, en orden alfabético)


# =============================================================================
# Constantes
# =============================================================================

# =============================================================================
# Tipos de datos              (dataclasses y enums)
# =============================================================================

# =============================================================================
# Funciones / Clases
# =============================================================================
class Ejemplo:
    # --- Subsección dentro de una clase --------------------------------------
    ...
```

- Cada `__init__.py` es sólo un docstring con la lista de archivos de la carpeta.
- Los separadores `# ===` marcan secciones del archivo y los `# ---` subsecciones dentro de
  una clase o función.

### Decisiones de diseño

- **Ángulos en vez de coordenadas:** no dependen de la posición, la escala ni la estatura del
  usuario.
- **Ángulos 3D (*world landmarks*):** son más tolerantes a que la cámara no esté perfectamente de
  lado. Se puede volver a 2D con `ModelConfig.use_world_landmarks = False`.
- **Modo `VIDEO` (síncrono)** en vez de `LIVE_STREAM`: cada resultado corresponde exactamente al
  frame que se dibuja y el código es más simple. El tracking entre frames sigue activo.
- **Histéresis en el conteo** (0.25 / 0.75) para que el ruido no cuente repeticiones dobles.
- **Rutas absolutas** calculadas desde `__file__` en `config.py`: el programa funciona
  igual se lance desde donde se lance. `main.py` añade `src/` a `sys.path` para que los paquetes
  se importen como módulos de primer nivel (`from config import settings`).
- **Texto con Pillow:** `cv2.putText` no soporta caracteres no ASCII.
- **Un único `cv2.waitKey` por frame:** el código base lo llamaba varias veces y se perdían
  pulsaciones.

---

## 8. Configuración

Todos los parámetros están en [src/config.py](src/config.py). Las referencias
y reglas de cada ejercicio están en su archivo de [src/exercises/](src/exercises/).

| Grupo | Parámetro | Valor | Significado |
|-------|-----------|-------|-------------|
| Rutas | `MODELS_DIR` | `<raíz>/models` | Se puede cambiar con la variable de entorno `POSE_GAME_MODELS_DIR` |
| Rutas | `RESULTS_DIR` | `<raíz>/results` | Se puede cambiar con la variable de entorno `POSE_GAME_RESULTS_DIR` |
| Cámara | `source`, `width`, `height`, `mirror` | `0`, 1280, 720, `True` | Fuente y resolución |
| Modelo | `variant` | `full` | `lite` / `full` / `heavy` |
| Modelo | `min_*_confidence` | 0.5 | Umbrales de MediaPipe |
| Modelo | `use_world_landmarks` | `True` | Ángulos 3D |
| Juego | `exercise_duration`, `target_reps` | 60 s, 10 | Fin de cada ejercicio |
| Juego | `ready_hold_seconds`, `ready_progress` | 1.5 s, 0.2 | Arranque automático en reposo |
| Juego | `min_visibility` | 0.5 | Visibilidad mínima de las articulaciones |
| Juego | `smoothing_alpha` | 0.45 | EMA (1 = sin suavizado) |
| Puntuación | `rest_threshold`, `peak_threshold`, `partial_threshold` | 0.25, 0.75, 0.45 | Conteo |
| Puntuación | `weight_rom`, `weight_trajectory`, `weight_form` | 0.35, 0.35, 0.30 | Pesos (se normalizan) |
| Puntuación | `trajectory_sensitivity`, `trajectory_noise_floor` | 6.0, 0.015 | Exigencia de la trayectoria |
| Puntuación | `min_rep_seconds`, `max_rep_seconds` | 0.8, 8.0 | Tempo |

**Calibración:** si al probar con cámara el juego no cuenta bien un ejercicio, ajusta primero
`rest` / `peak` de sus `FeatureSpec`. En la consola se imprime cada repetición con su desglose,
lo que sirve de ayuda.

---

## 9. Resultados guardados (formato JSON)

Cada partida completada se guarda en `results/session_AAAAMMDD_HHMMSS.json`. Ejemplo (valores
ilustrativos):

```json
{
  "date": "2026-11-11T11:58:30",
  "total_points": 296,
  "exercises": [
    {
      "exercise": "pushup", "name": "Flexión", "skipped": false,
      "reps": 3, "partial_reps": 1, "points": 296,
      "average_score": 98.5, "best_score": 100.0, "duration_s": 12.4,
      "repetitions": [
        {"total": 100.0, "rom": 1.0, "trajectory": 1.0, "form": 1.0, "tempo": 1.0,
         "duration": 1.4, "feedback": []}
      ]
    }
  ]
}
```

---

## 10. Pruebas

Hay pruebas que no necesitan cámara. Generan trayectorias y esqueletos sintéticos y recorren el
juego completo, del menú al resumen, sin abrir ventana:

```bash
.venv/bin/python -m unittest discover -s tests -v        # Windows: .venv\Scripts\python -m unittest discover -s tests -v
```

Qué comprueban:
- el progreso vale 0 / 0.5 / 1 en reposo / punto medio / máximo;
- una repetición ideal puntúa > 90;
- un recorrido corto cuenta como incompleta;
- una repetición rápida, a tirones o con mala técnica puntúa al menos 10 puntos menos y da el
  consejo adecuado;
- la sesión completa con una flexión sintética da 2 repeticiones, 1 incompleta y el JSON
  guardado.

---

## 11. Generar el .zip de entrega

```bash
.venv/bin/python scripts/make_zip.py "Nombre Apellido1 Apellido2" ["Nombre2 Apellido1 Apellido2"]
# -> nombre_apellido1_apellido2_nombre2_apellido1_apellido2.zip en la raíz del proyecto
```

El script quita tildes y espacios del nombre. Excluye `.venv/`, `models/`, `results/`, `.git/`,
cachés y `.task`: el receptor lo despliega con `scripts/dependencies.sh` o `.bat`.

---

## 12. Cómo añadir un ejercicio nuevo

1. Crea `src/exercises/mi_ejercicio.py` con una subclase de `Exercise`:

   ```python
   from exercises.base import Exercise, FeatureSpec

   class Sentadilla(Exercise):
       key = "squat"                       # Identificador para --exercises
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

   Opcionalmente se pueden sobreescribir `check_posture()` (impedir el conteo fuera de posición)
   y `active_sides()` (elegir el lado, como hace la sentadilla búlgara).
2. Regístralo en `EXERCISES` dentro de `src/exercises/catalog.py`.
3. Añádelo a `GameConfig.exercises` o pásalo con `--exercises squat`.
4. Añade un caso en `tests/test_logic.py`. Las pruebas de progreso y puntuación ideal recorren
   `EXERCISES` automáticamente.

---

## 13. Solución de problemas

| Problema | Solución |
|----------|----------|
| `se necesita Python 3.11 o 3.12` | Instala Python 3.12 (`sudo apt install python3.12 python3.12-venv`, o desde python.org en Windows) o indica uno: `PYTHON=/ruta/python3.12 bash scripts/dependencies.sh` |
| `Error creando el venv` | `sudo apt install python3-venv` (o `python3.12-venv`) |
| `No se pudo abrir la cámara '0'` | Comprueba que la cámara existe y que ningún otro programa la usa; prueba `--source 1`. En Linux, el usuario debe estar en el grupo `video` |
| La cámara del móvil no conecta | Misma red Wi-Fi; abre la URL en el navegador del portátil para comprobarla; usa la URL del *vídeo* (`/video`), no la de la página |
| Va lento | `--model lite` y/o `--width 640 --height 480` |
| No cuenta repeticiones | Mira el aviso de la pantalla (visibilidad / postura). Colócate como dice la tabla de colocación; que se vea todo el cuerpo necesario y haya buena luz |
| Cuenta repeticiones que no lo son, o no llega al máximo | Ajusta `rest` / `peak` del ejercicio o `peak_threshold` (sección 8) |
| Avisos de Qt / Wayland en la consola (Linux) | Son inofensivos; si la ventana no aparece, prueba `QT_QPA_PLATFORM=xcb bash scripts/run.sh` |
| Mensajes `W0000` / `INFO: Created TensorFlow Lite XNNPACK delegate` | Son logs normales de MediaPipe |

---

## 14. Notas para agentes de IA / desarrolladores

- **Punto de entrada único:** `src/main.py`. Lanzar siempre con el Python de `.venv`
  (`scripts/run.sh` / `run.bat`).
- **Imports:** siempre absolutos y del módulo concreto (`from vision.pose import PoseFrame`),
  porque `main.py`, `scripts/download_models.py` y `tests/test_logic.py` añaden `src/` a
  `sys.path`. Los `__init__.py` sólo tienen un docstring: no reexportan nada. No uses imports
  relativos (`from .base`).
- **Rutas:** usa siempre las constantes absolutas de `config` (`PROJECT_ROOT`, `MODELS_DIR`,
  `RESULTS_DIR`…). No uses rutas relativas al directorio de trabajo.
- **Sin ciclos de importación:** respeta el orden de dependencias de la sección 7. `config`,
  `vision/models.py` y `vision/pose.py` no deben importar `mediapipe` ni `cv2`, para que `download_models.py` sea ligero.
- **Coordenadas:** `PoseFrame.image_xy` está en píxeles del frame mostrado (ya espejado). La `y`
  crece hacia abajo. Los ángulos usan `world_xyz` si existe.
- **UI:** dibuja siempre a través de `Canvas` en unidades de diseño (alto = 720). El texto se
  encola y se pinta en `Canvas.finish()`. No uses `cv2.putText`, porque no soporta tildes.
- **Bucle:** un único `cv2.waitKey(1)` por frame, en `Game.run()`. Los timestamps del detector
  deben ser estrictamente crecientes (ya gestionado en `PoseDetector`).
- **Sin cámara:** para probar cambios, usa o amplía `tests/test_logic.py`. `Game._update()`,
  `Game._draw()` y `Game._handle_key()` se pueden llamar directamente con poses sintéticas y un
  reloj simulado (ver `TestGameFlow`).
- **Dependencias:** si se cambia `requirements.txt`, mantener la compatibilidad con Python
  3.11-3.12 y `numpy < 2` / `protobuf < 5` (restricciones de mediapipe 0.10.21).
- **No versionar:** `.venv/`, `models/`, `results/`, `*.zip` (ya están en `.gitignore`).
