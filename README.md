# MiEntrenadorPersonal: juego de entrenamiento con visión por computador

**Práctica 2 de IPM (Interacción Persona-Máquina) mediante visión por computador.**

MiEntrenadorPersonal es un **juego serio** (*serious game*) del ámbito de la salud. Usa la cámara del
portátil o del móvil (o un **vídeo grabado**) y el modelo **MediaPipe Pose Landmarker** para detectar los 33 puntos del
cuerpo, y guía al usuario en tres ejercicios de fuerza con su propio peso:

1. **Dominada**
2. **Flexión** 
3. **Sentadilla búlgara** 

El juego **cuenta las repeticiones automáticamente** y **puntúa la técnica de cada una (0-100)**
comparando el movimiento del usuario con una pose de reposo, una pose de máximo esfuerzo y una
trayectoria de referencia entre ambas. Es la misma idea que los juegos de "dibuja un círculo":
cuanto más se parece tu movimiento a la referencia, más puntos.

- **Repetición de prueba (calibración):** antes de cada ejercicio el usuario hace una repetición
  lenta y completa, y de ella salen *sus* poses de reposo y de máximo y *su* trayectoria. Las
  repeticiones siguientes se comparan con esa referencia personal.
- **Gráfica de cada repetición** frente a la de referencia, en vivo y al final del ejercicio.
- Consejos en tiempo real ("mantén el cuerpo recto", "baja más"…), sonidos e **historial** de
  partidas con su evolución.
- Interfaz con **botones** (ratón) además del teclado, selector de **cámara o vídeo** y de
  ejercicios en el menú, y opción de **grabar la ventana** en un `.mp4`.

> Requisito de uso: **cámara estática** (apoyada, sin moverse durante el ejercicio). Vale igual
> para grabar un vídeo y pasárselo al juego con `--video` (ver [Jugar con un vídeo grabado](#jugar-con-un-vídeo-grabado)).

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
| Cámara o vídeo | Webcam, móvil con una app que emita la cámara por red (ver [Cámara del móvil](#cámara-del-móvil)) o un vídeo grabado (ver [Jugar con un vídeo grabado](#jugar-con-un-vídeo-grabado)) |
| Conexión a internet | Sólo en la instalación (paquetes pip y modelos `.task`, ~46 MB los tres) |
| Opcional: sonido | Funciona tal cual en Windows y macOS. En Linux: `sudo apt install libportaudio2` (o tener `paplay`/`aplay`). Sin audio, el juego funciona igual |
| Opcional: explorador de archivos | Para el botón "Buscar en el ordenador…". En Windows y macOS viene con Python. En Linux: `sudo apt install python3.12-tk` (o `python3-tk`) |
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
| `--source X` | `0` | Índice de webcam (`0`, `1`…), URL de stream (cámara del móvil) o ruta de un vídeo |
| `--video ARCHIVO` | — | Juega con un vídeo grabado en vez de la cámara (`.mp4`, `.avi`, `.mov`…) |
| `--model {lite,full,heavy}` | `full` | `lite` es más rápido (equipos lentos) y `heavy` más preciso |
| `--exercises ...` | `pullup pushup bulgarian_squat` | Ejercicios y orden |
| `--reps N` | `10` | Repeticiones objetivo por ejercicio (termina al alcanzarlas) |
| `--duration S` | `60` | Segundos máximos por ejercicio |
| `--width W --height H` | `1280 720` | Resolución pedida a la webcam |
| `--no-mirror` | — | No mostrar la imagen en espejo (por defecto sólo se espeja la cámara, no los vídeos) |
| `--mirror` | — | Espejar también un vídeo grabado |
| `--start S` | `0` | Con un vídeo: empezar en el segundo `S` (para saltarse el principio) |
| `--no-calibrate` | — | Sin repetición de prueba: usar las referencias estándar (también se cambia en el menú) |
| `--record [ARCHIVO]` | — | Grabar la ventana en un `.mp4` (por defecto `results/grabacion_*.mp4`). También con la tecla `G` |
| `--no-sound` | — | Sin efectos de sonido |
| `--no-download` | — | No descargar el modelo si falta (da error) |
| `--no-save` | — | No guardar resultados en `results/` |

Ejemplos:

```bash
bash scripts/run.sh --model lite
bash scripts/run.sh --exercises pushup bulgarian_squat --reps 8 --duration 45
bash scripts/run.sh --video videos/flexiones.mp4 --exercises pushup --start 3 --record
```

Casi todo se puede elegir también **desde el menú**, sin escribir opciones: la fuente (cámara
o vídeo), los ejercicios y si se hace la repetición de prueba. Si no hay cámara, el juego
arranca igual y pide abrir un vídeo.

### Cámara del móvil

Instala en el móvil una app que emita la cámara por la red local, por ejemplo **IP Webcam**
(Android) o **DroidCam** (Android/iOS). Conecta el móvil y el portátil a la misma red Wi-Fi y
pasa la URL del vídeo:

```bash
bash scripts/run.sh --source http://192.168.1.20:8080/video     # IP Webcam
bash scripts/run.sh --source http://192.168.1.20:4747/video     # DroidCam
```

### Jugar con un vídeo grabado

En vez de la cámara se le puede pasar un vídeo ya grabado. **El juego funciona exactamente igual
que en vivo**: se abre la ventana, se pulsa `ENTER` y la persona del vídeo sigue las indicaciones
de la pantalla (ponerse en reposo, esperar la cuenta atrás, hacer las repeticiones). Sirve, por
ejemplo, para el vídeo de demostración de la práctica.

```bash
bash scripts/run.sh --video demo.mp4 --exercises pushup               # sólo flexiones
bash scripts/run.sh --video demo.mp4                                  # los tres ejercicios seguidos
scripts\run.bat --video demo.mp4 --exercises pushup bulgarian_squat   # Windows
```

**Sin terminal:** lanza el juego sin opciones (`bash scripts/run.sh`, o doble clic en
`scripts\run.bat`) y en el menú pulsa **Abrir vídeo… (V)**. Aparecen los vídeos de la carpeta
[`videos/`](videos/) del proyecto (cópialos ahí) y un botón **Buscar en el ordenador…** que abre
el explorador de archivos. Luego elige los ejercicios y pulsa **EMPEZAR**.

- El vídeo se queda **en pausa en el menú** (se ve el primer frame) y **empieza al pulsar
  EMPEZAR** (`ENTER` o clic). Con `--start 5` empieza en el segundo 5.
- Los vídeos verticales (del móvil) se centran sobre un fondo oscuro, y los muy grandes (4K) se
  reducen, para que la interfaz quepa y el juego vaya fluido.
- Se reproduce a su **velocidad real**. Si el equipo no llega a procesar todos los frames, se
  saltan frames para no retrasarse, igual que pasaría con la cámara.
- El **reloj del juego es el del vídeo**: la cuenta atrás, el tiempo del ejercicio y el tempo de
  cada repetición son exactos aunque el equipo vaya lento.
- `P` **pausa / reanuda** el vídeo. En pausa el juego también se congela (el tiempo no corre).
- Arriba a la derecha se ve la posición del vídeo (`VÍDEO 0:12 / 1:05`).
- Cuando **se acaba el vídeo**, se cierra el ejercicio en curso con lo conseguido. Los ejercicios
  que falten se marcan como saltados y se llega al resumen. Si se acaba durante las
  instrucciones, pulsa `N` para saltar.
- `R` o "jugar otra vez" en el resumen **rebobinan el vídeo** y vuelven al menú.
- Por defecto el vídeo **no se espeja** (se ve tal cual se grabó); `--mirror` lo espeja.

**Cómo grabar el vídeo:**

1. Cámara quieta y colocada como dice la [tabla de colocación](#colocación-para-cada-ejercicio)
   (de lado para la flexión y la sentadilla búlgara), con el cuerpo entero en el encuadre y buena luz.
2. Empieza a grabar unos segundos antes de colocarte: en el juego, pulsar `ENTER` en el menú
   arranca el vídeo desde el principio.
3. Sigue el flujo del juego: colócate en **reposo** y quédate quieto ~2 s (el juego necesita
   1,5 s de reposo para empezar), espera los **3 s de cuenta atrás**, haz **1 repetición de
   prueba lenta y completa** (la calibración; si la desactivas, sáltate este paso), vuelve al
   reposo un momento y haz las repeticiones.
   Como referencia puedes ejecutar el juego con la cámara mientras grabas con el móvil, y así
   ves las indicaciones en tiempo real.
4. Si grabas varios ejercicios en el mismo vídeo, hazlos en el orden de `--exercises`. Entre uno
   y otro, el juego espera en la pantalla de resultado hasta que pulsas `ENTER`, y el vídeo sigue
   corriendo mientras tanto: deja unos segundos de margen o pulsa `P` para pausar. Si no, es más
   cómodo grabar **un vídeo por ejercicio**.

**Para la presentación:** `--record` (o la tecla `G`) graba la ventana del juego en
`results/grabacion_*.mp4` mientras juegas. Si el equipo va lento al grabar la pantalla, puedes
poner esa grabación en su lugar. Se abre con VLC o cualquier reproductor que lea MPEG-4.

---

## 5. Cómo se juega

### Flujo

```
           ┌── SELECTOR DE VÍDEO (V)
MENÚ ──────┤
  │        └── HISTORIAL (H)
  │ EMPEZAR
  ▼
┌─ por cada ejercicio ───────────────────────────────────────────────────────────┐
│ INSTRUCCIONES ──(reposo mantenido 1,5 s o ENTER)──▶ CUENTA ATRÁS (3 s)        │
│      │                                                  │                     │
│      N (saltar)                                          ▼                     │
│      │                     REPETICIÓN DE PRUEBA (si la calibración está activa)│
│      │                       1 rep lenta → tu reposo, tu máximo, tu trayectoria│
│      │                       (ENTER = usar la referencia estándar)             │
│      │                                                  ▼                     │
│      ▼                                   ACTIVO (hasta 10 reps o 60 s)         │
│ RESULTADO (con la gráfica de tu mejor repetición) ◀─────┘                      │
└───────┬────────────────────────────────────────────────────────────────────────┘
        ▼
    RESUMEN (puntos totales, comparación con la partida anterior, se guarda el JSON)
```

1. **Menú**: elige la **fuente** (cámara o vídeo), los **ejercicios** (casillas) y si se hace
   la **repetición de prueba**, y pulsa **EMPEZAR**. Desde aquí también se abre el **historial**.
2. **Instrucciones**: se explica el ejercicio y dónde poner la cámara. Colócate en la **posición
   de reposo**. Cuando el sistema te detecta en ella, un anillo se llena durante 1,5 s y empieza
   solo (o pulsa `ENTER`).
3. **Cuenta atrás** de 3 s, con pitidos.
4. **Repetición de prueba** (si está activada): haz 1 repetición lenta y completa. El juego
   guarda tu reposo, tu máximo y cómo te has movido, y te avisa con "¡Referencia guardada!".
   Si no llegas al menos a la mitad del recorrido estándar, te pide repetirla. Tienes 20 s;
   si no, o si pulsas `ENTER`, se usa la referencia estándar.
5. **Ejercicio**. En pantalla aparecen:
   - repeticiones / objetivo, tiempo restante y puntos;
   - una **barra vertical de progreso** reposo → máximo, con las marcas de los umbrales;
   - el esqueleto, con las articulaciones medidas resaltadas (azul = reposo, verde = máximo);
   - la puntuación de la **última repetición** desglosada (rango, trayectoria, técnica) y su
     **gráfica** frente a la referencia (línea discontinua);
   - **avisos en vivo** (técnica, postura incorrecta, persona fuera de cámara);
   - tras cada repetición, un mensaje ("¡Perfecta!", "+82", "Incompleta"…) con consejos.
6. **Resultado del ejercicio**: repeticiones, puntos, medias por componente, los consejos más
   repetidos y la **gráfica de tu mejor repetición** (con las demás en gris) frente a la
   referencia.
7. **Resumen**: tabla de los ejercicios, la puntuación total y la diferencia con la partida
   anterior.
8. **Historial**: gráficas de los puntos y de la puntuación media de las últimas partidas, y
   una tabla con las más recientes.

### Controles

Todas las pantallas tienen **botones** que se pulsan con el ratón; cada botón indica entre
paréntesis su tecla.

| Tecla | Acción |
|-------|--------|
| `ENTER` / `ESPACIO` | Empezar / continuar / terminar el ejercicio actual / usar la referencia estándar |
| `1`, `2`, `3` (menú) | Activar o desactivar cada ejercicio |
| `C` / `V` (menú) | Usar la cámara / abrir un vídeo |
| `K` (menú) | Activar o desactivar la repetición de prueba |
| `H` (menú y resumen) | Historial |
| `1`…`8` / `B` (selector de vídeo) | Elegir un vídeo de la lista / buscar en el ordenador |
| `P` | Pausar / reanudar el vídeo (también con clic en el indicador del vídeo) |
| `G` | Empezar / parar la grabación de la ventana |
| `N` | Saltar ejercicio |
| `F` | Pantalla completa |
| `R` | Volver al menú (reinicia la partida; con un vídeo, lo rebobina) |
| `ESC` | Volver (en el selector de vídeo y el historial) o salir |
| `Q` / cerrar ventana | Salir |

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

Esos son los valores **estándar**. Con la repetición de prueba se sustituyen por los del propio
usuario (ver [6.6](#66-repetición-de-prueba-calibración)).

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

### 6.6 Repetición de prueba (calibración)

Antes de cada ejercicio, tras la cuenta atrás, el usuario hace una repetición lenta y completa
([src/analysis/calibration.py](src/analysis/calibration.py)):

1. **Reposo:** mediana de los ángulos mientras espera quieto en las instrucciones y en la cuenta
   atrás.
2. **Repetición:** se detecta con las referencias estándar, igual que el contador: empieza al
   pasar `p > 0.25` y acaba al volver a `p ≤ 0.25`. Tiene que llegar al menos a `p ≥ 0.5`; si
   se queda corta, se pide repetirla.
3. **Máximo:** media de los ángulos en los frames de máximo progreso.
4. **Comprobación por ángulo:** cada ángulo tiene que moverse en el mismo sentido que el
   estándar y al menos un 40 % de su recorrido. Si no (p. ej. el hombro en una dominada mal
   hecha), ese ángulo se queda con la referencia estándar y se avisa en pantalla. Si falla el
   ángulo principal, se pide repetir.
5. **Trayectoria de referencia:** la ida y la vuelta de la repetición se remuestrean, se fuerzan
   monótonas (para quitar el ruido del detector) y se escalan de 0 a 1. Sustituyen al perfil
   coseno ideal en la comparación de la [sección 6.5](#65-puntuación-de-cada-repetición-0-100).
   Si una fase de un ángulo apenas se mueve, se usa el coseno.

A partir de ahí el progreso, el rango y la trayectoria se miden respecto a **tu** reposo, **tu**
máximo y **tu** forma de moverte. Si desactivas la calibración (menú o `--no-calibrate`), se
usan las referencias estándar y el perfil coseno.

### 6.7 Gráfica de la repetición

Cada `RepScore` guarda dos curvas (40 puntos, ida y vuelta): `curve`, el progreso real con su
profundidad, y `reference_curve`, la referencia con la que se ha comparado. Se dibujan en el HUD
(última repetición) y en el resultado (mejor repetición, con las demás en gris). En el eje
vertical, abajo es el reposo y arriba el máximo.

---

## 7. Arquitectura del código

### Estructura de carpetas

```
IPM_Prac_2/
├── README.md
├── requirements.txt            # Dependencias fijadas (Python 3.11-3.12)
├── models/                     # Modelos .task descargados (no versionado)
├── results/                    # Sesiones guardadas en JSON y grabaciones (no versionado)
├── videos/                     # Vídeos para el selector del menú (los vídeos no se versionan)
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
    │   ├── camera.py           #   Webcam, stream URL o vídeo grabado; espejo, control de fallos
    │   ├── models.py           #   Descarga atómica y localización de los modelos
    │   ├── detector.py         #   Envoltorio del Pose Landmarker de MediaPipe
    │   └── pose.py             #   Landmarks, geometría y PoseFrame
    ├── exercises/              # Definición de los ejercicios
    │   ├── base.py             #   Clase base Exercise
    │   ├── pullup.py, pushup.py, bulgarian_squat.py
    │   └── catalog.py          #   Registro EXERCISES y build_exercises()
    ├── analysis/               # Conteo, puntuación y calibración
    │   ├── counter.py          #   RepCounter (máquina de estados) + ExerciseTracker (pipeline por frame)
    │   ├── scoring.py          #   Puntuación de cada rep y trayectoria de referencia
    │   └── calibration.py      #   Calibrator: referencias a partir de la repetición de prueba
    ├── ui/                     # Interfaz
    │   ├── canvas.py           #   Paleta, lienzo escalable y texto Unicode
    │   ├── widgets.py          #   Botones, casillas y gráficas
    │   ├── screens.py          #   Pantallas del juego
    │   ├── dialogs.py          #   Lista de vídeos y explorador de archivos (tkinter, en otro proceso)
    │   ├── sound.py            #   Pitidos generados (sounddevice / winsound / paplay)
    │   └── recorder.py         #   Grabación de la ventana en .mp4
    └── game/                   # Lógica del juego
        ├── game.py             #   Fases, acciones (teclado y ratón) y bucle principal
        ├── session.py          #   Resultados y guardado en JSON
        └── history.py          #   Lectura del historial de partidas
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
Game._update()           avanza las fases, calibra (Calibrator) y guarda las repeticiones en ExerciseSession
Game._draw()             Canvas + screens.* (los botones se registran en Canvas.buttons) → cv2.imshow
Game._handle_key()       un único cv2.waitKey por frame  ─┐
Game._handle_click()     clic → botón bajo el ratón      ─┴─▶ Game._do(acción)
```

### Dependencias entre paquetes (sin ciclos)

```
config ◀── vision.models, vision.camera
vision.pose ◀── vision.detector
config, vision.pose ◀── exercises ◀── analysis.scoring ◀── analysis.counter, analysis.calibration ◀── ui ◀── game ◀── main
```

### Clases principales

| Clase / función | Archivo | Responsabilidad |
|-----------------|---------|-----------------|
| `Settings`, `CameraConfig`, `ModelConfig`, `GameConfig`, `ScoringConfig` | `config.py` | Parámetros (dataclasses) y la instancia única `settings` |
| `ensure_model`, `download_model` | `vision/models.py` | Devuelve la ruta absoluta del modelo; descarga a `.part` y renombra (atómico) |
| `Camera` | `vision/camera.py` | `read()` → frame o `None`; lanza `RuntimeError` tras N fallos seguidos. Con un vídeo: `play()`, `pause()`, `rewind()`, `clock()` (reloj del vídeo) y `ended` |
| `NoSource`, `fit_frame` | `vision/camera.py` | Fuente vacía para arrancar sin cámara; adaptación de frames verticales o muy grandes |
| `PoseDetector` | `vision/detector.py` | `detect(frame)` → `PoseFrame` o `None` |
| `PoseFrame` | `vision/pose.py` | Landmarks de un frame; `angle(a, b, c)` en 3D |
| `Landmark`, `side_landmark`, `POSE_CONNECTIONS` | `vision/pose.py` | Índices de los 33 puntos y conexiones del esqueleto |
| `Exercise`, `FeatureSpec`, `Measurement` | `exercises/base.py` | Clase base: medición, normalización, progreso, utilidades de reglas. `set_reference()` / `reset_reference()` cambian entre referencias calibradas y estándar |
| `PullUp`, `PushUp`, `BulgarianSquat` | `exercises/*.py` | Referencias, postura y reglas de cada ejercicio |
| `EXERCISES`, `build_exercises` | `exercises/catalog.py` | Registro `clave → clase` |
| `RepCounter`, `RepEvent` | `analysis/counter.py` | Máquina de estados REPOSO/MOVIENDO/PICO |
| `ExerciseTracker`, `TrackerState` | `analysis/counter.py` | Pipeline por frame (medida → suavizado → progreso → postura → reglas → contador) |
| `score_repetition`, `RepScore` | `analysis/scoring.py` | Puntuación y consejos de una repetición |
| `trajectory_error`, `reference_trajectory`, `build_profile` | `analysis/scoring.py` | Comparación alineada por fases con el perfil coseno o el calibrado |
| `Calibrator`, `CalibrationResult` | `analysis/calibration.py` | Repetición de prueba → reposo, máximo y perfil del usuario |
| `Canvas` | `ui/canvas.py` | Dibujo en "unidades de diseño" (alto 720) escaladas a cualquier resolución |
| `TextRenderer` | `ui/canvas.py` | Texto con Pillow (tildes y eñes); una conversión BGR↔RGB por frame |
| `button`, `checkbox`, `draw_rep_chart`, `draw_line_chart` | `ui/widgets.py` | Botones clicables y gráficas |
| `draw_*`, `MenuInfo`, `PickerInfo` | `ui/screens.py` | Menú, selector de vídeo, historial, instrucciones, cuenta atrás, repetición de prueba, HUD, resultado y resumen |
| `Sound` | `ui/sound.py` | Efectos de sonido sin bloquear (y en silencio si no hay audio) |
| `Recorder` | `ui/recorder.py` | Graba la ventana en `.mp4` a fps fijos según el reloj real |
| `list_videos`, `ask_video_file` | `ui/dialogs.py` | Vídeos de `videos/` y explorador de archivos |
| `Game`, `Phase` | `game/game.py` | Bucle principal y máquina de fases. `_do(acción)` es el único punto por el que pasan teclas y clics |
| `ExerciseSession`, `save_sessions` | `game/session.py` | Resultados por ejercicio y guardado en JSON |
| `load_history`, `HistoryEntry` | `game/history.py` | Partidas guardadas, para el historial y la comparación |

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
| Rutas | `VIDEOS_DIR` | `<raíz>/videos` | Carpeta del selector de vídeo (`POSE_GAME_VIDEOS_DIR`) |
| Cámara | `source`, `width`, `height` | `0`, 1280, 720 | Fuente y resolución (los frames más altos que `height` se reducen) |
| Cámara | `mirror`, `mirror_video` | `True`, `False` | Espejo con la cámara / con un vídeo |
| Cámara | `start_seconds` | 0 | Segundo en el que empieza un vídeo |
| Modelo | `variant` | `full` | `lite` / `full` / `heavy` |
| Modelo | `min_*_confidence` | 0.5 | Umbrales de MediaPipe |
| Modelo | `use_world_landmarks` | `True` | Ángulos 3D |
| Juego | `exercise_duration`, `target_reps` | 60 s, 10 | Fin de cada ejercicio |
| Juego | `ready_hold_seconds`, `ready_progress` | 1.5 s, 0.2 | Arranque automático en reposo |
| Juego | `min_visibility` | 0.5 | Visibilidad mínima de las articulaciones |
| Juego | `smoothing_alpha` | 0.45 | EMA (1 = sin suavizado) |
| Juego | `calibrate`, `calibration_timeout` | `True`, 20 s | Repetición de prueba y tiempo para hacerla |
| Juego | `calibration_min_progress`, `calibration_min_span` | 0.5, 0.4 | Profundidad mínima de la repetición de prueba y recorrido mínimo de cada ángulo (fracción del estándar) |
| Juego | `sound`, `record_path`, `record_fps` | `True`, `None`, 30 | Sonido y grabación de la ventana |
| Puntuación | `rest_threshold`, `peak_threshold`, `partial_threshold` | 0.25, 0.75, 0.45 | Conteo |
| Puntuación | `weight_rom`, `weight_trajectory`, `weight_form` | 0.35, 0.35, 0.30 | Pesos (se normalizan) |
| Puntuación | `trajectory_sensitivity`, `trajectory_noise_floor` | 6.0, 0.015 | Exigencia de la trayectoria |
| Puntuación | `min_rep_seconds`, `max_rep_seconds` | 0.8, 8.0 | Tempo |

**Ajuste de las referencias estándar:** si al probar con cámara (sin repetición de prueba) el
juego no cuenta bien un ejercicio, ajusta primero `rest` / `peak` de sus `FeatureSpec`. En la consola se imprime cada repetición con su desglose,
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
  guardado;
- con un archivo de vídeo: empieza en pausa, avanza al darle a play, el reloj se congela en
  pausa, detecta el final y se rebobina;
- el juego con un vídeo: arranca el vídeo con `ENTER`, la pausa congela la cuenta atrás, al
  acabarse el vídeo cierra el ejercicio y salta los que faltan, y "jugar otra vez" lo rebobina;
- la calibración: aprende un reposo y un máximo personales, pide repetir una repetición de
  prueba demasiado corta, deja en estándar un ángulo que no se mueve, y el perfil de una
  repetición ideal coincide con el coseno;
- el juego con repetición de prueba: instrucciones → cuenta atrás → prueba → ejercicio, con la
  referencia personal guardada en el JSON; y que se puede saltar con la referencia estándar;
- el menú (casillas, calibración, fuente, sin fuente no se empieza), el clic en un botón y que
  todas las pantallas se dibujan en 16:9 y 4:3;
- el historial, la grabación de la ventana, el sonido (en silencio), la adaptación de frames
  verticales y `--start`.

---

## 11. Generar el .zip de entrega

```bash
.venv/bin/python scripts/make_zip.py "Nombre Apellido1 Apellido2" ["Nombre2 Apellido1 Apellido2"]
# -> nombre_apellido1_apellido2_nombre2_apellido1_apellido2.zip en la raíz del proyecto
```

El script quita tildes y espacios del nombre. Excluye `.venv/`, `models/`, `results/`, `.git/`,
cachés, `.task` y los vídeos de `videos/` (pesan mucho: entrégalos aparte si hace falta): el receptor lo despliega con `scripts/dependencies.sh` o `.bat`.

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
| `No se pudo abrir el vídeo` | Comprueba la ruta. Si el formato no lo lee OpenCV, conviértelo a `.mp4` (H.264), p. ej. `ffmpeg -i entrada.mov salida.mp4` |
| Con un vídeo no empieza el ejercicio | La persona del vídeo debe estar en reposo ~1,5 s durante las instrucciones; si no, pulsa `ENTER` para empezar la cuenta atrás a mano |
| La repetición de prueba no se acepta ("Otra vez") | Tiene que llegar al menos a la mitad del recorrido estándar y volver al reposo. Si no se puede, pulsa `ENTER` para usar la referencia estándar, o desactívala en el menú (`K`) |
| Puntúa demasiado alto con repeticiones cortas | La referencia es tu repetición de prueba: si fue corta, se compara con eso. Haz la prueba completa o desactiva la calibración |
| `No hay salida de audio disponible` | En Linux: `sudo apt install libportaudio2`. El juego funciona igual sin sonido (`--no-sound` quita el aviso) |
| "Buscar en el ordenador…" desactivado | Falta tkinter. En Linux: `sudo apt install python3.12-tk`. Mientras tanto, copia los vídeos en `videos/` |
| La grabación `.mp4` no se abre | Ábrela con VLC (usa el códec MPEG-4 Part 2) |
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
  deben ser estrictamente crecientes (ya gestionado en `PoseDetector`, también al rebobinar un vídeo).
- **Reloj:** usa siempre `Game._now()` y no `time.monotonic()`: con un vídeo es el reloj del
  vídeo (se congela en pausa). Sólo se detecta y se llama a `_update()` con frames nuevos
  (`Camera.frame_id`).
- **Acciones:** teclas y clics acaban en `Game._do(acción)`. Un botón nuevo se dibuja con
  `widgets.button(c, "acción", ...)` y se trata en `_do()`; no hace falta tocar el ratón.
- **Referencias:** no modifiques `rest_vector` / `peak_vector` a mano; usa
  `Exercise.set_reference()` / `reset_reference()`. Cada partida empieza con las estándar.
- **Sin cámara:** para probar cambios, usa o amplía `tests/test_logic.py`. `Game._update()`,
  `Game._draw()`, `Game._handle_key()` y `Game._do()` se pueden llamar directamente con poses
  sintéticas y un reloj simulado (ver `TestGameFlow` y `TestCalibratedGame`).
- **Dependencias:** si se cambia `requirements.txt`, mantener la compatibilidad con Python
  3.11-3.12 y `numpy < 2` / `protobuf < 5` (restricciones de mediapipe 0.10.21).
- **No versionar:** `.venv/`, `models/`, `results/`, `*.zip` y los vídeos de `videos/` (ya
  están en `.gitignore`).
