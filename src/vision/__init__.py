"""
vision/  ·  Entrada de vídeo y detección de la pose.

Archivos:
    camera.py    Captura de la webcam o de la cámara del móvil.
    models.py    Descarga y localización de los modelos de MediaPipe.
    detector.py  Envoltorio del Pose Landmarker de MediaPipe.
    pose.py      Landmarks, geometría y la clase PoseFrame.

Este __init__ no importa nada a propósito: así se puede usar vision.pose o
vision.models sin cargar OpenCV ni MediaPipe. Importa siempre el módulo
concreto, p. ej. ``from vision.pose import PoseFrame``.
"""
