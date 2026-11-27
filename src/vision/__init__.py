"""
Entrada de vídeo y detección de la pose: camera.py (captura), models.py
(descarga de modelos), detector.py (MediaPipe) y pose.py (landmarks y geometría).

Este __init__ no importa nada para poder usar vision.pose o vision.models sin
cargar OpenCV ni MediaPipe. Importa siempre el módulo concreto.
"""
