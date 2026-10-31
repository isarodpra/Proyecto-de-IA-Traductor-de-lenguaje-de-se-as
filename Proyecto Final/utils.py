"""
Módulo de Utilidades Gráficas y Métricas.
Provee funciones para visualización HUD (Head-Up Display), medición de latencia/FPS,
y superposición de textos con fondos translúcidos.
"""

import cv2
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional


class MedidorRendimiento:
    """Calcula FPS reales y latencia promedio de procesamiento."""

    def __init__(self, ventana: int = 15):
        self.ventana = ventana
        self.tiempos_frame = []
        self.tiempos_inferencia = []
        self.ultimo_tiempo = time.time()

    def registrar_frame(self):
        ahora = time.time()
        delta = ahora - self.ultimo_tiempo
        self.ultimo_tiempo = ahora
        if delta > 0:
            self.tiempos_frame.append(delta)
            if len(self.tiempos_frame) > self.ventana:
                self.tiempos_frame.pop(0)

    def registrar_inferencia(self, duracion_ms: float):
        self.tiempos_inferencia.append(duracion_ms)
        if len(self.tiempos_inferencia) > self.ventana:
            self.tiempos_inferencia.pop(0)

    @property
    def fps(self) -> float:
        if not self.tiempos_frame:
            return 0.0
        return len(self.tiempos_frame) / sum(self.tiempos_frame)

    @property
    def latencia_ms(self) -> float:
        if not self.tiempos_inferencia:
            return 0.0
        return sum(self.tiempos_inferencia) / len(self.tiempos_inferencia)


def superponer_panel_translucido(
    frame: np.ndarray,
    x: int,
    y: int,
    ancho: int,
    alto: int,
    color_bgr: Tuple[int, int, int] = (20, 20, 20),
    alpha: float = 0.65
) -> np.ndarray:
    """Dibuja un rectángulo con transparencia sobre la imagen."""
    h, w = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ancho)
    y2 = min(h, y + alto)

    sub_img = frame[y1:y2, x1:x2]
    rect_color = np.full(sub_img.shape, color_bgr, dtype=np.uint8)
    cv2.addWeighted(rect_color, alpha, sub_img, 1 - alpha, 0, sub_img)
    return frame


def dibujar_texto_con_fondo(
    frame: np.ndarray,
    texto: str,
    pos: Tuple[int, int],
    escala: float = 0.6,
    color_texto: Tuple[int, int, int] = (255, 255, 255),
    color_fondo: Tuple[int, int, int] = (30, 30, 30),
    grosor: int = 1,
    padding: int = 6
) -> np.ndarray:
    """Renderiza texto sobre una placa rectangular con contraste."""
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    (ancho_t, alto_t), baseline = cv2.getTextSize(texto, fuente, escala, grosor)
    x, y = pos

    x1 = x - padding
    y1 = y - alto_t - padding
    x2 = x + ancho_t + padding
    y2 = y + baseline + padding

    superponer_panel_translucido(frame, x1, y1, x2 - x1, y2 - y1, color_fondo, alpha=0.7)
    cv2.putText(frame, texto, (x, y), fuente, escala, color_texto, grosor, cv2.LINE_AA)
    return frame


def dibujar_barra_confianza(
    frame: np.ndarray,
    x: int,
    y: int,
    ancho: int,
    alto: int,
    porcentaje: float,
    umbral: float = 0.70
) -> None:
    """Dibuja una barra de progreso que cambia de color según el umbral."""
    porcentaje_clamped = max(0.0, min(1.0, porcentaje))
    ancho_progreso = int(ancho * porcentaje_clamped)

    # Fondo barra
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (50, 50, 50), -1)

    # Color verde si supera el umbral, amarillo/naranja si está por debajo
    if porcentaje >= umbral:
        color = (0, 220, 100)
    elif porcentaje >= umbral * 0.7:
        color = (0, 180, 255)
    else:
        color = (50, 50, 220)

    cv2.rectangle(frame, (x, y), (x + ancho_progreso, y + alto), color, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (180, 180, 180), 1)

    # Marca del umbral
    x_umbral = x + int(ancho * umbral)
    cv2.line(frame, (x_umbral, y - 2), (x_umbral, y + alto + 2), (255, 255, 255), 1)


def unir_paneles(panel_izquierdo: np.ndarray, panel_derecho: np.ndarray) -> np.ndarray:
    """Concatena horizontalmente dos paneles asegurando misma altura y agrega una línea divisoria."""
    h1, w1 = panel_izquierdo.shape[:2]
    h2, w2 = panel_derecho.shape[:2]

    # Igualar alturas si es necesario
    if h1 != h2:
        panel_derecho = cv2.resize(panel_derecho, (w2, h1))

    combinado = np.hstack((panel_izquierdo, panel_derecho))

    # Línea divisoria central
    cv2.line(combinado, (w1, 0), (w1, h1), (60, 60, 60), 2)
    return combinado
