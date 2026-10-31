"""
Módulo de Extracción de Landmarks y Procesamiento Visual.
Soporta MediaPipe (Holistic / Hands / Pose) con fallback automático basado en
OpenCV Haar Cascades y análisis de contornos normalizados para garantizar funcionamiento offline.
Genera además vistas complementarias (filtro Canny o renderizado de esqueleto).
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from config import (
    TOTAL_CARACTERISTICAS,
    CANTIDAD_LANDMARKS_MANO,
    CANTIDAD_LANDMARKS_POSE,
    FRAME_ANCHO,
    FRAME_ALTO
)

# Intentar importar MediaPipe
HAY_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAY_MEDIAPIPE = True
except (ImportError, Exception):
    HAY_MEDIAPIPE = False


class ExtractorLandmarks:
    """Clase unificada para extracción de puntos clave corporales y de manos."""

    def __init__(self, usar_mediapipe: bool = True):
        self.usar_mediapipe = usar_mediapipe and HAY_MEDIAPIPE
        self.modo_fallback = not self.usar_mediapipe

        if self.usar_mediapipe:
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # Fallback: Haar Cascade frontal de OpenCV
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def procesar_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Procesa el frame y devuelve:
        - vector de características 1D de longitud fija (TOTAL_CARACTERISTICAS).
        - objeto de resultados para dibujo posterior.
        """
        if self.usar_mediapipe:
            return self._procesar_mediapipe(frame_bgr)
        else:
            return self._procesar_fallback(frame_bgr)

    def _procesar_mediapipe(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Extraer mano izquierda (21 landmarks * 3 = 63)
        if results.left_hand_landmarks:
            mano_izq = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten()
        else:
            mano_izq = np.zeros(CANTIDAD_LANDMARKS_MANO)

        # Extraer mano derecha (21 landmarks * 3 = 63)
        if results.right_hand_landmarks:
            mano_der = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten()
        else:
            mano_der = np.zeros(CANTIDAD_LANDMARKS_MANO)

        # Extraer puntos de pose corporal (33 landmarks * 3 = 99)
        if results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            ).flatten()
        else:
            pose = np.zeros(CANTIDAD_LANDMARKS_POSE)

        vector_completo = np.concatenate([mano_izq, mano_der, pose]).astype(np.float32)

        # Asegurar longitud estricta
        if len(vector_completo) != TOTAL_CARACTERISTICAS:
            vector_completo = np.pad(
                vector_completo,
                (0, max(0, TOTAL_CARACTERISTICAS - len(vector_completo)))
            )[:TOTAL_CARACTERISTICAS]

        return vector_completo, results

    def _procesar_fallback(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Fallback cuando MediaPipe no está instalado.
        Detecta rostro y regiones de mano usando segmentación por tono de piel HSV y contornos.
        Produce un vector normalizado de tamaño TOTAL_CARACTERISTICAS.
        """
        alto, ancho, _ = frame_bgr.shape
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Detección de rostro
        caras = self.face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        cara_box = caras[0] if len(caras) > 0 else None

        # Máscara de tono de piel en HSV
        bajo_piel = np.array([0, 30, 60], dtype=np.uint8)
        alto_piel = np.array([25, 255, 255], dtype=np.uint8)
        mascara_piel = cv2.inRange(hsv, bajo_piel, alto_piel)
        mascara_piel = cv2.GaussianBlur(mascara_piel, (5, 5), 0)

        # Si hay cara detectada, enmascararla para buscar las manos fuera del rostro
        if cara_box is not None:
            fx, fy, fw, fh = cara_box
            cv2.rectangle(mascara_piel, (fx, fy), (fx + fw, fy + fh), 0, -1)

        contornos, _ = cv2.findContours(mascara_piel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Ordenar contornos por área descendente
        contornos_validos = sorted([c for c in contornos if cv2.contourArea(c) > 400], key=cv2.contourArea, reverse=True)

        features = []
        # Normalizar datos de cara
        if cara_box is not None:
            features.extend([cara_box[0] / ancho, cara_box[1] / alto, cara_box[2] / ancho, cara_box[3] / alto])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Extraer puntos característicos de los 2 mayores contornos (manos)
        for i in range(2):
            if i < len(contornos_validos):
                c = contornos_validos[i]
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = (M["m10"] / M["m00"]) / ancho
                    cy = (M["m01"] / M["m00"]) / alto
                else:
                    cx, cy = 0.0, 0.0
                bx, by, bw, bh = cv2.boundingRect(c)
                features.extend([cx, cy, bx / ancho, by / alto, bw / ancho, bh / alto])

                # Muestrear puntos del contorno
                puntos = c[:, 0, :]
                if len(puntos) >= 10:
                    indices = np.linspace(0, len(puntos) - 1, 15, dtype=int)
                    muestreados = puntos[indices]
                    for px, py in muestreados:
                        features.extend([px / ancho, py / alto, 0.0])
                else:
                    features.extend([0.0] * 45)
            else:
                features.extend([0.0] * 51)

        # Rellenar o recortar al tamaño TOTAL_CARACTERISTICAS
        vector_fallback = np.array(features, dtype=np.float32)
        if len(vector_fallback) < TOTAL_CARACTERISTICAS:
            vector_fallback = np.pad(vector_fallback, (0, TOTAL_CARACTERISTICAS - len(vector_fallback)))
        else:
            vector_fallback = vector_fallback[:TOTAL_CARACTERISTICAS]

        info_fallback = {
            "cara": cara_box,
            "manos": [cv2.boundingRect(c) for c in contornos_validos[:2]]
        }
        return vector_fallback, info_fallback

    def dibujar_landmarks(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        """Dibuja los puntos sobre el frame original."""
        frame_copia = frame_bgr.copy()
        if self.usar_mediapipe and results is not None:
            # Dibujar manos
            if getattr(results, 'left_hand_landmarks', None):
                self.mp_drawing.draw_landmarks(
                    frame_copia,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            if getattr(results, 'right_hand_landmarks', None):
                self.mp_drawing.draw_landmarks(
                    frame_copia,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            # Dibujar pose esencial
            if getattr(results, 'pose_landmarks', None):
                self.mp_drawing.draw_landmarks(
                    frame_copia,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
        elif self.modo_fallback and isinstance(results, dict):
            # Dibujar información del modo fallback
            if results.get("cara") is not None:
                x, y, w, h = results["cara"]
                cv2.rectangle(frame_copia, (x, y), (x + w, y + h), (255, 120, 0), 2)
                cv2.putText(frame_copia, "Rostro", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
            for i, (mx, my, mw, mh) in enumerate(results.get("manos", [])):
                cv2.rectangle(frame_copia, (mx, my), (mx + mw, my + mh), (0, 255, 128), 2)
                cv2.putText(frame_copia, f"Mano {i+1}", (mx, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        return frame_copia

    def cerrar(self):
        """Libera recursos del procesador."""
        if self.usar_mediapipe and hasattr(self, 'holistic'):
            self.holistic.close()


def generar_panel_derecho(
    frame_bgr: np.ndarray,
    results: Any,
    extractor: ExtractorLandmarks,
    usar_canny: bool = True
) -> np.ndarray:
    """
    Genera el panel derecho para la visualización en ventana doble:
    - Si usar_canny=True: Aplica filtro Canny con bordes estilizados en color neón/cian.
    - Si usar_canny=False: Renderiza el esqueleto de landmarks sobre lienzo negro.
    """
    alto, ancho = frame_bgr.shape[:2]

    if usar_canny:
        # Preprocesar para Canny
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        difuminado = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(difuminado, threshold1=40, threshold2=120)

        # Crear panel RGB con bordes coloreados (cian/verde tecnológico)
        panel = np.zeros((alto, ancho, 3), dtype=np.uint8)
        panel[bordes > 0] = (240, 200, 30)  # Cian/Celeste tecnológico

        # Agregar encabezado discreto
        cv2.putText(panel, "VISION ARTIFICIAL: FILTRO CANNY", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return panel
    else:
        # Esqueleto sobre fondo oscuro
        panel = np.zeros((alto, ancho, 3), dtype=np.uint8)
        if extractor.usar_mediapipe and results is not None:
            if getattr(results, 'left_hand_landmarks', None):
                extractor.mp_drawing.draw_landmarks(
                    panel,
                    results.left_hand_landmarks,
                    extractor.mp_holistic.HAND_CONNECTIONS,
                    extractor.mp_drawing_styles.get_default_hand_landmarks_style(),
                    extractor.mp_drawing_styles.get_default_hand_connections_style()
                )
            if getattr(results, 'right_hand_landmarks', None):
                extractor.mp_drawing.draw_landmarks(
                    panel,
                    results.right_hand_landmarks,
                    extractor.mp_holistic.HAND_CONNECTIONS,
                    extractor.mp_drawing_styles.get_default_hand_landmarks_style(),
                    extractor.mp_drawing_styles.get_default_hand_connections_style()
                )
            if getattr(results, 'pose_landmarks', None):
                extractor.mp_drawing.draw_landmarks(
                    panel,
                    results.pose_landmarks,
                    extractor.mp_holistic.POSE_CONNECTIONS
                )
        cv2.putText(panel, "ESQUELETO LANDMARKS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1, cv2.LINE_AA)
        return panel
