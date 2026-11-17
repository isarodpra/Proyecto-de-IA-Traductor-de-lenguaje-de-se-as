"""
Módulo de Extracción de Landmarks y Procesamiento Visual.
Utiliza MediaPipe Tasks (HandLandmarker y PoseLandmarker) con modelos locales
optimizados en 'modelos/' para detección articular precisa (21 puntos por mano, 33 de pose).
Incluye normalización relativa invariante a distancia y herramientas de dibujo sobre video.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np

from config import (
    TOTAL_CARACTERISTICAS,
    CANTIDAD_LANDMARKS_MANO,
    CANTIDAD_LANDMARKS_POSE,
    FRAME_ANCHO,
    FRAME_ALTO,
    USAR_FILTRO_CLAHE,
    NORMALIZAR_LANDMARKS,
    DIR_MODELOS
)

# Intentar importar MediaPipe Tasks
HAY_MEDIAPIPE_TASKS = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAY_MEDIAPIPE_TASKS = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")
except Exception:
    HAY_MEDIAPIPE_TASKS = False

# Conexiones estándar de las 21 articulaciones de la mano
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Meñique y palma
]

# Conexiones de pose corporal esencial (hombros, brazos y torso)
CONEXIONES_POSE = [
    (11, 12),               # Hombro a hombro
    (11, 13), (13, 15),     # Brazo izquierdo
    (12, 14), (14, 16),     # Brazo derecho
    (11, 23), (12, 24),     # Hombros a cadera
    (23, 24)                # Cadera
]

_CLAHE_CACHE = None


def aplicar_filtro_clahe(frame_bgr: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Mejora adaptativa de contraste local en el espacio LAB para baja iluminación."""
    global _CLAHE_CACHE
    try:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if _CLAHE_CACHE is None:
            _CLAHE_CACHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        cl = _CLAHE_CACHE.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame_bgr


def normalizar_puntos_mano(mano_flat: np.ndarray) -> np.ndarray:
    """
    Normaliza los 21 puntos clave 3D de una mano (vector de 63 floats):
    1. Centra la muñeca (landmark 0) en (0, 0, 0).
    2. Escala por la longitud de la palma (distancia muñeca a base del dedo medio lm 9).
    Hace el vector invariante a distancia y posición en la pantalla.
    """
    if len(mano_flat) < 63 or np.all(mano_flat == 0):
        return np.zeros(63, dtype=np.float32)

    pts = mano_flat.reshape(21, 3).copy()
    muñeca = pts[0, :].copy()
    pts_centrados = pts - muñeca

    dist_escala = np.linalg.norm(pts_centrados[9, :])
    if dist_escala > 1e-4:
        pts_centrados = pts_centrados / dist_escala

    return pts_centrados.flatten().astype(np.float32)


def normalizar_vector_landmarks(vector_completo: np.ndarray) -> np.ndarray:
    """Normaliza las manos izquierda y derecha dentro del vector de 225 características."""
    if len(vector_completo) != TOTAL_CARACTERISTICAS:
        return vector_completo

    vec = vector_completo.copy()
    # Mano izquierda: [0:63]
    if np.any(vec[:63] != 0):
        vec[:63] = normalizar_puntos_mano(vec[:63])

    # Mano derecha: [63:126]
    if np.any(vec[63:126] != 0):
        vec[63:126] = normalizar_puntos_mano(vec[63:126])

    return vec


def validar_calidad_landmarks(vector: np.ndarray, tipo: str = "estatica") -> Tuple[bool, str]:
    """Evalúa que haya al menos una mano detectada de forma nítida dentro de encuadre."""
    if vector is None or len(vector) < 126:
        return False, "Sin detección de manos"

    mano_izq = vector[:63]
    mano_der = vector[63:126]
    hay_izq = bool(np.any(mano_izq != 0))
    hay_der = bool(np.any(mano_der != 0))

    if not hay_izq and not hay_der:
        return False, "Muestre su mano claramente frente a la cámara"

    return True, "Excelente detección articular"


class ExtractorLandmarks:
    """Extractor moderno de puntos clave utilizando MediaPipe Tasks (Hands & Pose)."""

    def __init__(self, usar_mediapipe: bool = True, usar_clahe: bool = USAR_FILTRO_CLAHE):
        self.usar_clahe = usar_clahe
        self.usar_tasks = False
        self.hand_detector = None
        self.pose_detector = None

        ruta_hand_task = DIR_MODELOS / "hand_landmarker.task"
        ruta_pose_task = DIR_MODELOS / "pose_landmarker.task"

        if usar_mediapipe and HAY_MEDIAPIPE_TASKS and ruta_hand_task.exists():
            try:
                base_options_hand = python.BaseOptions(model_asset_path=str(ruta_hand_task))
                hand_options = vision.HandLandmarkerOptions(
                    base_options=base_options_hand,
                    num_hands=2,
                    min_hand_detection_confidence=0.45,
                    min_tracking_confidence=0.45
                )
                self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
                self.usar_tasks = True

                if ruta_pose_task.exists():
                    try:
                        base_options_pose = python.BaseOptions(model_asset_path=str(ruta_pose_task))
                        pose_options = vision.PoseLandmarkerOptions(
                            base_options=base_options_pose,
                            min_pose_detection_confidence=0.45,
                            min_tracking_confidence=0.45
                        )
                        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
                    except Exception:
                        self.pose_detector = None

                print("[OK] MediaPipe Tasks (HandLandmarker) inicializado exitosamente.")
            except Exception as e:
                print(f"[!] Error inicializando MediaPipe Tasks: {e}")
                self.usar_tasks = False

        self.modo_fallback = not self.usar_tasks

    def procesar_frame(self, frame_bgr: np.ndarray, normalizar: Optional[bool] = None) -> Tuple[np.ndarray, Any]:
        """Procesa un fotograma y extrae el vector 225D junto con los metadatos de detección."""
        if normalizar is None:
            normalizar = NORMALIZAR_LANDMARKS

        frame_proc = aplicar_filtro_clahe(frame_bgr) if self.usar_clahe else frame_bgr

        if self.usar_tasks and self.hand_detector:
            vector, results = self._procesar_mediapipe_tasks(frame_proc)
        else:
            vector, results = self._procesar_neutro(frame_proc)

        if normalizar:
            vector = normalizar_vector_landmarks(vector)

        return vector, results

    def _procesar_mediapipe_tasks(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Ejecuta inferencia neural de manos y pose sobre el frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            hand_result = self.hand_detector.detect(mp_image)
        except Exception:
            hand_result = None

        pose_result = None
        if self.pose_detector:
            try:
                pose_result = self.pose_detector.detect(mp_image)
            except Exception:
                pose_result = None

        mano_izq = np.zeros(CANTIDAD_LANDMARKS_MANO, dtype=np.float32)
        mano_der = np.zeros(CANTIDAD_LANDMARKS_MANO, dtype=np.float32)
        pose_arr = np.zeros(CANTIDAD_LANDMARKS_POSE, dtype=np.float32)

        # Asignar coordenadas de manos
        if hand_result and hand_result.hand_landmarks:
            for i, lms in enumerate(hand_result.hand_landmarks):
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
                es_izq = False
                if hand_result.handedness and i < len(hand_result.handedness):
                    lado = hand_result.handedness[i][0].category_name
                    es_izq = (lado.lower() == "left")
                else:
                    es_izq = (lms[0].x < 0.5)

                if es_izq and np.all(mano_izq == 0):
                    mano_izq = pts
                elif not es_izq and np.all(mano_der == 0):
                    mano_der = pts
                elif np.all(mano_izq == 0):
                    mano_izq = pts
                else:
                    mano_der = pts

        # Asignar pose corporal
        if pose_result and pose_result.pose_landmarks:
            p_lms = pose_result.pose_landmarks[0]
            pose_arr = np.array([[lm.x, lm.y, lm.z] for lm in p_lms], dtype=np.float32).flatten()
            if len(pose_arr) > CANTIDAD_LANDMARKS_POSE:
                pose_arr = pose_arr[:CANTIDAD_LANDMARKS_POSE]

        vector_completo = np.concatenate([mano_izq, mano_der, pose_arr]).astype(np.float32)
        if len(vector_completo) != TOTAL_CARACTERISTICAS:
            vector_completo = np.pad(vector_completo, (0, max(0, TOTAL_CARACTERISTICAS - len(vector_completo))))[:TOTAL_CARACTERISTICAS]

        info_results = {
            "tipo": "tasks",
            "hand_result": hand_result,
            "pose_result": pose_result
        }
        return vector_completo, info_results

    def _procesar_neutro(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Modo neutro seguro cuando no hay modelos de landmarks."""
        vector = np.zeros(TOTAL_CARACTERISTICAS, dtype=np.float32)
        return vector, {"tipo": "neutro"}

    def obtener_geometria_mano_rostro(self, results: Any) -> Dict[str, Any]:
        """Extrae métricas espaciales y de presencia de manos reales sobre el cuerpo."""
        datos = {
            "hay_mano": False,
            "hay_mano_izq": False,
            "hay_mano_der": False,
            "centroide_mano": None,
            "hay_rostro": False,
            "posicion_nariz": None,
            "altura_hombros": None,
            "mano_en_zona_activa": False,
            "distancia_mano_rostro": None
        }

        if isinstance(results, dict) and results.get("tipo") == "tasks":
            hand_res = results.get("hand_result")
            pose_res = results.get("pose_result")

            puntos_mano = []
            if hand_res and hand_res.hand_landmarks:
                datos["hay_mano"] = True
                for i, lms in enumerate(hand_res.hand_landmarks):
                    if hand_res.handedness and i < len(hand_res.handedness):
                        lado = hand_res.handedness[i][0].category_name
                        if lado.lower() == "left":
                            datos["hay_mano_izq"] = True
                        else:
                            datos["hay_mano_der"] = True
                    for lm in lms:
                        puntos_mano.append((lm.x, lm.y))

            if puntos_mano:
                arr_p = np.array(puntos_mano)
                cx = float(np.mean(arr_p[:, 0]))
                cy = float(np.mean(arr_p[:, 1]))
                datos["centroide_mano"] = (cx, cy)
                datos["mano_en_zona_activa"] = True

            if pose_res and pose_res.pose_landmarks:
                p_lms = pose_res.pose_landmarks[0]
                nariz = p_lms[0]
                datos["hay_rostro"] = True
                datos["posicion_nariz"] = (float(nariz.x), float(nariz.y))

                h_izq = p_lms[11]
                h_der = p_lms[12]
                datos["altura_hombros"] = float((h_izq.y + h_der.y) / 2.0)

                if datos["centroide_mano"] and datos["posicion_nariz"]:
                    mx, my = datos["centroide_mano"]
                    nx, ny = datos["posicion_nariz"]
                    datos["distancia_mano_rostro"] = float(np.sqrt((mx - nx)**2 + (my - ny)**2))

        return datos

    def dibujar_landmarks(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        """Dibuja el esqueleto articular completo y estético sobre las manos reales."""
        frame_copia = frame_bgr.copy()

        if isinstance(results, dict) and results.get("tipo") == "tasks":
            alto, ancho = frame_copia.shape[:2]
            hand_res = results.get("hand_result")
            pose_res = results.get("pose_result")

            # 1. Dibujar pose esencial (hombros, brazos y cabeza)
            if pose_res and pose_res.pose_landmarks:
                p_lms = pose_res.pose_landmarks[0]
                for idx1, idx2 in CONEXIONES_POSE:
                    if idx1 < len(p_lms) and idx2 < len(p_lms):
                        pt1 = (int(p_lms[idx1].x * ancho), int(p_lms[idx1].y * alto))
                        pt2 = (int(p_lms[idx2].x * ancho), int(p_lms[idx2].y * alto))
                        cv2.line(frame_copia, pt1, pt2, (180, 180, 180), 2, cv2.LINE_AA)

                for idx in [0, 11, 12, 13, 14, 15, 16]:
                    if idx < len(p_lms):
                        cx = int(p_lms[idx].x * ancho)
                        cy = int(p_lms[idx].y * alto)
                        cv2.circle(frame_copia, (cx, cy), 5, (0, 220, 255), -1, cv2.LINE_AA)

            # 2. Dibujar manos con esqueleto articular preciso
            if hand_res and hand_res.hand_landmarks:
                for h_idx, lms in enumerate(hand_res.hand_landmarks):
                    # Cian brillante para primera mano, magenta para segunda mano
                    color_huesos = (255, 220, 0) if h_idx == 0 else (220, 60, 220)
                    color_articulacion = (0, 255, 120)

                    # Líneas óseas
                    for idx1, idx2 in CONEXIONES_MANO:
                        pt1 = (int(lms[idx1].x * ancho), int(lms[idx1].y * alto))
                        pt2 = (int(lms[idx2].x * ancho), int(lms[idx2].y * alto))
                        cv2.line(frame_copia, pt1, pt2, color_huesos, 2, cv2.LINE_AA)

                    # Nudillos
                    for lm in lms:
                        cx, cy = int(lm.x * ancho), int(lm.y * alto)
                        cv2.circle(frame_copia, (cx, cy), 4, color_articulacion, -1, cv2.LINE_AA)
                        cv2.circle(frame_copia, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

                    # Etiqueta limpia sobre la muñeca
                    mx = int(lms[0].x * ancho)
                    my = int(lms[0].y * alto)
                    etiqueta = "Mano"
                    if hand_res.handedness and h_idx < len(hand_res.handedness):
                        etiqueta = f"Mano ({hand_res.handedness[h_idx][0].category_name})"
                    cv2.putText(frame_copia, etiqueta, (mx - 25, max(20, my - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_huesos, 2, cv2.LINE_AA)

        return frame_copia

    def cerrar(self):
        """Libera detectores de MediaPipe Tasks."""
        try:
            if self.hand_detector and hasattr(self.hand_detector, "close"):
                self.hand_detector.close()
            if self.pose_detector and hasattr(self.pose_detector, "close"):
                self.pose_detector.close()
        except Exception:
            pass


def generar_panel_derecho(
    frame_bgr: np.ndarray,
    results: Any,
    extractor: ExtractorLandmarks,
    usar_canny: bool = True
) -> np.ndarray:
    """Genera panel visual Canny neón o esqueleto sobre fondo negro."""
    alto, ancho = frame_bgr.shape[:2]

    if usar_canny:
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        difuminado = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(difuminado, threshold1=40, threshold2=120)

        panel = np.zeros((alto, ancho, 3), dtype=np.uint8)
        panel[:, :, 0] = bordes
        panel[:, :, 1] = bordes
        panel[:, :, 2] = (bordes * 0.4).astype(np.uint8)
        return panel
    else:
        lienzo_negro = np.zeros((alto, ancho, 3), dtype=np.uint8)
        return extractor.dibujar_landmarks(lienzo_negro, results)
