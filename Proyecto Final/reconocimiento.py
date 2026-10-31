"""
Módulo de Reconocimiento y Traducción en Tiempo Real con Máquina de Estados y Cooldown.
Implementa una máquina de estados deliberada:
- Detección de presencia de mano y reposo.
- Confirmación sostenida para posturas estáticas (Hold timer).
- Captura de trayectoria para gestos dinámicos.
- Tiempo de enfriamiento/espera (Cooldown) tras cada detección para asimilar y dar tiempo.
- Panel de traducción multilínea con word-wrapping y autoscroll.
"""

import cv2
import json
import time
import joblib
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import numpy as np

from config import (
    CAMARA_INDEX,
    FRAME_ANCHO,
    FRAME_ALTO,
    FRAMES_POR_SECUENCIA,
    TOTAL_CARACTERISTICAS,
    CANTIDAD_LANDMARKS_MANO,
    UMBRAL_CONFIANZA,
    VENTANA_SUAVIZADO,
    UMBRAL_MOVIMIENTO_DINAMICO,
    UMBRAL_PRESENCIA_MANO,
    TIEMPO_CONFIRMACION_ESTATICA,
    TIEMPO_COOLDOWN_POST_DETECCION,
    MODO_AUTO,
    MODO_SOLO_ESTATICO,
    MODO_SOLO_DINAMICO,
    RUTA_ETIQUETAS
)
import db
from landmarks import ExtractorLandmarks, generar_panel_derecho
from traduccion import TraductorSecuencias
from utils import (
    MedidorRendimiento,
    dibujar_barra_confianza,
    superponer_panel_translucido,
    unir_paneles,
    dibujar_panel_traduccion_multilinea
)
from entrenamiento import (
    RUTA_MODELO_JOBLIB,
    extraer_caracteristicas_dinamicas
)


class EstadosReconocedor:
    REPOSO = "REPOSO"
    LISTO = "LISTO"
    CONFIRMANDO_ESTATICA = "CONFIRMANDO_ESTATICA"
    CAPTURANDO_DINAMICA = "CAPTURANDO_DINAMICA"
    COOLDOWN = "COOLDOWN"


class ReconocedorEnVivo:
    """Motor de inferencia continua con máquina de estados y tiempo de asimilación."""

    def __init__(self):
        self.extractor = ExtractorLandmarks()
        self.traductor = TraductorSecuencias()
        self.medidor = MedidorRendimiento()

        # Buffers temporales
        self.buffer_landmarks = deque(maxlen=FRAMES_POR_SECUENCIA)
        self.buffer_manos_previas = deque(maxlen=8)

        # Modos de operación
        self.modos_disponibles = [MODO_AUTO, MODO_SOLO_ESTATICO, MODO_SOLO_DINAMICO]
        self.indice_modo = 0

        # Modelos y metadatos
        self.modelo_estatico = None
        self.clases_estaticas: List[str] = []
        self.modelo_dinamico = None
        self.clases_dinamicas: List[str] = []
        self.metadatos = {}

        # Máquina de estados
        self.estado_actual = EstadosReconocedor.REPOSO
        self.postura_en_observacion: Optional[str] = None
        self.tiempo_inicio_postura: float = 0.0

        # Trayectoria dinámica
        self.secuencia_dinamica_en_curso = []

        # Cooldown tras detección
        self.tiempo_inicio_cooldown: float = 0.0
        self.ultima_seña_confirmada: Optional[str] = None
        self.ultima_confianza: float = 0.0

        self._cargar_modelos()

    @property
    def modo_actual(self) -> str:
        return self.modos_disponibles[self.indice_modo]

    def alternar_modo(self):
        """Alterna entre Auto, Solo Estático y Solo Dinámico."""
        self.indice_modo = (self.indice_modo + 1) % len(self.modos_disponibles)
        return self.modo_actual

    def _cargar_modelos(self):
        """Carga el modelo híbrido desacoplado desde disco una sola vez."""
        if not RUTA_MODELO_JOBLIB.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en '{RUTA_MODELO_JOBLIB}'. "
                "Por favor entrene el modelo desde la opción 2 del menú."
            )

        print(f"[*] Cargando modelo híbrido desde '{RUTA_MODELO_JOBLIB.name}'...")
        paquete = joblib.load(str(RUTA_MODELO_JOBLIB))

        self.modelo_estatico = paquete.get("modelo_estatico")
        self.clases_estaticas = paquete.get("clases_estaticas", [])
        self.modelo_dinamico = paquete.get("modelo_dinamico")
        self.clases_dinamicas = paquete.get("clases_dinamicas", [])
        self.metadatos = paquete.get("metadatos", {})

        print(f"[OK] Modelo cargado: {len(self.clases_estaticas)} estáticas, {len(self.clases_dinamicas)} dinámicas.")

    def evaluar_presencia_y_movimiento(self, vector_landmarks: np.ndarray) -> Tuple[bool, float]:
        """Calcula presencia y velocidad de las manos entre frames."""
        landmarks_manos = vector_landmarks[:CANTIDAD_LANDMARKS_MANO * 2]
        magnitud_manos = float(np.mean(np.abs(landmarks_manos)))

        mano_presente = magnitud_manos > UMBRAL_PRESENCIA_MANO

        if not mano_presente:
            self.buffer_manos_previas.clear()
            return False, 0.0

        self.buffer_manos_previas.append(landmarks_manos)

        if len(self.buffer_manos_previas) < 3:
            return True, 0.0

        diffs = []
        lista_m = list(self.buffer_manos_previas)
        for i in range(1, len(lista_m)):
            diffs.append(np.mean(np.abs(lista_m[i] - lista_m[i - 1])))

        velocidad_promedio = float(np.mean(diffs))
        return True, velocidad_promedio

    def clasificar_frame_estatico(self, vector_landmarks: np.ndarray) -> Tuple[Optional[str], float]:
        """Clasifica la postura estática instantánea."""
        if not self.modelo_estatico or not self.clases_estaticas:
            return None, 0.0

        t0 = time.time()
        proba = self.modelo_estatico.predict_proba([vector_landmarks])[0]
        self.medidor.registrar_inferencia((time.time() - t0) * 1000.0)

        idx_max = int(np.argmax(proba))
        confianza = float(proba[idx_max])

        if confianza >= UMBRAL_CONFIANZA:
            return self.clases_estaticas[idx_max], confianza
        return None, confianza

    def clasificar_secuencia_dinamica(self, secuencia: List[np.ndarray]) -> Tuple[Optional[str], float]:
        """Clasifica la trayectoria dinámica completa."""
        if not self.modelo_dinamico or not self.clases_dinamicas:
            return None, 0.0

        arr_sec = np.array(secuencia, dtype=np.float32)
        if len(arr_sec) < FRAMES_POR_SECUENCIA:
            arr_sec = np.pad(arr_sec, ((0, FRAMES_POR_SECUENCIA - len(arr_sec)), (0, 0)), mode='edge')
        else:
            arr_sec = arr_sec[-FRAMES_POR_SECUENCIA:]

        features = extraer_caracteristicas_dinamicas(arr_sec)
        t0 = time.time()
        proba = self.modelo_dinamico.predict_proba([features])[0]
        self.medidor.registrar_inferencia((time.time() - t0) * 1000.0)

        idx_max = int(np.argmax(proba))
        confianza = float(proba[idx_max])

        if confianza >= UMBRAL_CONFIANZA:
            return self.clases_dinamicas[idx_max], confianza
        return None, confianza

    def ejecutar_bucle_vivo(self):
        """Bucle interactivo con máquina de estados, cooldown y panel multilínea."""
        cap = cv2.VideoCapture(CAMARA_INDEX)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {CAMARA_INDEX}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_ANCHO)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_ALTO)

        modo_canny = True
        nombre_ventana = "Reconocimiento y Traduccion de Lenguaje de Senas en Vivo"
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 60)
        print("       RECONOCIMIENTO EN VIVO INICIADO")
        print(" [m]        : Alternar MODO (AUTO <-> SOLO ESTÁTICO <-> SOLO DINÁMICO)")
        print(" [c]        : Alternar panel derecho (Canny <-> Esqueleto)")
        print(" [ESPACIO]  : Guardar oración actual en base de datos SQLite")
        print(" [x] / [b]  : Borrar última palabra/letra")
        print(" [l]        : Limpiar toda la oración")
        print(" [q] / [ESC]: Salir al menú")
        print("=" * 60 + "\n")

        registro_predicciones_sesion = []

        try:
            while True:
                ahora = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                self.medidor.registrar_frame()
                frame = cv2.flip(frame, 1)

                # Extraer landmarks y geometría espacial explícita mano-rostro
                vector_caracteristicas, results = self.extractor.procesar_frame(frame)
                geometria = self.extractor.obtener_geometria_mano_rostro(results)

                hay_mano = geometria["hay_mano"]
                hay_rostro = geometria["hay_rostro"]
                centroide_mano = geometria["centroide_mano"]
                mano_en_zona_activa = geometria["mano_en_zona_activa"]

                _, velocidad_movimiento = self.evaluar_presencia_y_movimiento(vector_caracteristicas)

                # Variables para renderizado HUD
                texto_estado_hud = "LISTO"
                color_estado_hud = (200, 200, 200)
                progreso_barra = 0.0
                texto_barra = ""
                mostrar_barra = False

                # -------------------------------------------------------------
                # MÁQUINA DE ESTADOS DEL RECONOCIMIENTO
                # -------------------------------------------------------------

                # 1. ESTADO: COOLDOWN (Espera y asimilación tras confirmar seña)
                if self.estado_actual == EstadosReconocedor.COOLDOWN:
                    tiempo_transcurrido = ahora - self.tiempo_inicio_cooldown
                    tiempo_restante = max(0.0, TIEMPO_COOLDOWN_POST_DETECCION - tiempo_transcurrido)
                    progreso_cooldown = min(1.0, tiempo_transcurrido / TIEMPO_COOLDOWN_POST_DETECCION)

                    texto_estado_hud = f"DETECTADO: '{self.ultima_seña_confirmada}'"
                    color_estado_hud = (0, 255, 120)
                    mostrar_barra = True
                    progreso_barra = 1.0 - progreso_cooldown
                    texto_barra = f"Esperando siguiente sena ({tiempo_restante:.1f}s)..."

                    # Al finalizar el cooldown, resetear buffers y volver a LISTO o REPOSO
                    if tiempo_transcurrido >= TIEMPO_COOLDOWN_POST_DETECCION:
                        self.estado_actual = EstadosReconocedor.LISTO if hay_mano else EstadosReconocedor.REPOSO
                        self.postura_en_observacion = None
                        self.tiempo_inicio_postura = 0.0
                        self.secuencia_dinamica_en_curso.clear()
                        self.buffer_manos_previas.clear()
                        if hasattr(self, 'trayectoria_centroides_mano'):
                            self.trayectoria_centroides_mano.clear()

                # 2. ESTADO: REPOSO (Sin mano en pantalla - Regla: La mano es primaria y obligatoria)
                elif not hay_mano:
                    self.estado_actual = EstadosReconocedor.REPOSO
                    self.traductor.notificar_reposo()
                    self.postura_en_observacion = None
                    self.secuencia_dinamica_en_curso.clear()
                    if hasattr(self, 'trayectoria_centroides_mano'):
                        self.trayectoria_centroides_mano.clear()

                    if hay_rostro:
                        texto_estado_hud = "ROSTRO DETECTADO: Levante la mano para signar"
                        color_estado_hud = (120, 200, 255)
                    else:
                        texto_estado_hud = "REPOSO: Coloque su mano en el encuadre"
                        color_estado_hud = (130, 130, 130)

                # 3. ESTADOS ACTIVOS (Mano presente y en pantalla)
                else:
                    if not hasattr(self, 'trayectoria_centroides_mano'):
                        self.trayectoria_centroides_mano = []

                    # Determinar si procede evaluar estático o dinámico
                    preferir_estatico = (self.modo_actual == MODO_SOLO_ESTATICO) or (
                        self.modo_actual == MODO_AUTO and velocidad_movimiento <= UMBRAL_MOVIMIENTO_DINAMICO
                    )
                    preferir_dinamico = (self.modo_actual == MODO_SOLO_DINAMICO) or (
                        self.modo_actual == MODO_AUTO and velocidad_movimiento > UMBRAL_MOVIMIENTO_DINAMICO
                    )

                    # RAMA DINÁMICA: Movimiento activo de la MANO
                    if preferir_dinamico and self.clases_dinamicas:
                        self.estado_actual = EstadosReconocedor.CAPTURANDO_DINAMICA
                        self.secuencia_dinamica_en_curso.append(vector_caracteristicas)
                        if centroide_mano is not None:
                            self.trayectoria_centroides_mano.append(centroide_mano)

                        progreso_din = min(1.0, len(self.secuencia_dinamica_en_curso) / FRAMES_POR_SECUENCIA)

                        texto_estado_hud = "MOVIMIENTO DE MANO: Analizando trayectoria..."
                        color_estado_hud = (0, 160, 255)
                        mostrar_barra = True
                        progreso_barra = progreso_din
                        texto_barra = f"Capturando gesto dinamico: {int(progreso_din * 100)}%"

                        # Al culminar la trayectoria de 30 frames
                        if len(self.secuencia_dinamica_en_curso) >= FRAMES_POR_SECUENCIA:
                            # 1. VERIFICACIÓN CINEMÁTICA ESTRICTA DE LA MANO
                            pts_mano = np.array([p for p in self.trayectoria_centroides_mano if p is not None])
                            movimiento_mano_suficiente = False
                            if len(pts_mano) >= 12:
                                # Desplazamiento acumulado entre frames
                                dist_acumulada_mano = float(np.sum(np.sqrt(np.sum(np.diff(pts_mano, axis=0)**2, axis=1))))
                                rango_x = float(np.max(pts_mano[:, 0]) - np.min(pts_mano[:, 0]))
                                rango_y = float(np.max(pts_mano[:, 1]) - np.min(pts_mano[:, 1]))
                                amplitud_mano = rango_x + rango_y

                                # Exigir que la mano se haya desplazado de verdad (amplitud real de saludo)
                                if dist_acumulada_mano >= 0.12 and amplitud_mano >= 0.07:
                                    movimiento_mano_suficiente = True

                            # 2. VERIFICACIÓN DE POSICIÓN RELATIVA AL ROSTRO
                            # Para gestos como HOLA, la mano debe estar en zona activa cerca del rostro
                            posicion_valida = mano_en_zona_activa and (hay_rostro or geometria["distancia_mano_rostro"] is not None)

                            if movimiento_mano_suficiente and posicion_valida:
                                seña_din, conf_din = self.clasificar_secuencia_dinamica(self.secuencia_dinamica_en_curso)
                                if seña_din and not seña_din.startswith("_"):
                                    palabra = self.traductor.procesar_gesto_dinamico(seña_din, conf_din)
                                    if palabra:
                                        registro_predicciones_sesion.append({
                                            "sena": seña_din,
                                            "palabra": palabra,
                                            "confianza": round(conf_din, 3),
                                            "tipo": "dinamica",
                                            "timestamp": ahora
                                        })
                                        # Activar Cooldown
                                        self.estado_actual = EstadosReconocedor.COOLDOWN
                                        self.tiempo_inicio_cooldown = ahora
                                        self.ultima_seña_confirmada = seña_din
                                        self.ultima_confianza = conf_din
                                else:
                                    self.estado_actual = EstadosReconocedor.LISTO
                            else:
                                # Si no hubo movimiento suficiente de la mano, no disparar HOLA
                                self.estado_actual = EstadosReconocedor.LISTO

                            self.secuencia_dinamica_en_curso.clear()
                            self.trayectoria_centroides_mano.clear()

                    # RAMA ESTÁTICA: Mano quieta/firme en postura
                    elif preferir_estatico and self.clases_estaticas:
                        self.secuencia_dinamica_en_curso.clear()
                        seña_detectada, conf = self.clasificar_frame_estatico(vector_caracteristicas)

                        if seña_detectada:
                            if seña_detectada != self.postura_en_observacion:
                                # Cambio a nueva postura: iniciar cronómetro de fijación
                                self.postura_en_observacion = seña_detectada
                                self.tiempo_inicio_postura = ahora
                                self.estado_actual = EstadosReconocedor.CONFIRMANDO_ESTATICA

                            # Calcular cuánto tiempo se ha mantenido la postura fija
                            duracion_hold = ahora - self.tiempo_inicio_postura
                            progreso_hold = min(1.0, duracion_hold / TIEMPO_CONFIRMACION_ESTATICA)

                            texto_estado_hud = f"POSTURA: {seña_detectada}"
                            color_estado_hud = (0, 230, 255)
                            mostrar_barra = True
                            progreso_barra = progreso_hold
                            texto_barra = f"Confirmando letra: {int(progreso_hold * 100)}%"

                            # Si se mantuvo firme durante todo el tiempo de confirmación
                            if duracion_hold >= TIEMPO_CONFIRMACION_ESTATICA:
                                letra = self.traductor.procesar_seña_estatica(seña_detectada, conf)
                                if letra:
                                    registro_predicciones_sesion.append({
                                        "sena": seña_detectada,
                                        "palabra": letra,
                                        "confianza": round(conf, 3),
                                        "tipo": "estatica",
                                        "timestamp": ahora
                                    })
                                    # Activar Cooldown tras confirmación
                                    self.estado_actual = EstadosReconocedor.COOLDOWN
                                    self.tiempo_inicio_cooldown = ahora
                                    self.ultima_seña_confirmada = seña_detectada
                                    self.ultima_confianza = conf
                        else:
                            self.estado_actual = EstadosReconocedor.LISTO
                            self.postura_en_observacion = None
                            self.tiempo_inicio_postura = 0.0
                            texto_estado_hud = "MANO LISTA (Defina una postura fija o realice un gesto)"
                            color_estado_hud = (180, 210, 120)

                # -------------------------------------------------------------
                # RENDERIZADO VISUAL
                # -------------------------------------------------------------
                frame_izq = self.extractor.dibujar_landmarks(frame, results)
                alto_f, ancho_f = frame_izq.shape[:2]

                # 1. Panel Superior HUD
                superponer_panel_translucido(frame_izq, 10, 10, 460, 140, (22, 22, 22), 0.80)

                cv2.putText(frame_izq, f"FPS: {self.medidor.fps:.1f} | Latencia: {self.medidor.latencia_ms:.1f} ms",
                            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1, cv2.LINE_AA)

                cv2.putText(frame_izq, f"MODO: [{self.modo_actual}] (tecla 'm')",
                            (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 210, 80), 1, cv2.LINE_AA)

                cv2.putText(frame_izq, texto_estado_hud, (20, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_estado_hud, 2, cv2.LINE_AA)

                # Barra de confirmación / Cooldown
                if mostrar_barra:
                    cv2.putText(frame_izq, texto_barra, (20, 114),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                    dibujar_barra_confianza(frame_izq, 20, 122, 420, 12, progreso_barra, umbral=0.99)
                else:
                    cv2.putText(frame_izq, f"Velocidad: {velocidad_movimiento:.3f} | Umbral mov: {UMBRAL_MOVIMIENTO_DINAMICO:.3f}",
                                (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

                # 2. Panel Inferior de Traducción Multilínea (Word-wrapping + Auto-scroll)
                oracion_completa = self.traductor.obtener_oracion_actual()
                dibujar_panel_traduccion_multilinea(frame_izq, oracion_completa, alto_panel=85)

                # 3. Panel Derecho (Canny o Esqueleto)
                panel_der = generar_panel_derecho(frame, results, self.extractor, usar_canny=modo_canny)

                # 4. Unir Paneles en Pantalla Doble
                pantalla_doble = unir_paneles(frame_izq, panel_der)

                cv2.imshow(nombre_ventana, pantalla_doble)
                key = cv2.waitKey(1) & 0xFF

                if key in [27, ord('q')]:
                    if oracion_completa:
                        db.guardar_traduccion(oracion_completa, registro_predicciones_sesion)
                    break

                elif key == ord('m'):
                    nuevo_modo = self.alternar_modo()
                    print(f"[*] Modo de reconocimiento cambiado a: {nuevo_modo}")

                elif key == ord('c'):
                    modo_canny = not modo_canny

                elif key in [ord('x'), ord('b')]:  # Borrar último carácter
                    if self.traductor.borrar_ultimo():
                        print("[*] Último carácter o palabra eliminada de la traducción.")

                elif key == 32:  # Barra espaciadora: guardar en SQLite
                    if oracion_completa:
                        db.guardar_traduccion(oracion_completa, registro_predicciones_sesion)
                        print(f"\n[OK] Oración guardada en SQLite: '{oracion_completa}'")
                        self.traductor.limpiar()
                        registro_predicciones_sesion.clear()

                elif key == ord('l'):  # Limpiar traducción
                    self.traductor.limpiar()
                    registro_predicciones_sesion.clear()

        finally:
            cap.release()
            self.extractor.cerrar()
            cv2.destroyAllWindows()
            print("[OK] Sesión de reconocimiento finalizada.")


def iniciar_reconocimiento():
    """Punto de entrada desde el menú principal."""
    try:
        motor = ReconocedorEnVivo()
        motor.ejecutar_bucle_vivo()
    except Exception as e:
        print(f"\n[ERROR AL INICIAR RECONOCIMIENTO] {e}")
