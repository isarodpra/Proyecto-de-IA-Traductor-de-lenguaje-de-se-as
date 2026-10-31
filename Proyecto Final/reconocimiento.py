"""
Módulo de Reconocimiento y Traducción en Tiempo Real.
Carga el modelo una sola vez, gestiona el buffer temporal de landmarks,
aplica suavizado de predicciones, renderiza la pantalla doble (cámara + Canny/esqueleto)
y persiste el historial de traducciones en SQLite.
"""

import cv2
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np

from config import (
    CAMARA_INDEX,
    FRAME_ANCHO,
    FRAME_ALTO,
    FRAMES_POR_SECUENCIA,
    TOTAL_CARACTERISTICAS,
    UMBRAL_CONFIANZA,
    VENTANA_SUAVIZADO,
    RUTA_MODELO_KERAS,
    RUTA_ETIQUETAS
)
import db
from landmarks import ExtractorLandmarks, generar_panel_derecho
from traduccion import TraductorSecuencias
from utils import (
    MedidorRendimiento,
    dibujar_barra_confianza,
    superponer_panel_translucido,
    unir_paneles
)
from entrenamiento import HAY_TENSORFLOW, ClasificadorFallbackNumpy

if HAY_TENSORFLOW:
    from tensorflow import keras


class ReconocedorEnVivo:
    """Motor de inferencia continua y renderizado visual."""

    def __init__(self):
        self.etiquetas: Dict[int, str] = {}
        self.modelo: Any = None
        self.es_modelo_keras: bool = False
        self.extractor = ExtractorLandmarks()
        self.traductor = TraductorSecuencias()
        self.medidor = MedidorRendimiento()
        self.buffer_landmarks = deque(maxlen=FRAMES_POR_SECUENCIA)
        self.historial_predicciones = deque(maxlen=VENTANA_SUAVIZADO)

        self._cargar_etiquetas_y_modelo()

    def _cargar_etiquetas_y_modelo(self):
        """Carga el modelo y etiquetas una sola vez en memoria para asegurar baja latencia."""
        if not RUTA_ETIQUETAS.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de etiquetas en '{RUTA_ETIQUETAS}'. "
                "Debe registrar señas y entrenar el modelo antes de iniciar el reconocimiento."
            )

        with open(RUTA_ETIQUETAS, "r", encoding="utf-8") as f:
            datos_etiquetas = json.load(f)
            self.etiquetas = {int(k): v for k, v in datos_etiquetas.items()}

        if HAY_TENSORFLOW and RUTA_MODELO_KERAS.exists():
            print(f"[*] Cargando modelo Keras desde '{RUTA_MODELO_KERAS}'...")
            self.modelo = keras.models.load_model(str(RUTA_MODELO_KERAS))
            self.es_modelo_keras = True
            print("[✓] Modelo Keras cargado en memoria.")
        else:
            ruta_fallback = RUTA_MODELO_KERAS.with_suffix(".json")
            if ruta_fallback.exists():
                print(f"[*] Cargando modelo ligero desde '{ruta_fallback}'...")
                self.modelo = ClasificadorFallbackNumpy.load(ruta_fallback)
                self.es_modelo_keras = False
                print("[✓] Modelo cargado en memoria.")
            else:
                raise FileNotFoundError(
                    "No se encontró ningún modelo entrenado (.keras o .json). "
                    "Por favor entrene el modelo desde el menú principal opción 2."
                )

    def predecir(self) -> Tuple[Optional[str], float]:
        """Realiza inferencia sobre el buffer de secuencia actual."""
        if len(self.buffer_landmarks) < FRAMES_POR_SECUENCIA:
            return None, 0.0

        t_inicio = time.time()
        # Convertir deque a array con forma (1, FRAMES_POR_SECUENCIA, TOTAL_CARACTERISTICAS)
        secuencia_arr = np.expand_dims(np.array(self.buffer_landmarks, dtype=np.float32), axis=0)

        if self.es_modelo_keras:
            prediccion = self.modelo(secuencia_arr, training=False).numpy()[0]
        else:
            prediccion = self.modelo.predict_proba(secuencia_arr[0])

        duracion_ms = (time.time() - t_inicio) * 1000.0
        self.medidor.registrar_inferencia(duracion_ms)

        indice_clase = int(np.argmax(prediccion))
        confianza = float(prediccion[indice_clase])

        # Suavizado temporal: guardar en cola y tomar moda ponderada
        self.historial_predicciones.append((indice_clase, confianza))

        conteo_votos = {}
        suma_confianzas = {}
        for idx, conf in self.historial_predicciones:
            if conf >= UMBRAL_CONFIANZA:
                conteo_votos[idx] = conteo_votos.get(idx, 0) + 1
                suma_confianzas[idx] = suma_confianzas.get(idx, 0.0) + conf

        if not conteo_votos:
            return None, confianza

        # Seleccionar la clase con más votos consistentes
        clase_ganadora = max(conteo_votos, key=conteo_votos.get)
        votos = conteo_votos[clase_ganadora]

        # Exigir mayoría de votos en la ventana para evitar parpadeos
        if votos >= (VENTANA_SUAVIZADO // 2 + 1):
            conf_promedio = suma_confianzas[clase_ganadora] / votos
            return self.etiquetas.get(clase_ganadora, "DESCONOCIDO"), conf_promedio

        return None, confianza

    def ejecutar_bucle_vivo(self):
        """Abre la cámara y ejecuta la inferencia continua con visualización doble."""
        cap = cv2.VideoCapture(CAMARA_INDEX)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo acceder a la cámara {CAMARA_INDEX}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_ANCHO)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_ALTO)

        modo_canny = True
        nombre_ventana = "Reconocimiento y Traduccion de Lenguaje de Senas en Vivo"
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 55)
        print("    RECONOCIMIENTO EN VIVO INICIADO")
        print(" [c]        : Alternar panel derecho (Canny <-> Landmarks)")
        print(" [ESPACIO]  : Guardar oracion actual en base de datos")
        print(" [l]        : Limpiar oracion actual")
        print(" [q] / [ESC]: Salir al menu")
        print("=" * 55 + "\n")

        registro_predicciones_sesion = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[!] Señal de video interrumpida.")
                    break

                self.medidor.registrar_frame()
                frame = cv2.flip(frame, 1)

                # Extraer características
                vector_caracteristicas, results = self.extractor.procesar_frame(frame)
                self.buffer_landmarks.append(vector_caracteristicas)

                # Dibujar esqueleto sobre el frame izquierdo
                frame_izq = self.extractor.dibujar_landmarks(frame, results)

                # Realizar predicción con el buffer
                seña_detectada, confianza = self.predecir()

                # Si se detectó una seña sólida, enviarla al traductor con debounce
                if seña_detectada:
                    palabra_nueva = self.traductor.agregar_prediccion(seña_detectada, confianza)
                    if palabra_nueva:
                        registro_predicciones_sesion.append({
                            "sena": seña_detectada,
                            "palabra": palabra_nueva,
                            "confianza": round(confianza, 3),
                            "timestamp": time.time()
                        })

                # Generar panel derecho (Canny o Esqueleto puro)
                panel_der = generar_panel_derecho(frame, results, self.extractor, usar_canny=modo_canny)

                # Renderizar panel HUD superior en panel izquierdo
                superponer_panel_translucido(frame_izq, 10, 10, 420, 130, (25, 25, 25), 0.75)
                cv2.putText(frame_izq, f"FPS: {self.medidor.fps:.1f} | Latencia: {self.medidor.latencia_ms:.1f} ms",
                            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

                txt_seña = seña_detectada if seña_detectada else "..."
                color_seña = (0, 255, 120) if seña_detectada else (160, 160, 160)
                cv2.putText(frame_izq, f"SENA: {txt_seña}", (20, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_seña, 2)

                cv2.putText(frame_izq, f"Confianza: {confianza * 100:.1f}%",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                dibujar_barra_confianza(frame_izq, 150, 83, 180, 14, confianza, UMBRAL_CONFIANZA)

                # Renderizar franja inferior con la oración traducida
                alto_f, ancho_f = frame_izq.shape[:2]
                superponer_panel_translucido(frame_izq, 0, alto_f - 60, ancho_f, 60, (15, 15, 15), 0.85)

                oracion = self.traductor.obtener_oracion_actual()
                texto_mostrar = f"Traduccion: {oracion}" if oracion else "Traduccion: (realice una sena frente a la camara)"
                cv2.putText(frame_izq, texto_mostrar, (15, alto_f - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                # Unir paneles en ventana doble
                pantalla_completa = unir_paneles(frame_izq, panel_der)

                cv2.imshow(nombre_ventana, pantalla_completa)
                key = cv2.waitKey(1) & 0xFF

                if key in [27, ord('q')]:
                    # Guardar automáticamente la oración final en SQLite si hay contenido
                    if oracion:
                        db.guardar_traduccion(oracion, registro_predicciones_sesion)
                    break

                elif key == ord('c'):
                    modo_canny = not modo_canny

                elif key == 32:  # Barra espaciadora: guardar en DB
                    if oracion:
                        db.guardar_traduccion(oracion, registro_predicciones_sesion)
                        print(f"\n[✓] Oración guardada en SQLite: '{oracion}'")
                        self.traductor.limpiar()
                        registro_predicciones_sesion = []

                elif key == ord('l'):
                    self.traductor.limpiar()
                    registro_predicciones_sesion = []

        finally:
            cap.release()
            self.extractor.cerrar()
            cv2.destroyAllWindows()
            print("[✓] Ventana de reconocimiento cerrada.")


def iniciar_reconocimiento():
    """Función de entrada para el menú principal."""
    try:
        motor = ReconocedorEnVivo()
        motor.ejecutar_bucle_vivo()
    except Exception as e:
        print(f"\n[ERROR AL INICIAR RECONOCIMIENTO] {e}")
