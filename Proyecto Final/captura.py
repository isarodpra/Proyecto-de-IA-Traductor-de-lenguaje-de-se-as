"""
Módulo de Captura y Registro de Muestras de Lenguaje de Señas.
Permite registrar señas estáticas y dinámicas, grabando imágenes, landmarks en .npy
y persistiendo metadatos en SQLite con soporte para train/val y borrado inmediato.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    DIR_SENAS,
    DIR_SECUENCIAS,
    CAMARA_INDEX,
    FRAME_ANCHO,
    FRAME_ALTO,
    FRAMES_POR_SECUENCIA
)
import db
from landmarks import ExtractorLandmarks, generar_panel_derecho
from utils import dibujar_texto_con_fondo, superponer_panel_translucido, unir_paneles


def iniciar_interfaz_captura():
    """Flujo de consola interactivo para configurar y comenzar la captura de señas."""
    print("\n" + "=" * 55)
    print("        MÓDULO DE REGISTRO DE SEÑAS Y MUESTRAS")
    print("=" * 55)

    nombre = input(">> Ingrese el nombre de la seña (ej. HOLA, GRACIAS, A): ").strip()
    if not nombre:
        print("[!] Nombre inválido. Operación cancelada.")
        return

    print("\nSeleccione el tipo de seña:")
    print("  1. Estática (Letras del abecedario, números, gestos fijos)")
    print("  2. Dinámica (Palabras con movimiento secuencial de manos)")
    opcion_tipo = input("Opción (1/2, default: 1): ").strip()
    tipo = "dinamica" if opcion_tipo == "2" else "estatica"

    descripcion = input(">> Descripción opcional (ej. Saludo con palma abierta): ").strip()

    # Registrar o recuperar ID de seña en SQLite
    sena_id = db.registrar_o_obtener_sena(nombre, tipo, descripcion)
    print(f"\n[✓] Seña '{nombre.upper()}' registrada (ID: {sena_id}, Tipo: {tipo}).")

    ejecutar_captura_en_vivo(sena_id, nombre.upper(), tipo)


def ejecutar_captura_en_vivo(sena_id: int, nombre_sena: str, tipo_sena: str):
    """Abre la cámara y gestiona la captura visual interactiva."""
    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara en el índice {CAMARA_INDEX}.")
        print("Verifique que la cámara esté conectada y no esté en uso por otra app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_ALTO)

    extractor = ExtractorLandmarks()

    conjunto_actual = "entrenamiento"
    secuencia_buffer = []
    grabando_secuencia = False
    conteo_secuencia = 0
    mensaje_notificacion = "Listo para capturar"
    tiempo_notificacion = time.time()

    # Directorios específicos para esta clase
    dir_estaticas_train = DIR_SENAS / nombre_sena / "entrenamiento"
    dir_estaticas_val = DIR_SENAS / nombre_sena / "validacion"
    dir_dinamicas_train = DIR_SECUENCIAS / nombre_sena / "entrenamiento"
    dir_dinamicas_val = DIR_SECUENCIAS / nombre_sena / "validacion"

    for d in [dir_estaticas_train, dir_estaticas_val, dir_dinamicas_train, dir_dinamicas_val]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[CONTROLES]")
    print(" [ESPACIO]  : Capturar muestra (estática) o Iniciar grabación (dinámica)")
    print(" [v]        : Alternar conjunto (Entrenamiento <-> Validación)")
    print(" [b]        : Borrar última muestra registrada")
    print(" [q] / [ESC]: Salir y regresar al menú principal\n")

    nombre_ventana = "Registro de Senas - Lenguaje de Senas AI"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Error leyendo frame de la cámara.")
                break

            # Espejar para mayor naturalidad al usuario
            frame = cv2.flip(frame, 1)

            # Extraer landmarks
            vector_caracteristicas, results = extractor.procesar_frame(frame)

            # Dibujar landmarks en el frame de la cámara
            frame_dibujado = extractor.dibujar_landmarks(frame, results)

            # Obtener conteos actualizados
            conteos = db.obtener_conteo_muestras_por_sena(sena_id)

            # Panel de información HUD en frame_dibujado
            superponer_panel_translucido(frame_dibujado, 10, 10, 360, 120, (20, 20, 20), 0.7)
            cv2.putText(frame_dibujado, f"SENA: {nombre_sena} ({tipo_sena.upper()})",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame_dibujado, f"Conjunto: [{conjunto_actual.upper()}] (tecla 'v')",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 1)
            cv2.putText(frame_dibujado, f"Train: {conteos['entrenamiento']} | Val: {conteos['validacion']} | Total: {conteos['total']}",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

            # Mensaje temporal de notificación
            if time.time() - tiempo_notificacion < 2.5:
                cv2.putText(frame_dibujado, f">> {mensaje_notificacion}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Manejo de grabación para señas dinámicas
            if grabando_secuencia:
                secuencia_buffer.append(vector_caracteristicas)
                conteo_secuencia += 1

                # Visualizar barra de progreso de la secuencia
                progreso = conteo_secuencia / FRAMES_POR_SECUENCIA
                ancho_barra = int(frame.shape[1] * 0.8)
                x_inicio = int(frame.shape[1] * 0.1)
                y_pos = frame.shape[0] - 40
                cv2.rectangle(frame_dibujado, (x_inicio, y_pos), (x_inicio + ancho_barra, y_pos + 18), (40, 40, 40), -1)
                cv2.rectangle(frame_dibujado, (x_inicio, y_pos), (x_inicio + int(ancho_barra * progreso), y_pos + 18), (0, 120, 255), -1)
                cv2.putText(frame_dibujado, f"GRABANDO SECUENCIA: {conteo_secuencia}/{FRAMES_POR_SECUENCIA}",
                            (x_inicio + 10, y_pos - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

                if conteo_secuencia >= FRAMES_POR_SECUENCIA:
                    # Guardar secuencia .npy
                    timestamp = int(time.time() * 1000)
                    carpeta_destino = dir_dinamicas_train if conjunto_actual == "entrenamiento" else dir_dinamicas_val
                    ruta_npy = carpeta_destino / f"seq_{nombre_sena}_{timestamp}.npy"

                    arr_secuencia = np.array(secuencia_buffer, dtype=np.float32)  # Forma: (30, TOTAL_CARACTERISTICAS)
                    np.save(str(ruta_npy), arr_secuencia)

                    # Guardar en base de datos
                    db.guardar_secuencia(sena_id, conjunto_actual, str(ruta_npy), FRAMES_POR_SECUENCIA)

                    mensaje_notificacion = f"Secuencia guardada ({conteo_secuencia} frames)"
                    tiempo_notificacion = time.time()
                    grabando_secuencia = False
                    secuencia_buffer = []
                    conteo_secuencia = 0

            # Generar panel derecho (Canny)
            panel_der = generar_panel_derecho(frame, results, extractor, usar_canny=True)

            # Unir paneles en ventana doble
            pantalla_doble = unir_paneles(frame_dibujado, panel_der)

            cv2.imshow(nombre_ventana, pantalla_doble)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q')]:  # ESC o q
                break

            elif key == ord('v'):  # Alternar conjunto
                conjunto_actual = "validacion" if conjunto_actual == "entrenamiento" else "entrenamiento"
                mensaje_notificacion = f"Cambiado a: {conjunto_actual.upper()}"
                tiempo_notificacion = time.time()

            elif key == 32:  # Barra espaciadora
                if tipo_sena == "dinamica":
                    if not grabando_secuencia:
                        grabando_secuencia = True
                        secuencia_buffer = []
                        conteo_secuencia = 0
                        mensaje_notificacion = "Iniciando grabacion de secuencia..."
                        tiempo_notificacion = time.time()
                else:
                    # Guardar muestra estática
                    timestamp = int(time.time() * 1000)
                    carpeta_destino = dir_estaticas_train if conjunto_actual == "entrenamiento" else dir_estaticas_val

                    ruta_img = carpeta_destino / f"img_{nombre_sena}_{timestamp}.jpg"
                    ruta_landmarks = carpeta_destino / f"lm_{nombre_sena}_{timestamp}.npy"

                    # Guardar imagen recortada o frame completo
                    cv2.imwrite(str(ruta_img), frame)
                    np.save(str(ruta_landmarks), vector_caracteristicas)

                    # Guardar en SQLite
                    db.guardar_muestra(sena_id, conjunto_actual, str(ruta_img), str(ruta_landmarks))

                    mensaje_notificacion = f"Muestra estatica guardada en {conjunto_actual}"
                    tiempo_notificacion = time.time()

            elif key == ord('b'):  # Borrar última muestra
                ruta_borrada = db.eliminar_ultima_muestra(sena_id)
                if ruta_borrada:
                    try:
                        p = Path(ruta_borrada)
                        if p.exists():
                            p.unlink()
                        # Si era estática con imagen gemela
                        p_img = p.with_name(p.name.replace("lm_", "img_").replace(".npy", ".jpg"))
                        if p_img.exists():
                            p_img.unlink()
                        mensaje_notificacion = "Ultima muestra eliminada con exito"
                    except Exception as e:
                        mensaje_notificacion = f"Registro eliminado de DB ({e})"
                else:
                    mensaje_notificacion = "No habia muestras para eliminar"
                tiempo_notificacion = time.time()

    finally:
        cap.release()
        extractor.cerrar()
        cv2.destroyAllWindows()
        print("\n[✓] Sesión de captura finalizada.")
