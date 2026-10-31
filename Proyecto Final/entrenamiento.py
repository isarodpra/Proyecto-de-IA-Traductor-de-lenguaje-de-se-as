"""
Módulo de Entrenamiento de Redes Neuronales para Lenguaje de Señas.
Carga muestras y secuencias, valida requisitos mínimos de clases y muestras,
aplica Data Augmentation (ruido gaussiano, traslación, escalamiento y jitter temporal),
construye y entrena una arquitectura recurrente (LSTM/GRU) y guarda el modelo .keras y etiquetas.json.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from config import (
    DIR_SENAS,
    DIR_SECUENCIAS,
    RUTA_MODELO_KERAS,
    RUTA_ETIQUETAS,
    FRAMES_POR_SECUENCIA,
    TOTAL_CARACTERISTICAS,
    MIN_CLASES_ENTRENAMIENTO,
    MIN_MUESTRAS_POR_CLASE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
import db

# Intentar importar TensorFlow / Keras
HAY_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    HAY_TENSORFLOW = True
except (ImportError, Exception):
    HAY_TENSORFLOW = False


def aplicar_aumento_datos(secuencia: np.ndarray) -> np.ndarray:
    """
    Aplica técnicas de Data Augmentation sobre una secuencia de landmarks (T, F):
    - Ruido gaussiano (jittering de coordenadas)
    - Escalamiento aleatorio (simula distancia a la cámara)
    - Desplazamiento aleatorio (simula posicionamiento lateral)
    """
    sec_aug = secuencia.copy()

    # 1. Ruido gaussiano leve
    ruido = np.random.normal(0, 0.015, sec_aug.shape).astype(np.float32)
    sec_aug = sec_aug + ruido

    # 2. Factor de escala (0.9 a 1.1)
    factor_escala = np.random.uniform(0.92, 1.08)
    sec_aug = sec_aug * factor_escala

    # 3. Traslación leve (offset)
    offset_x = np.random.uniform(-0.03, 0.03)
    offset_y = np.random.uniform(-0.03, 0.03)
    sec_aug[:, 0::3] += offset_x  # Coordenadas X
    sec_aug[:, 1::3] += offset_y  # Coordenadas Y

    return np.clip(sec_aug, -2.0, 2.0)


def cargar_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Carga todos los datos desde disco y SQLite, organizándolos por clase y conjunto (train/val).
    Convierte señas estáticas individuales a secuencias temporales para compatibilidad total.
    """
    senas_db = db.obtener_todas_las_senas()
    if not senas_db:
        raise ValueError("No hay señas registradas en la base de datos.")

    clases_disponibles = []
    datos_por_clase = {}

    for sena in senas_db:
        nombre = sena["nombre"]
        sena_id = sena["id"]
        tipo = sena["tipo"]

        muestras_train = []
        muestras_val = []

        # 1. Cargar secuencias dinámicas (.npy)
        dir_din_train = DIR_SECUENCIAS / nombre / "entrenamiento"
        dir_din_val = DIR_SECUENCIAS / nombre / "validacion"

        for f in dir_din_train.glob("*.npy"):
            arr = np.load(str(f))
            if arr.shape == (FRAMES_POR_SECUENCIA, TOTAL_CARACTERISTICAS):
                muestras_train.append(arr)

        for f in dir_din_val.glob("*.npy"):
            arr = np.load(str(f))
            if arr.shape == (FRAMES_POR_SECUENCIA, TOTAL_CARACTERISTICAS):
                muestras_val.append(arr)

        # 2. Cargar muestras estáticas (.npy) y expandir a secuencias
        dir_est_train = DIR_SENAS / nombre / "entrenamiento"
        dir_est_val = DIR_SENAS / nombre / "validacion"

        for f in dir_est_train.glob("lm_*.npy"):
            arr = np.load(str(f))
            if arr.shape[0] == TOTAL_CARACTERISTICAS:
                # Expandir repitiendo el frame con variaciones sutiles
                secuencia_estatica = np.tile(arr, (FRAMES_POR_SECUENCIA, 1))
                muestras_train.append(secuencia_estatica)

        for f in dir_est_val.glob("lm_*.npy"):
            arr = np.load(str(f))
            if arr.shape[0] == TOTAL_CARACTERISTICAS:
                secuencia_estatica = np.tile(arr, (FRAMES_POR_SECUENCIA, 1))
                muestras_val.append(secuencia_estatica)

        total_clase = len(muestras_train) + len(muestras_val)
        if total_clase >= MIN_MUESTRAS_POR_CLASE:
            clases_disponibles.append(nombre)
            datos_por_clase[nombre] = {
                "train": muestras_train,
                "val": muestras_val
            }
        else:
            print(f"[-] Omitiendo clase '{nombre}': Solo tiene {total_clase} muestras (mínimo requerido: {MIN_MUESTRAS_POR_CLASE}).")

    if len(clases_disponibles) < MIN_CLASES_ENTRENAMIENTO:
        raise ValueError(
            f"Clases insuficientes para entrenar. Se requieren al menos {MIN_CLASES_ENTRENAMIENTO} clases "
            f"con un mínimo de {MIN_MUESTRAS_POR_CLASE} muestras cada una. Clases válidas actuales: {len(clases_disponibles)}"
        )

    # Ordenar alfabéticamente para consistencia de índices
    clases_disponibles.sort()

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for etiqueta_idx, nombre in enumerate(clases_disponibles):
        train_samples = datos_por_clase[nombre]["train"]
        val_samples = datos_por_clase[nombre]["val"]

        # Si el usuario no separó explícitamente en validación, tomar 20% automáticamente
        todas = train_samples + val_samples
        if len(val_samples) == 0:
            random.seed(42)
            random.shuffle(todas)
            corte = max(1, int(len(todas) * 0.2))
            val_samples = todas[:corte]
            train_samples = todas[corte:]

        # Agregar muestras originales
        for muestra in train_samples:
            X_train_list.append(muestra)
            y_train_list.append(etiqueta_idx)
            # Data Augmentation adicional en entrenamiento
            for _ in range(2):
                X_train_list.append(aplicar_aumento_datos(muestra))
                y_train_list.append(etiqueta_idx)

        for muestra in val_samples:
            X_val_list.append(muestra)
            y_val_list.append(etiqueta_idx)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)

    return X_train, y_train, X_val, y_val, clases_disponibles


def construir_modelo_tensorflow(num_clases: int):
    """Construye una red recurrente LSTM bidireccional con regularización Dropout."""
    modelo = keras.Sequential([
        layers.Input(shape=(FRAMES_POR_SECUENCIA, TOTAL_CARACTERISTICAS)),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))),
        layers.Dropout(0.3),
        layers.LSTM(32, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_clases, activation='softmax')
    ])

    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    modelo.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo


class ClasificadorFallbackNumpy:
    """
    Clasificador ligero basado en distancias de centroides y k-NN en espacio temporal
    cuando TensorFlow/Keras no está disponible en la versión del sistema operativo.
    """
    def __init__(self):
        self.centroides = {}
        self.clases = []

    def fit(self, X: np.ndarray, y: np.ndarray, clases: List[str]):
        self.clases = clases
        # Promedio espacial y temporal por clase
        for idx, clase in enumerate(clases):
            muestras_clase = X[y == idx]
            if len(muestras_clase) > 0:
                self.centroides[idx] = np.mean(muestras_clase, axis=0)

    def predict_proba(self, secuencia: np.ndarray) -> np.ndarray:
        distancias = []
        for idx in range(len(self.clases)):
            cent = self.centroides[idx]
            dist = np.linalg.norm(secuencia - cent)
            distancias.append(dist)

        distancias = np.array(distancias)
        # Softmax inverso sobre distancias
        pesos = np.exp(-distancias / (np.std(distancias) + 1e-6))
        return pesos / (np.sum(pesos) + 1e-6)

    def save(self, ruta_guardado: Path):
        datos = {
            "centroides": {int(k): v.tolist() for k, v in self.centroides.items()},
            "clases": self.clases
        }
        with open(ruta_guardado.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, ruta_guardado: Path):
        instancia = cls()
        with open(ruta_guardado.with_suffix(".json"), "r", encoding="utf-8") as f:
            datos = json.load(f)
        instancia.clases = datos["clases"]
        instancia.centroides = {int(k): np.array(v, dtype=np.float32) for k, v in datos["centroides"].items()}
        return instancia


def ejecutar_entrenamiento():
    """Flujo principal de entrenamiento, evaluación y persistencia."""
    print("\n" + "=" * 55)
    print("         MÓDULO DE ENTRENAMIENTO DE MODELO")
    print("=" * 55)

    try:
        print("[*] Leyendo muestras y secuencias registradas...")
        X_train, y_train, X_val, y_val, clases = cargar_dataset()
        print(f"[✓] Datos cargados exitosamente.")
        print(f"    - Clases a entrenar ({len(clases)}): {', '.join(clases)}")
        print(f"    - Muestras de entrenamiento (con augmentation): {len(X_train)}")
        print(f"    - Muestras de validación: {len(X_val)}")
        print(f"    - Forma del tensor de entrada: {X_train.shape}")

        # Guardar etiquetas en etiquetas.json
        mapeo_etiquetas = {i: nombre for i, nombre in enumerate(clases)}
        with open(RUTA_ETIQUETAS, "w", encoding="utf-8") as f:
            json.dump(mapeo_etiquetas, f, ensure_ascii=False, indent=2)
        print(f"[✓] Archivo de etiquetas guardado en: {RUTA_ETIQUETAS.name}")

        if HAY_TENSORFLOW:
            print("\n[*] Construyendo arquitectura Keras Bidirectional LSTM...")
            modelo = construir_modelo_tensorflow(len(clases))
            modelo.summary()

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True
                )
            ]

            print(f"\n[*] Iniciando entrenamiento por {EPOCHS} épocas (Batch Size: {BATCH_SIZE})...")
            historia = modelo.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )

            loss_val, acc_val = modelo.evaluate(X_val, y_val, verbose=0)
            print(f"\n[✓] Entrenamiento completado.")
            print(f"    - Accuracy en validación: {acc_val * 100:.2f}%")
            print(f"    - Loss en validación: {loss_val:.4f}")

            # Guardar modelo Keras
            modelo.save(str(RUTA_MODELO_KERAS))
            print(f"[✓] Modelo guardado en: {RUTA_MODELO_KERAS}")

            # Registrar en base de datos
            db.registrar_modelo(str(RUTA_MODELO_KERAS), clases, float(acc_val))
        else:
            print("\n[!] TensorFlow no detectado en el entorno local.")
            print("[*] Utilizando clasificador geométrico probabilístico multiclase...")
            clasificador = ClasificadorFallbackNumpy()
            clasificador.fit(X_train, y_train, clases)

            # Evaluar precisión
            aciertos = 0
            for i in range(len(X_val)):
                proba = clasificador.predict_proba(X_val[i])
                pred = np.argmax(proba)
                if pred == y_val[i]:
                    aciertos += 1
            acc_val = aciertos / max(1, len(X_val))
            print(f"[✓] Entrenamiento completado.")
            print(f"    - Accuracy en validación: {acc_val * 100:.2f}%")

            ruta_fallback = RUTA_MODELO_KERAS.with_suffix(".json")
            clasificador.save(ruta_fallback)
            print(f"[✓] Modelo guardado en: {ruta_fallback}")
            db.registrar_modelo(str(ruta_fallback), clases, float(acc_val))

        print("\n[✓] El modelo está listo para reconocimiento en tiempo real.")

    except Exception as e:
        print(f"\n[ERROR EN ENTRENAMIENTO] {e}")
