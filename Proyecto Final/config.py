"""
Módulo de Configuración Global del Sistema de Reconocimiento de Lenguaje de Señas.
Define rutas relativas, hiperparámetros de captura, entrenamiento y reconocimiento.
"""

from pathlib import Path

# Directorios base del proyecto
BASE_DIR = Path(__file__).resolve().parent
DIR_DATOS = BASE_DIR / "datos"
DIR_SENAS = DIR_DATOS / "senas"
DIR_SECUENCIAS = DIR_DATOS / "secuencias"
DIR_DB = DIR_DATOS / "db"
DIR_MODELOS = BASE_DIR / "modelos"

# Rutas de archivos clave
RUTA_DB = DIR_DB / "senas.db"
RUTA_MODELO_KERAS = DIR_MODELOS / "modelo_senas.keras"
RUTA_ETIQUETAS = BASE_DIR / "etiquetas.json"
DIR_HAAR_CASCADES = BASE_DIR / "cascades"

# Configuración de captura y video
CAMARA_INDEX = 0
FRAME_ANCHO = 640
FRAME_ALTO = 480
FPS_OBJETIVO = 30

# Parámetros para señas dinámicas y temporales
FRAMES_POR_SECUENCIA = 30  # Cantidad de frames para una muestra dinámica
BUFFER_HISTORIAL_FRAMES = 30  # Longitud del buffer deslizante en reconocimiento en vivo

# Dimensiones del vector de características (Landmarks)
# Mano izquierda: 21 * 3 = 63
# Mano derecha: 21 * 3 = 63
# Pose esencial (hombros, codos, muñecas, nariz, etc.): 33 * 3 = 99
# Total características por frame = 225
CANTIDAD_LANDMARKS_MANO = 21 * 3
CANTIDAD_LANDMARKS_POSE = 33 * 3
TOTAL_CARACTERISTICAS = (CANTIDAD_LANDMARKS_MANO * 2) + CANTIDAD_LANDMARKS_POSE

# Umbrales de reconocimiento y traducción
UMBRAL_CONFIANZA = 0.70  # Confianza mínima para aceptar una predicción
VENTANA_SUAVIZADO = 7    # Cantidad de predicciones para suavizado temporal (moda/media)
TIEMPO_DEBOUNCE_SEGUNDOS = 1.5  # Tiempo mínimo entre palabras consecutivas idénticas
TIEMPO_AUTO_ESPACIO_SEGUNDOS = 3.5  # Pausa sin señas para insertar espacio o cierre

# Hiperparámetros de entrenamiento
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
MIN_CLASES_ENTRENAMIENTO = 2
MIN_MUESTRAS_POR_CLASE = 5

# Crear automáticamente la estructura de directorios necesaria
for directorio in [DIR_DATOS, DIR_SENAS, DIR_SECUENCIAS, DIR_DB, DIR_MODELOS, DIR_HAAR_CASCADES]:
    directorio.mkdir(parents=True, exist_ok=True)
