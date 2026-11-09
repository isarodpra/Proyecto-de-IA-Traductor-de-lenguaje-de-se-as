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
UMBRAL_CONFIANZA = 0.65  # Confianza mínima para aceptar una predicción
VENTANA_SUAVIZADO = 7    # Cantidad de predicciones para suavizado temporal
TIEMPO_DEBOUNCE_SEGUNDOS = 1.2  # Tiempo mínimo entre palabras distintas
TIEMPO_CONFIRMACION_ESTATICA = 0.60  # Segundos que la mano debe estar quieta para confirmar letra
TIEMPO_ESTABILIDAD_ESTATICA = TIEMPO_CONFIRMACION_ESTATICA  # Alias de compatibilidad
TIEMPO_COOLDOWN_POST_DETECCION = 1.50  # Segundos de espera tras detectar antes de admitir la siguiente seña
TIEMPO_AUTO_ESPACIO_SEGUNDOS = 4.0  # Pausa sin señas para insertar espacio o cierre
UMBRAL_MOVIMIENTO_DINAMICO = 0.040  # Varianza/velocidad de landmarks para considerar movimiento activo
UMBRAL_PRESENCIA_MANO = 0.05  # Magnitud mínima para considerar que hay una mano en pantalla

# Filtros y mejoras avanzadas
USAR_FILTRO_CLAHE = False          # False por defecto para máxima fluidez a 30-60 FPS (activable en Ajustes)
NORMALIZAR_LANDMARKS = True        # Normalización relativa e invariante a escala/distancia
VOZ_HABILITADA = True              # Síntesis de voz activa
VELOCIDAD_VOZ_TTS = 150            # Palabras por minuto en el sintetizador

# Modos de reconocimiento disponibles
MODO_AUTO = "AUTO (HIBRIDO)"
MODO_SOLO_ESTATICO = "SOLO ESTATICO"
MODO_SOLO_DINAMICO = "SOLO DINAMICO"

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
