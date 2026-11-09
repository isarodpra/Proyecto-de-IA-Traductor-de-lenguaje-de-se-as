"""
Módulo de Traducción y Síntesis de Lenguaje de Señas.
Convierte secuencias de señas reconocidas en oraciones coherentes en español,
aplica algoritmos de control de estado (transición, reposo y estabilidad),
evita el spam de repeticiones continuas y emite voz desacoplada con pyttsx3.
"""

import time
import threading
from typing import List, Dict, Optional, Tuple
from config import TIEMPO_DEBOUNCE_SEGUNDOS, TIEMPO_ESTABILIDAD_ESTATICA, TIEMPO_AUTO_ESPACIO_SEGUNDOS

# Intentar importar motor de voz pyttsx3
HAY_TTS = False
try:
    import pyttsx3
    HAY_TTS = True
except (ImportError, Exception):
    HAY_TTS = False


class MotorVozAsincrono:
    """Motor de síntesis de voz local que se ejecuta en segundo plano sin pausar el video."""

    def __init__(self):
        self.disponible = HAY_TTS
        self._cola_texto: List[str] = []
        self._bloqueo = threading.Lock()
        self._activo = True
        self._habilitado = True
        self._velocidad = 150

        if self.disponible:
            self._hilo = threading.Thread(target=self._bucle_locucion, daemon=True)
            self._hilo.start()

    def set_habilitado(self, habilitado: bool):
        self._habilitado = habilitado

    def set_velocidad(self, velocidad: int):
        self._velocidad = max(80, min(300, int(velocidad)))

    def hablar(self, texto: str):
        if not self.disponible or not self._habilitado or not texto.strip():
            return
        with self._bloqueo:
            self._cola_texto.append(texto.strip())

    def _bucle_locucion(self):
        try:
            engine = pyttsx3.init()
            while self._activo:
                texto_a_decir = None
                with self._bloqueo:
                    if self._cola_texto and self._habilitado:
                        texto_a_decir = self._cola_texto.pop(0)
                    elif self._cola_texto and not self._habilitado:
                        self._cola_texto.clear()

                if texto_a_decir:
                    try:
                        engine.setProperty('rate', self._velocidad)
                        engine.say(texto_a_decir)
                        engine.runAndWait()
                    except Exception:
                        pass
                else:
                    time.sleep(0.08)
        except Exception:
            self.disponible = False


class TraductorSecuencias:
    """Administra el buffer de palabras reconocidas, control de reposo y formación de oraciones."""

    MAPEO_PALABRAS = {
        "HOLA": "Hola",
        "ADIOS": "Adiós",
        "GRACIAS": "Muchas gracias",
        "POR_FAVOR": "Por favor",
        "SI": "Sí",
        "NO": "No",
        "BUENOS_DIAS": "Buenos días",
        "BUENAS_TARDES": "Buenas tardes",
        "COMO_ESTAS": "¿Cómo estás?",
        "BIEN": "Bien",
        "MAL": "Mal",
        "AYUDA": "Necesito ayuda",
        "NOMBRE": "Mi nombre es"
    }

    PATRONES_ORACIONES = {
        ("Hola", "¿Cómo estás?"): "¡Hola! ¿Cómo estás?",
        ("Bien", "Muchas gracias"): "Bien, muchas gracias.",
        ("Por favor", "Necesito ayuda"): "Por favor, necesito ayuda.",
        ("Sí", "Muchas gracias"): "Sí, muchas gracias."
    }

    def __init__(self, tiempo_estabilidad: float = TIEMPO_ESTABILIDAD_ESTATICA):
        self.tiempo_estabilidad = tiempo_estabilidad
        self.palabras_acumuladas: List[str] = []
        self.motor_voz = MotorVozAsincrono()

        # Variables de control de estado
        self.seña_en_observacion: Optional[str] = None
        self.tiempo_inicio_seña: float = 0.0
        self.ultima_seña_confirmada: Optional[str] = None
        self.mano_en_reposo: bool = True
        self.tiempo_ultimo_evento: float = time.time()

    def notificar_reposo(self):
        """Indica que no hay mano detectada o que la mano está bajada."""
        self.mano_en_reposo = True
        self.seña_en_observacion = None
        self.tiempo_inicio_seña = 0.0

    def procesar_seña_estatica(self, etiqueta_clase: str, confianza: float) -> Optional[str]:
        """
        Procesa una seña estática (letra o postura fija).
        Solo se confirma cuando la postura se mantiene estable durante tiempo_estabilidad.
        NO se vuelve a emitir mientras el usuario mantenga la misma postura.
        """
        ahora = time.time()

        if etiqueta_clase != self.seña_en_observacion:
            # Cambio de postura detectado: reiniciar cronómetro de estabilidad
            self.seña_en_observacion = etiqueta_clase
            self.tiempo_inicio_seña = ahora
            return None

        # Si se mantiene la misma postura
        duracion_estable = ahora - self.tiempo_inicio_seña
        if duracion_estable >= self.tiempo_estabilidad:
            # Solo confirmar si es diferente a la última confirmada O si hubo reposo intermedio
            if (etiqueta_clase != self.ultima_seña_confirmada) or self.mano_en_reposo:
                palabra = self.MAPEO_PALABRAS.get(etiqueta_clase.upper(), etiqueta_clase.upper())
                self._insertar_token(palabra)
                self.ultima_seña_confirmada = etiqueta_clase
                self.mano_en_reposo = False
                self.tiempo_ultimo_evento = ahora
                self.motor_voz.hablar(palabra)
                return palabra

        return None

    def procesar_gesto_dinamico(self, etiqueta_clase: str, confianza: float) -> Optional[str]:
        """
        Procesa un gesto dinámico (trayectoria culminada).
        Se inserta una sola vez al terminar el movimiento.
        """
        ahora = time.time()
        # Para dinámicas, aplicar debounce básico entre el mismo gesto
        if (etiqueta_clase == self.ultima_seña_confirmada) and not self.mano_en_reposo:
            if (ahora - self.tiempo_ultimo_evento) < TIEMPO_DEBOUNCE_SEGUNDOS:
                return None

        palabra = self.MAPEO_PALABRAS.get(etiqueta_clase.upper(), etiqueta_clase.capitalize())
        self._insertar_token(palabra)
        self.ultima_seña_confirmada = etiqueta_clase
        self.mano_en_reposo = False
        self.tiempo_ultimo_evento = ahora
        self.motor_voz.hablar(palabra)
        return palabra

    def _insertar_token(self, token: str):
        """Agrega el token a la lista acumulada."""
        self.palabras_acumuladas.append(token)

    def borrar_ultimo(self) -> bool:
        """Elimina el último elemento agregado."""
        if self.palabras_acumuladas:
            self.palabras_acumuladas.pop()
            self.ultima_seña_confirmada = None
            return True
        return False

    def obtener_oracion_actual(self) -> str:
        """Construye la oración actual uniendo letras consecutivas o separando palabras."""
        if not self.palabras_acumuladas:
            return ""

        # Formar texto uniendo letras continuas y espaciando palabras completas
        partes = []
        token_actual = ""

        for item in self.palabras_acumuladas:
            # Si es letra individual (longitud 1)
            if len(item) == 1 and item.isalpha():
                token_actual += item
            else:
                if token_actual:
                    partes.append(token_actual)
                    token_actual = ""
                partes.append(item)

        if token_actual:
            partes.append(token_actual)

        # Reglas contextuales de unión de palabras
        if len(partes) >= 2:
            par = (partes[-2], partes[-1])
            if par in self.PATRONES_ORACIONES:
                base = partes[:-2]
                oracion_unida = " ".join(base) + (" " if base else "") + self.PATRONES_ORACIONES[par]
                return oracion_unida.strip()

        return " ".join(partes).strip()

    def limpiar(self):
        """Reinicia la oración actual y el estado."""
        self.palabras_acumuladas.clear()
        self.ultima_seña_confirmada = None
        self.seña_en_observacion = None
        self.tiempo_inicio_seña = 0.0
        self.mano_en_reposo = True
