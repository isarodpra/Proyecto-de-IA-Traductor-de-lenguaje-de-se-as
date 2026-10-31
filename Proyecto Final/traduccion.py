"""
Módulo de Traducción y Síntesis de Lenguaje de Señas.
Convierte secuencias de señas reconocidas en oraciones coherentes en español,
aplica algoritmos de 'debounce' para evitar repeticiones no deseadas,
aplica diccionario contextual de corrección y síntesis de voz por hilo desacoplado.
"""

import time
import threading
from typing import List, Dict, Optional
from config import TIEMPO_DEBOUNCE_SEGUNDOS, TIEMPO_AUTO_ESPACIO_SEGUNDOS

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

        if self.disponible:
            self._hilo = threading.Thread(target=self._bucle_locucion, daemon=True)
            self._hilo.start()

    def hablar(self, texto: str):
        if not self.disponible or not texto.strip():
            return
        with self._bloqueo:
            self._cola_texto.append(texto.strip())

    def _bucle_locucion(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Velocidad natural
            while self._activo:
                texto_a_decir = None
                with self._bloqueo:
                    if self._cola_texto:
                        texto_a_decir = self._cola_texto.pop(0)

                if texto_a_decir:
                    engine.say(texto_a_decir)
                    engine.runAndWait()
                else:
                    time.sleep(0.1)
        except Exception as e:
            # En caso de error de hardware de audio, deshabilitar sin interrumpir el video
            self.disponible = False


class TraductorSecuencias:
    """Administra el buffer de palabras reconocidas, debounce y formación de oraciones."""

    # Diccionario de equivalencias y mejoras contextuales
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

    def __init__(self, tiempo_debounce: float = TIEMPO_DEBOUNCE_SEGUNDOS):
        self.tiempo_debounce = tiempo_debounce
        self.ultima_palabra: Optional[str] = None
        self.tiempo_ultima_palabra: float = 0.0
        self.palabras_acumuladas: List[str] = []
        self.motor_voz = MotorVozAsincrono()

    def agregar_prediccion(self, etiqueta_clase: str, confianza: float) -> Optional[str]:
        """
        Recibe una predicción del modelo y la agrega a la oración si supera el filtro debounce.
        Retorna la palabra añadida o None si fue filtrada.
        """
        ahora = time.time()
        palabra_traducida = self.MAPEO_PALABRAS.get(etiqueta_clase.upper(), etiqueta_clase.capitalize())

        # Comprobar condición de debounce: misma palabra en un lapso corto
        if palabra_traducida == self.ultima_palabra:
            if (ahora - self.tiempo_ultima_palabra) < self.tiempo_debounce:
                return None  # Supresión por rebote/repetición continua

        # Agregar nueva palabra
        self.palabras_acumuladas.append(palabra_traducida)
        self.ultima_palabra = palabra_traducida
        self.tiempo_ultima_palabra = ahora

        # Pronunciar la palabra si la voz está activa
        self.motor_voz.hablar(palabra_traducida)
        return palabra_traducida

    def obtener_oracion_actual(self) -> str:
        """Genera la oración completa acumulada aplicando reglas de diccionarios de patrones."""
        if not self.palabras_acumuladas:
            return ""

        # Verificar si las últimas dos palabras forman un patrón especial
        if len(self.palabras_acumuladas) >= 2:
            par = (self.palabras_acumuladas[-2], self.palabras_acumuladas[-1])
            if par in self.PATRONES_ORACIONES:
                base = self.palabras_acumuladas[:-2]
                oracion_unida = " ".join(base) + (" " if base else "") + self.PATRONES_ORACIONES[par]
                return oracion_unida.strip()

        return " ".join(self.palabras_acumuladas).strip()

    def limpiar(self):
        """Reinicia la oración actual."""
        self.palabras_acumuladas.clear()
        self.ultima_palabra = None
        self.tiempo_ultima_palabra = 0.0
