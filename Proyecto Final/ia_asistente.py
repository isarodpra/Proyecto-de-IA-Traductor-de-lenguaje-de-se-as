"""
Módulo de Asistencia Inteligente con Google Gemini API (Visión y NLP).
Provee:
1. Validación visual y pedagógica de señas con Gemini 2.5 Flash (Quality Gate).
2. Traducción contextual de glosas de señas a oraciones naturales en español (NLP).
3. Fallback automático a reglas locales en caso de desconexión o fallo de red.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Configurar salida de consola segura para Windows (evita UnicodeEncodeError con cp1252)
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Directorios y rutas de entorno
DIR_ACTUAL = Path(__file__).resolve().parent
RUTA_ENV = DIR_ACTUAL / ".env"

try:
    from dotenv import load_dotenv
    if RUTA_ENV.exists():
        load_dotenv(RUTA_ENV)
    else:
        load_dotenv()
except ImportError:
    # Lectura manual de .env si python-dotenv no estuviera instalado
    if RUTA_ENV.exists():
        try:
            with open(RUTA_ENV, "r", encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if linea and not linea.startswith("#") and "=" in linea:
                        k, v = linea.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass

# Importación oficial de Google GenAI
HAY_GENAI = False
try:
    from google import genai
    HAY_GENAI = True
except ImportError:
    HAY_GENAI = False


class AsistenteIA:
    """Controlador central para servicios cognitivos multimodales de Google Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or ""
        self.api_key = raw_key.strip().strip('"').strip("'")
        self.client = None
        self.modelo = "gemini-2.5-flash"
        self._inicializar_cliente()

    def _inicializar_cliente(self):
        """Inicializa el cliente de Gemini usando 'from google import genai'."""
        if not self.api_key or not HAY_GENAI:
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[!] Error inicializando cliente Gemini: {e}")
            self.client = None

    def actualizar_api_key(self, nueva_key: str):
        """Actualiza la clave de API en tiempo de ejecución y en .env."""
        self.api_key = nueva_key.strip().strip('"').strip("'")
        os.environ["GEMINI_API_KEY"] = self.api_key
        try:
            with open(RUTA_ENV, "w", encoding="utf-8") as f:
                f.write(f"# Google AI Studio API Key\nGEMINI_API_KEY={self.api_key}\n")
        except Exception:
            pass
        self._inicializar_cliente()

    def esta_disponible(self) -> bool:
        """Indica si el cliente de IA está configurado y listo."""
        return self.client is not None and bool(self.api_key)

    def verificar_conexion(self) -> Tuple[bool, str]:
        """Envía un ping ligero a Gemini para validar conectividad y validez de la API Key."""
        if not self.esta_disponible():
            if not HAY_GENAI:
                return False, "Biblioteca google-genai no disponible en este entorno Python."
            return False, "API Key de Gemini no configurada."

        try:
            resp = self.client.models.generate_content(
                model=self.modelo,
                contents="Responde únicamente: OK"
            )
            if resp and resp.text:
                return True, "Conexión exitosa con Google Gemini (API lista)."
            return False, "Respuesta vacía del servidor."
        except Exception as e:
            msg = str(e)
            if "getaddrinfo failed" in msg or "ConnectionError" in msg:
                return False, "Sin conexión a internet (modo offline activo)."
            elif "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
                return False, "Clave API inválida o sin permisos."
            return False, f"Fallo de conexión con Gemini: {msg[:100]}"

    def validar_calidad_sena(
        self,
        frame_bgr: np.ndarray,
        nombre_sena: str,
        tipo_sena: str = "estatica"
    ) -> Dict[str, Any]:
        """
        Analiza una muestra visual capturada frente a la cámara con Gemini Vision.
        Determina si la mano está bien orientada, si los dedos son nítidos y si coincide con la seña.
        """
        if not self.esta_disponible() or frame_bgr is None:
            return {
                "valida": True,
                "calidad": "desconocida",
                "diagnostico": "Validación con IA no disponible (modo local activo).",
                "recomendacion": "Asegúrese de centrar la mano y mantener buena iluminación."
            }

        try:
            # Codificar imagen a JPEG en memoria
            _, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            bytes_img = buffer.tobytes()

            # Usar genai.types directamente desde el módulo genai
            part_imagen = genai.types.Part.from_bytes(data=bytes_img, mime_type="image/jpeg")

            prompt = (
                f"Eres un instructor y evaluador de Lenguaje de Señas (LSM / ASL).\n"
                f"El usuario está intentando realizar la seña: '{nombre_sena}' (tipo: {tipo_sena}).\n"
                f"Analiza minuciosamente la imagen proporcionada:\n"
                f"1. ¿Se observa claramente la mano y los dedos sin cortes ni desenfoque?\n"
                f"2. ¿La postura corresponde de forma aceptable a la seña '{nombre_sena}' o a una mano visible para entrenamiento?\n"
                f"3. ¿La iluminación y encuadre son adecuados?\n\n"
                f"Devuelve OBLIGATORIAMENTE un objeto JSON con este esquema exacto:\n"
                f'{{\n'
                f'  "valida": true o false,\n'
                f'  "calidad": "alta" o "media" o "baja",\n'
                f'  "diagnostico": "Explicación breve de 1 o 2 oraciones.",\n'
                f'  "recomendacion": "Consejo constructivo breve para mejorar la postura o posición."\n'
                f'}}'
            )

            resp = self.client.models.generate_content(
                model=self.modelo,
                contents=[part_imagen, prompt]
            )
            texto_respuesta = resp.text.strip() if resp and resp.text else ""

            # Extraer JSON de la respuesta
            match = re.search(r"\{.*\}", texto_respuesta, re.DOTALL)
            if match:
                datos = json.loads(match.group(0))
                return {
                    "valida": bool(datos.get("valida", True)),
                    "calidad": str(datos.get("calidad", "media")),
                    "diagnostico": str(datos.get("diagnostico", "Muestra procesada por IA.")),
                    "recomendacion": str(datos.get("recomendacion", "Sin recomendaciones."))
                }
            else:
                return {
                    "valida": True,
                    "calidad": "media",
                    "diagnostico": texto_respuesta[:160] if texto_respuesta else "Postura analizada.",
                    "recomendacion": "Mantenga la mano estable y centrada."
                }

        except Exception as e:
            return {
                "valida": True,
                "calidad": "desconocida",
                "diagnostico": f"No se pudo completar la consulta a Gemini: {e}",
                "recomendacion": "Verifique su conexión a internet."
            }

    def estructurar_oracion_natural(self, tokens_glosas: List[str]) -> str:
        """
        Convierte una secuencia cruda de tokens de señas (glosas) en una oración
        natural, coherente y bien redactada en español.
        Ejemplo: ['HOLA', 'COMO', 'ESTAS', 'TU'] -> '¡Hola! ¿Cómo estás tú?'
        """
        if not tokens_glosas:
            return ""

        texto_glosas = " ".join(tokens_glosas)

        if not self.esta_disponible():
            return texto_glosas.capitalize() + "."

        try:
            prompt = (
                f"Eres un intérprete profesional de Lenguaje de Señas a español escrito.\n"
                f"Convierte la siguiente secuencia de glosas o palabras reconocidas en una oración en español natural, "
                f"gramaticalmente correcta, fluida y con puntuación adecuada (signos de interrogación, mayúsculas, comas).\n"
                f"Si hay palabras redundantes o aisladas, adáptalas al contexto de forma natural.\n\n"
                f"Secuencia de señas: \"{texto_glosas}\"\n\n"
                f"Devuelve ÚNICAMENTE la oración final traducida, sin explicaciones ni comillas adicionales."
            )

            resp = self.client.models.generate_content(
                model=self.modelo,
                contents=prompt
            )
            oracion = resp.text.strip() if resp and resp.text else texto_glosas
            return oracion.strip('"\'')
        except Exception:
            return texto_glosas.capitalize() + "."


# Bloque de ejecución directa para pruebas de diagnóstico desde consola
if __name__ == "__main__":
    print("=" * 60)
    print("        DIAGNÓSTICO DEL MÓDULO IA_ASISTENTE (GEMINI)")
    print("=" * 60)
    print(f"[*] google.genai disponible: {'SÍ' if HAY_GENAI else 'NO'}")
    asistente = AsistenteIA()
    print(f"[*] API Key configurada: {'SÍ (' + asistente.api_key[:6] + '...)' if asistente.api_key else 'NO'}")
    ok, detalle = asistente.verificar_conexion()
    print(f"[*] Estado de conexión: {'[OK] CONECTADO' if ok else '[!] DESCONECTADO'}")
    print(f"[*] Detalle: {detalle}")

    if ok:
        print("\n[*] Probando estructuración de oración...")
        ejemplo = ["HOLA", "AMIGO", "COMO", "ESTAR", "HOY"]
        resultado = asistente.estructurar_oracion_natural(ejemplo)
        print(f"    Glosas: {ejemplo}")
        print(f"    Oración fluida: {resultado}")
    print("=" * 60)
