"""
Módulo de Gestión de Base de Datos SQLite para el Sistema de Lenguaje de Señas.
Maneja tablas, índices, inserciones y consultas históricas de señas, muestras y modelos.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from config import RUTA_DB


def get_db_connection() -> sqlite3.Connection:
    """Crea y retorna una conexión a la base de datos con row_factory habilitado."""
    conexion = sqlite3.connect(RUTA_DB)
    conexion.row_factory = sqlite3.Row
    conexion.execute("PRAGMA foreign_keys = ON;")
    return conexion


def inicializar_db() -> None:
    """Crea las tablas e índices si no existen en SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Tabla: senas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS senas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                tipo TEXT NOT NULL CHECK(tipo IN ('estatica', 'dinamica')),
                descripcion TEXT,
                fecha TEXT NOT NULL
            );
        """)

        # Tabla: muestras (para señas estáticas: imagen y landmarks)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS muestras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sena_id INTEGER NOT NULL,
                conjunto TEXT NOT NULL CHECK(conjunto IN ('entrenamiento', 'validacion', 'prueba')),
                ruta_imagen TEXT,
                ruta_landmarks TEXT NOT NULL,
                fecha TEXT NOT NULL,
                FOREIGN KEY (sena_id) REFERENCES senas (id) ON DELETE CASCADE
            );
        """)

        # Tabla: secuencias (para señas dinámicas: series temporales de N frames)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS secuencias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sena_id INTEGER NOT NULL,
                conjunto TEXT NOT NULL CHECK(conjunto IN ('entrenamiento', 'validacion', 'prueba')),
                ruta_npy TEXT NOT NULL,
                num_frames INTEGER NOT NULL,
                fecha TEXT NOT NULL,
                FOREIGN KEY (sena_id) REFERENCES senas (id) ON DELETE CASCADE
            );
        """)

        # Tabla: traducciones (registro en vivo)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traducciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                predicciones_json TEXT,
                fecha TEXT NOT NULL
            );
        """)

        # Tabla: modelos (registro de versiones entrenadas)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modelos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ruta_modelo TEXT NOT NULL,
                etiquetas_json TEXT NOT NULL,
                accuracy REAL,
                fecha TEXT NOT NULL
            );
        """)

        # Tabla: configuracion (clave-valor general)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuracion (
                clave TEXT PRIMARY KEY,
                valor TEXT NOT NULL
            );
        """)

        # Índices requeridos para optimizar consultas por seña, conjunto y fecha
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_muestras_sena_id ON muestras(sena_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_muestras_conjunto ON muestras(conjunto);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_muestras_fecha ON muestras(fecha);")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secuencias_sena_id ON secuencias(sena_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secuencias_conjunto ON secuencias(conjunto);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secuencias_fecha ON secuencias(fecha);")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_traducciones_fecha ON traducciones(fecha);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_modelos_fecha ON modelos(fecha);")

        conn.commit()


def registrar_o_obtener_sena(nombre: str, tipo: str = "estatica", descripcion: str = "") -> int:
    """Registra una seña si no existe o devuelve su ID actual."""
    nombre_normalizado = nombre.strip().upper()
    ahora = datetime.now().isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM senas WHERE nombre = ?", (nombre_normalizado,))
        fila = cursor.fetchone()
        if fila:
            return fila["id"]

        cursor.execute(
            "INSERT INTO senas (nombre, tipo, descripcion, fecha) VALUES (?, ?, ?, ?)",
            (nombre_normalizado, tipo, descripcion, ahora)
        )
        conn.commit()
        return cursor.lastrowid


def guardar_muestra(sena_id: int, conjunto: str, ruta_imagen: Optional[str], ruta_landmarks: str) -> int:
    """Guarda el registro de una muestra individual estática."""
    ahora = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO muestras (sena_id, conjunto, ruta_imagen, ruta_landmarks, fecha)
            VALUES (?, ?, ?, ?, ?)
        """, (sena_id, conjunto, str(ruta_imagen) if ruta_imagen else None, str(ruta_landmarks), ahora))
        conn.commit()
        return cursor.lastrowid


def guardar_secuencia(sena_id: int, conjunto: str, ruta_npy: str, num_frames: int) -> int:
    """Guarda el registro de una secuencia dinámica."""
    ahora = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO secuencias (sena_id, conjunto, ruta_npy, num_frames, fecha)
            VALUES (?, ?, ?, ?, ?)
        """, (sena_id, conjunto, str(ruta_npy), num_frames, ahora))
        conn.commit()
        return cursor.lastrowid


def eliminar_ultima_muestra(sena_id: int) -> Optional[str]:
    """Elimina la última muestra o secuencia registrada para una seña y devuelve la ruta de archivo para borrarlo."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Buscar en secuencias primero
        cursor.execute("""
            SELECT id, ruta_npy FROM secuencias WHERE sena_id = ? ORDER BY id DESC LIMIT 1
        """, (sena_id,))
        secuencia = cursor.fetchone()
        if secuencia:
            cursor.execute("DELETE FROM secuencias WHERE id = ?", (secuencia["id"],))
            conn.commit()
            return secuencia["ruta_npy"]

        # Si no hay en secuencias, buscar en muestras
        cursor.execute("""
            SELECT id, ruta_landmarks, ruta_imagen FROM muestras WHERE sena_id = ? ORDER BY id DESC LIMIT 1
        """, (sena_id,))
        muestra = cursor.fetchone()
        if muestra:
            cursor.execute("DELETE FROM muestras WHERE id = ?", (muestra["id"],))
            conn.commit()
            return muestra["ruta_landmarks"]

    return None


def guardar_traduccion(texto: str, predicciones: Optional[List[Dict[str, Any]]] = None) -> int:
    """Registra una oración o seña traducida en el histórico."""
    ahora = datetime.now().isoformat()
    pred_json = json.dumps(predicciones, ensure_ascii=False) if predicciones else None
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO traducciones (texto, predicciones_json, fecha)
            VALUES (?, ?, ?)
        """, (texto, pred_json, ahora))
        conn.commit()
        return cursor.lastrowid


def registrar_modelo(ruta_modelo: str, etiquetas: List[str], accuracy: float) -> int:
    """Registra una versión de modelo entrenado en la base de datos."""
    ahora = datetime.now().isoformat()
    etiquetas_json = json.dumps(etiquetas, ensure_ascii=False)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO modelos (ruta_modelo, etiquetas_json, accuracy, fecha)
            VALUES (?, ?, ?, ?)
        """, (str(ruta_modelo), etiquetas_json, float(accuracy), ahora))
        conn.commit()
        return cursor.lastrowid


def obtener_todas_las_senas() -> List[Dict[str, Any]]:
    """Devuelve la lista completa de señas registradas."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM senas ORDER BY nombre ASC")
        return [dict(row) for row in cursor.fetchall()]


def obtener_conteo_muestras_por_sena(sena_id: int) -> Dict[str, int]:
    """Obtiene el conteo de muestras por conjunto (entrenamiento/validación) para una seña."""
    conteo = {"entrenamiento": 0, "validacion": 0, "total": 0}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Conteo en muestras
        cursor.execute("""
            SELECT conjunto, COUNT(*) as c FROM muestras WHERE sena_id = ? GROUP BY conjunto
        """, (sena_id,))
        for fila in cursor.fetchall():
            conteo[fila["conjunto"]] = conteo.get(fila["conjunto"], 0) + fila["c"]

        # Conteo en secuencias
        cursor.execute("""
            SELECT conjunto, COUNT(*) as c FROM secuencias WHERE sena_id = ? GROUP BY conjunto
        """, (sena_id,))
        for fila in cursor.fetchall():
            conteo[fila["conjunto"]] = conteo.get(fila["conjunto"], 0) + fila["c"]

        conteo["total"] = sum(v for k, v in conteo.items() if k != "total")
    return conteo


def obtener_estadisticas_generales() -> Dict[str, Any]:
    """Recopila estadísticas globales del sistema para visualización."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM senas")
        total_senas = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM muestras")
        total_muestras = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM secuencias")
        total_secuencias = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as total FROM traducciones")
        total_traducciones = cursor.fetchone()["total"]

        cursor.execute("SELECT * FROM modelos ORDER BY id DESC LIMIT 1")
        ultimo_modelo = cursor.fetchone()

        cursor.execute("""
            SELECT s.nombre, s.tipo,
                   (SELECT COUNT(*) FROM muestras m WHERE m.sena_id = s.id) as estaticas,
                   (SELECT COUNT(*) FROM secuencias sq WHERE sq.sena_id = s.id) as dinamicas
            FROM senas s
            ORDER BY s.nombre ASC
        """)
        detalle_senas = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM traducciones ORDER BY id DESC LIMIT 10")
        ultimas_traducciones = [dict(row) for row in cursor.fetchall()]

    return {
        "total_senas": total_senas,
        "total_muestras": total_muestras,
        "total_secuencias": total_secuencias,
        "total_traducciones": total_traducciones,
        "ultimo_modelo": dict(ultimo_modelo) if ultimo_modelo else None,
        "detalle_senas": detalle_senas,
        "ultimas_traducciones": ultimas_traducciones
    }
