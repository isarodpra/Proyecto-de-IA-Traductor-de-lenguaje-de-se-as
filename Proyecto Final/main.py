"""
Módulo Principal y Punto de Entrada de la Aplicación.
Presenta el menú de consola interactivo y gestiona el ciclo de vida del sistema.
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Suprimir logs verbosos de TensorFlow y advertencias de scikit-learn
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import config
import db
from captura import iniciar_interfaz_captura, ejecutar_captura_en_vivo
from entrenamiento import ejecutar_entrenamiento
from reconocimiento import iniciar_reconocimiento


def lanzar_gui():
    """Lanza la interfaz gráfica de usuario en Tkinter."""
    try:
        from interfaz_gui import iniciar_interfaz_grafica
        iniciar_interfaz_grafica()
    except Exception as e:
        print(f"[!] No se pudo iniciar la interfaz gráfica: {e}")
        print("[*] Iniciando en modo consola alternativo...")
        menu_principal()


def capturar_pruebas():
    """Permite capturar muestras etiquetadas directamente en el conjunto de prueba."""
    print("\n" + "=" * 55)
    print("         CAPTURA DE MUESTRAS DE PRUEBA (TEST)")
    print("=" * 55)

    senas = db.obtener_todas_las_senas()
    if not senas:
        print("[!] No hay señas registradas. Primero registre una seña con la opción 1.")
        return

    print("Señas disponibles:")
    for idx, s in enumerate(senas, 1):
        print(f"  {idx}. {s['nombre']} ({s['tipo']})")

    opc = input("\nSeleccione el número de la seña para capturar pruebas: ").strip()
    try:
        seleccion = int(opc) - 1
        if 0 <= seleccion < len(senas):
            sena_elegida = senas[seleccion]
            print(f"\n[*] Iniciando captura de prueba para: {sena_elegida['nombre']}")
            # Ejecutar captura con conjunto fijado
            ejecutar_captura_en_vivo(sena_elegida["id"], sena_elegida["nombre"], sena_elegida["tipo"])
        else:
            print("[!] Selección fuera de rango.")
    except ValueError:
        print("[!] Entrada inválida.")


def mostrar_estadisticas_y_registros():
    """Consulta la base de datos SQLite y muestra estadísticas formateadas."""
    print("\n" + "=" * 60)
    print("           ESTADÍSTICAS DEL SISTEMA Y REGISTROS")
    print("=" * 60)

    stats = db.obtener_estadisticas_generales()

    print(f"\n[RESUMEN GENERAL]")
    print(f"  - Señas registradas en catálogo: {stats['total_senas']}")
    print(f"  - Muestras estáticas guardadas:  {stats['total_muestras']}")
    print(f"  - Secuencias dinámicas guardadas: {stats['total_secuencias']}")
    print(f"  - Traducciones guardadas:        {stats['total_traducciones']}")

    # Información del último modelo entrenado
    print(f"\n[ÚLTIMO MODELO ENTRENADO]")
    mod = stats["ultimo_modelo"]
    if mod:
        print(f"  - Archivo:   {Path(mod['ruta_modelo']).name}")
        print(f"  - Accuracy:  {mod['accuracy'] * 100:.2f}%")
        print(f"  - Fecha:     {mod['fecha']}")
        print(f"  - Etiquetas: {mod['etiquetas_json']}")
    else:
        print("  - Ningún modelo registrado aún.")

    # Detalle de muestras por clase
    print(f"\n[DETALLE POR CLASE]")
    print(f"  {'NOMBRE':<18} | {'TIPO':<10} | {'ESTÁTICAS':<10} | {'DINÁMICAS':<10}")
    print("  " + "-" * 54)
    for s in stats["detalle_senas"]:
        print(f"  {s['nombre']:<18} | {s['tipo']:<10} | {s['estaticas']:<10} | {s['dinamicas']:<10}")

    # Últimas traducciones realizadas
    print(f"\n[HISTORIAL RECIENTE DE TRADUCCIONES]")
    if stats["ultimas_traducciones"]:
        for t in stats["ultimas_traducciones"]:
            fecha_corta = t['fecha'].split('T')[0] if 'T' in t['fecha'] else t['fecha']
            print(f"  [{fecha_corta}] {t['texto']}")
    else:
        print("  - No hay oraciones registradas en el historial.")
    print("=" * 60)


def menu_principal():
    """Bucle principal de la interfaz de consola."""
    # Inicialización de la base de datos al arranque
    db.inicializar_db()

    while True:
        print("\n" + "=" * 48)
        print(" SISTEMA DE RECONOCIMIENTO DE LENGUAJE DE SEÑAS")
        print("=" * 48)
        print(" 1. Registrar señas (Consola + OpenCV)")
        print(" 2. Entrenar modelo")
        print(" 3. Reconocer y traducir en vivo")
        print(" 4. Capturar pruebas")
        print(" 5. Ver estadísticas/registros")
        print(" 6. 🚀 ABRIR INTERFAZ GRÁFICA (GUI TKINTER)")
        print(" 0. Salir")
        print("=" * 48)

        opcion = input(">> Seleccione una opción [0-6]: ").strip()

        if opcion == "1":
            iniciar_interfaz_captura()
        elif opcion == "2":
            ejecutar_entrenamiento()
        elif opcion == "3":
            iniciar_reconocimiento()
        elif opcion == "4":
            capturar_pruebas()
        elif opcion == "5":
            mostrar_estadisticas_y_registros()
        elif opcion == "6":
            lanzar_gui()
        elif opcion == "0":
            print("\nCerrando el sistema. ¡Hasta pronto!\n")
            sys.exit(0)
        else:
            print("[!] Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        menu_principal()
    else:
        lanzar_gui()
