"""
Interfaz Gráfica de Usuario (GUI) Moderna en Tkinter para el Sistema de Traducción
de Lenguaje de Señas en Tiempo Real.

Integra:
- Visor de cámara en vivo con landmarks y visualización Canny/Esqueleto.
- Transcripción y traducción en lenguaje natural con síntesis de voz (TTS).
- Registro guiado de muestras estáticas y dinámicas (con cuenta regresiva).
- Estudio de entrenamiento en segundo plano con consola de progreso interactiva.
- Historial relacional en SQLite con exportación a CSV y reportes de texto.
- Diccionario visual bidireccional (Texto a Señas).
- Panel de configuración y calibración de umbrales en tiempo real.
"""

import os
import warnings

# Suprimir logs verbosos de TensorFlow y advertencias de scikit-learn
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import sys
import time
import json
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import config
import db
from landmarks import (
    ExtractorLandmarks,
    generar_panel_derecho,
    aplicar_filtro_clahe,
    normalizar_vector_landmarks,
    validar_calidad_landmarks
)
from traduccion import TraductorSecuencias, MotorVozAsincrono
from entrenamiento import (
    ejecutar_entrenamiento,
    extraer_caracteristicas_dinamicas,
    RUTA_MODELO_JOBLIB
)

# Intentar soporte joblib para inferencia
try:
    import joblib
    HAY_JOBLIB = True
except ImportError:
    HAY_JOBLIB = False


# =============================================================================
# PALETA DE COLORES Y ESTILOS MODERNOS (DARK THEME)
# =============================================================================
COLOR_BG_DARK = "#11131F"        # Fondo principal
COLOR_PANEL_DARK = "#1A1D2E"     # Fondo de tarjetas y paneles
COLOR_INPUT_DARK = "#262A40"     # Controles, entradas y botones neutros
COLOR_ACCENT = "#6366F1"         # Índigo vibrante
COLOR_ACCENT_HOVER = "#4F46E5"   # Hover de acento
COLOR_SUCCESS = "#10B981"        # Esmeralda / OK
COLOR_DANGER = "#EF4444"         # Rojo / Cancelar / Detener
COLOR_WARNING = "#F59E0B"        # Ámbar / Alerta
COLOR_INFO = "#06B6D4"           # Cian informativo
COLOR_TEXT_LIGHT = "#F9FAFB"     # Texto blanco brillante
COLOR_TEXT_MUTED = "#9CA3AF"     # Texto secundario / gris
COLOR_BORDER = "#31364F"         # Bordes suaves

FONT_MAIN = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_HEADER = ("Segoe UI", 18, "bold")
FONT_BIG_TEXT = ("Segoe UI", 13)
FONT_CONSOLE = ("Consolas", 9)


class InterfazTraductorApp:
    """Ventana Principal y Controlador de la Aplicación de Escritorio."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Traductor de Lenguaje de Señas en Tiempo Real — IA & Visión")
        self.root.geometry("1240x820")
        self.root.minsize(1050, 720)
        self.root.configure(bg=COLOR_BG_DARK)

        # Inicializar base de datos SQLite
        db.inicializar_db()

        # Componentes de inferencia y visión
        self.extractor = ExtractorLandmarks(
            usar_mediapipe=True,
            usar_clahe=config.USAR_FILTRO_CLAHE
        )
        self.traductor = TraductorSecuencias()
        self.modelo_estatico = None
        self.clases_estaticas: List[str] = []
        self.modelo_dinamico = None
        self.clases_dinamicas: List[str] = []
        self._cargar_modelos()

        # Variables de control de cámara y flujo
        self.camara_activa = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.hilo_camara: Optional[threading.Thread] = None
        self.bloqueo_frame = threading.Lock()
        self.ultimo_frame_pil: Optional[ImageTk.PhotoImage] = None
        self.frame_bgr_reciente: Optional[np.ndarray] = None
        self.vector_caracteristicas_reciente: Optional[np.ndarray] = None
        self.results_mediapipe_reciente: Any = None
        self.calidad_muestra_reciente: Tuple[bool, str] = (False, "Sin cámara")

        # Modos de visualización
        self.modo_visualizacion = "LANDMARKS"  # "NORMAL", "LANDMARKS", "CANNY", "ESQUELETO"
        self.modo_reconocimiento = config.MODO_AUTO

        # Variables de la máquina de estados en vivo
        self.estado_en_vivo = "REPOSO"
        self.tiempo_inicio_cooldown = 0.0
        self.tiempo_inicio_postura = 0.0
        self.postura_en_observacion: Optional[str] = None
        self.ultima_seña_detectada = ""
        self.confianza_actual = 0.0
        self.buffer_secuencia_dinamica: List[np.ndarray] = []
        self.buffer_manos_previas: List[np.ndarray] = []
        self.trayectoria_centroides_mano: List[Tuple[float, float]] = []

        # Variables para pestaña de captura
        self.grabando_dinamica = False
        self.cuenta_regresiva = 0
        self.frames_grabados_dinamicos: List[np.ndarray] = []
        self.total_muestras_guardadas = 0

        # Registro de sesión para guardar en SQLite
        self.predicciones_sesion: List[Dict[str, Any]] = []

        # Configurar estilos ttk
        self._configurar_estilos_ttk()

        # Construir Interfaz Gráfica
        self._construir_encabezado()
        self._construir_notebook_pestanas()
        self._construir_barra_estado()

        # Protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)

        # Iniciar bucle de refresco de video en la GUI
        self.root.after(30, self._bucle_actualizar_gui)

    # =========================================================================
    # CARGA DE MODELOS
    # =========================================================================
    def _cargar_modelos(self):
        """Carga el modelo híbrido si existe."""
        if HAY_JOBLIB and RUTA_MODELO_JOBLIB.exists():
            try:
                paquete = joblib.load(str(RUTA_MODELO_JOBLIB))
                self.modelo_estatico = paquete.get("modelo_estatico")
                self.clases_estaticas = paquete.get("clases_estaticas", [])
                self.modelo_dinamico = paquete.get("modelo_dinamico")
                self.clases_dinamicas = paquete.get("clases_dinamicas", [])
                print(f"[OK] Modelo híbrido cargado: {len(self.clases_estaticas)} estáticas, {len(self.clases_dinamicas)} dinámicas.")
            except Exception as e:
                print(f"[!] Error al cargar modelo joblib: {e}")

    # =========================================================================
    # CONFIGURACIÓN DE ESTILOS VISUALES
    # =========================================================================
    def _configurar_estilos_ttk(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Estilo para Notebook y Pestañas
        self.style.configure("TNotebook", background=COLOR_BG_DARK, borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            background=COLOR_PANEL_DARK,
            foreground=COLOR_TEXT_MUTED,
            padding=[16, 8],
            font=FONT_BOLD,
            borderwidth=0
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", COLOR_ACCENT), ("active", COLOR_INPUT_DARK)],
            foreground=[("selected", COLOR_TEXT_LIGHT), ("active", COLOR_TEXT_LIGHT)]
        )

        # Barra de progreso
        self.style.configure(
            "Horizontal.TProgressbar",
            background=COLOR_SUCCESS,
            troughcolor=COLOR_INPUT_DARK,
            borderwidth=0,
            thickness=14
        )

        # Treeview (Tablas)
        self.style.configure(
            "Treeview",
            background=COLOR_PANEL_DARK,
            foreground=COLOR_TEXT_LIGHT,
            fieldbackground=COLOR_PANEL_DARK,
            rowheight=26,
            font=FONT_MAIN,
            borderwidth=0
        )
        self.style.configure(
            "Treeview.Heading",
            background=COLOR_INPUT_DARK,
            foreground=COLOR_TEXT_LIGHT,
            font=FONT_BOLD,
            borderwidth=1,
            relief="flat"
        )
        self.style.map("Treeview", background=[("selected", COLOR_ACCENT)])

    # =========================================================================
    # ENCABEZADO SUPERIOR
    # =========================================================================
    def _construir_encabezado(self):
        header_frame = tk.Frame(self.root, bg=COLOR_PANEL_DARK, height=64)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        # Logo / Título
        title_box = tk.Frame(header_frame, bg=COLOR_PANEL_DARK)
        title_box.pack(side=tk.LEFT, padx=18, pady=10)

        lbl_logo = tk.Label(
            title_box,
            text="🤟",
            font=("Segoe UI Emoji", 22),
            bg=COLOR_PANEL_DARK,
            fg=COLOR_ACCENT
        )
        lbl_logo.pack(side=tk.LEFT, padx=(0, 10))

        text_subbox = tk.Frame(title_box, bg=COLOR_PANEL_DARK)
        text_subbox.pack(side=tk.LEFT)

        lbl_titulo = tk.Label(
            text_subbox,
            text="TRADUCTOR DE LENGUAJE DE SEÑAS",
            font=FONT_TITLE,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        )
        lbl_titulo.pack(anchor="w")

        lbl_sub = tk.Label(
            text_subbox,
            text="Visión por Computadora · MediaPipe · Machine Learning en Tiempo Real",
            font=("Segoe UI", 8),
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_MUTED
        )
        lbl_sub.pack(anchor="w")

        # Botón rápido para activar/desactivar cámara
        self.btn_camara_master = tk.Button(
            header_frame,
            text="▶ INICIAR CÁMARA",
            font=FONT_BOLD,
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            activebackground="#059669",
            activeforeground=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.alternar_camara
        )
        self.btn_camara_master.pack(side=tk.RIGHT, padx=18, pady=12)

        # Indicador de estado de la cámara
        self.lbl_indicador_camara = tk.Label(
            header_frame,
            text="🔴 Cámara Detenida",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_DANGER
        )
        self.lbl_indicador_camara.pack(side=tk.RIGHT, padx=12, pady=12)

    # =========================================================================
    # PESTAÑAS (NOTEBOOK)
    # =========================================================================
    def _construir_notebook_pestanas(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        # Crear contenedores para cada pestaña
        self.tab_traduccion = tk.Frame(self.notebook, bg=COLOR_BG_DARK)
        self.tab_captura = tk.Frame(self.notebook, bg=COLOR_BG_DARK)
        self.tab_entrenamiento = tk.Frame(self.notebook, bg=COLOR_BG_DARK)
        self.tab_historial = tk.Frame(self.notebook, bg=COLOR_BG_DARK)
        self.tab_diccionario = tk.Frame(self.notebook, bg=COLOR_BG_DARK)
        self.tab_configuracion = tk.Frame(self.notebook, bg=COLOR_BG_DARK)

        self.notebook.add(self.tab_traduccion, text="  🎥 Traducción en Vivo  ")
        self.notebook.add(self.tab_captura, text="  📸 Registro y Captura  ")
        self.notebook.add(self.tab_entrenamiento, text="  🧠 Entrenamiento  ")
        self.notebook.add(self.tab_historial, text="  📊 Historial & BD  ")
        self.notebook.add(self.tab_diccionario, text="  📖 Diccionario Visual  ")
        self.notebook.add(self.tab_configuracion, text="  ⚙️ Configuración  ")

        # Construir contenido de cada pestaña
        self._construir_tab_traduccion()
        self._construir_tab_captura()
        self._construir_tab_entrenamiento()
        self._construir_tab_historial()
        self._construir_tab_diccionario()
        self._construir_tab_configuracion()

    # =========================================================================
    # 1. PESTAÑA: TRADUCCIÓN EN VIVO
    # =========================================================================
    def _construir_tab_traduccion(self):
        container = tk.Frame(self.tab_traduccion, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Panel izquierdo: Feed de Video
        left_panel = tk.Frame(container, bg=COLOR_PANEL_DARK, bd=1, relief=tk.FLAT)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Barra superior de controles del video
        top_ctrl = tk.Frame(left_panel, bg=COLOR_PANEL_DARK)
        top_ctrl.pack(fill=tk.X, padx=12, pady=8)

        lbl_feed = tk.Label(
            top_ctrl,
            text="VISOR DE CÁMARA & DETECCIÓN",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        )
        lbl_feed.pack(side=tk.LEFT)

        # Selector de Vista
        lbl_vista = tk.Label(top_ctrl, text="Vista:", font=FONT_MAIN, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_MUTED)
        lbl_vista.pack(side=tk.LEFT, padx=(16, 4))

        self.combo_vista = ttk.Combobox(
            top_ctrl,
            values=["LANDMARKS", "NORMAL", "CANNY", "ESQUELETO"],
            state="readonly",
            width=12
        )
        self.combo_vista.set("LANDMARKS")
        self.combo_vista.pack(side=tk.LEFT)
        self.combo_vista.bind("<<ComboboxSelected>>", self._al_cambiar_vista)

        # Selector de Modo de Reconocimiento
        lbl_modo = tk.Label(top_ctrl, text="Modo:", font=FONT_MAIN, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_MUTED)
        lbl_modo.pack(side=tk.LEFT, padx=(16, 4))

        self.combo_modo = ttk.Combobox(
            top_ctrl,
            values=[config.MODO_AUTO, config.MODO_SOLO_ESTATICO, config.MODO_SOLO_DINAMICO],
            state="readonly",
            width=16
        )
        self.combo_modo.set(self.modo_reconocimiento)
        self.combo_modo.pack(side=tk.LEFT)
        self.combo_modo.bind("<<ComboboxSelected>>", self._al_cambiar_modo)

        # Canvas para el Frame de Video
        self.canvas_video = tk.Canvas(left_panel, bg="#0A0B10", highlightthickness=0)
        self.canvas_video.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        # Barra de confianza y estado debajo del video
        bar_frame = tk.Frame(left_panel, bg=COLOR_PANEL_DARK)
        bar_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.lbl_estado_badge = tk.Label(
            bar_frame,
            text="ESTADO: REPOSO",
            font=FONT_BOLD,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_MUTED,
            padx=10,
            pady=4
        )
        self.lbl_estado_badge.pack(side=tk.LEFT)

        self.barra_confianza = ttk.Progressbar(
            bar_frame,
            orient="horizontal",
            mode="determinate",
            style="Horizontal.TProgressbar"
        )
        self.barra_confianza.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.lbl_confianza_val = tk.Label(
            bar_frame,
            text="Confianza: 0%",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            width=14
        )
        self.lbl_confianza_val.pack(side=tk.RIGHT)

        # Panel derecho: Transcripción, Traducción y Controles
        right_panel = tk.Frame(container, bg=COLOR_PANEL_DARK, width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_panel.pack_propagate(False)

        # Tarjeta: Última Seña Detectada
        card_sena = tk.Frame(right_panel, bg=COLOR_INPUT_DARK, bd=0)
        card_sena.pack(fill=tk.X, padx=14, pady=14)

        lbl_sena_tit = tk.Label(
            card_sena,
            text="SEÑA RECONOCIDA",
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_MUTED
        )
        lbl_sena_tit.pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_sena_grande = tk.Label(
            card_sena,
            text="---",
            font=("Segoe UI", 26, "bold"),
            bg=COLOR_INPUT_DARK,
            fg=COLOR_SUCCESS
        )
        self.lbl_sena_grande.pack(anchor="w", padx=12, pady=(0, 8))

        # Tarjeta: Oración Traducida
        lbl_traduccion_tit = tk.Label(
            right_panel,
            text="TRADUCCIÓN EN TIEMPO REAL (ESPAÑOL)",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        )
        lbl_traduccion_tit.pack(anchor="w", padx=14, pady=(6, 4))

        self.txt_traduccion = tk.Text(
            right_panel,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            font=FONT_BIG_TEXT,
            wrap=tk.WORD,
            height=8,
            bd=0,
            padx=10,
            pady=10
        )
        self.txt_traduccion.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        # Botones de Acción de Traducción
        actions_grid = tk.Frame(right_panel, bg=COLOR_PANEL_DARK)
        actions_grid.pack(fill=tk.X, padx=14, pady=(0, 10))

        btn_hablar = tk.Button(
            actions_grid,
            text="🔊 Hablar (Voz)",
            font=FONT_BOLD,
            bg=COLOR_ACCENT,
            fg=COLOR_TEXT_LIGHT,
            activebackground=COLOR_ACCENT_HOVER,
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.accion_hablar_oracion
        )
        btn_hablar.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)

        btn_guardar = tk.Button(
            actions_grid,
            text="💾 Guardar en BD",
            font=FONT_BOLD,
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            activebackground="#059669",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.accion_guardar_oracion
        )
        btn_guardar.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=3)

        btn_borrar = tk.Button(
            actions_grid,
            text="⌫ Borrar Última",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            pady=5,
            cursor="hand2",
            command=self.accion_borrar_ultima_palabra
        )
        btn_borrar.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=3)

        btn_limpiar = tk.Button(
            actions_grid,
            text="🧹 Limpiar Todo",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            pady=5,
            cursor="hand2",
            command=self.accion_limpiar_oracion
        )
        btn_limpiar.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=3)

        actions_grid.columnconfigure(0, weight=1)
        actions_grid.columnconfigure(1, weight=1)

        # Métricas de rendimiento
        metrics_box = tk.Frame(right_panel, bg=COLOR_INPUT_DARK, padx=10, pady=8)
        metrics_box.pack(fill=tk.X, padx=14, pady=(0, 14))

        self.lbl_metricas = tk.Label(
            metrics_box,
            text="FPS: -- | Inferencia: -- ms | Palabras: 0",
            font=("Segoe UI", 9),
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_metricas.pack(anchor="w")

    # =========================================================================
    # 2. PESTAÑA: REGISTRO Y CAPTURA
    # =========================================================================
    def _construir_tab_captura(self):
        container = tk.Frame(self.tab_captura, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Panel izquierdo: Video para captura
        left_panel = tk.Frame(container, bg=COLOR_PANEL_DARK)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        top_cap_bar = tk.Frame(left_panel, bg=COLOR_PANEL_DARK)
        top_cap_bar.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(
            top_cap_bar,
            text="VISOR DE CAPTURA Y ENCUADRE",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(side=tk.LEFT)

        self.lbl_calidad_badge = tk.Label(
            top_cap_bar,
            text="CALIDAD: PENDIENTE",
            font=FONT_BOLD,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_MUTED,
            padx=8,
            pady=2
        )
        self.lbl_calidad_badge.pack(side=tk.RIGHT)

        self.canvas_captura = tk.Canvas(left_panel, bg="#0A0B10", highlightthickness=0)
        self.canvas_captura.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Panel derecho: Formulario de Registro
        right_panel = tk.Frame(container, bg=COLOR_PANEL_DARK, width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        tk.Label(
            right_panel,
            text="REGISTRO DE SEÑAS Y MUESTRAS",
            font=FONT_TITLE,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(anchor="w", padx=16, pady=(16, 12))

        # Selección: Nueva o Existente
        self.var_modo_seleccion = tk.StringVar(value="NUEVA")
        sel_frame = tk.Frame(right_panel, bg=COLOR_PANEL_DARK)
        sel_frame.pack(fill=tk.X, padx=16, pady=(0, 10))

        tk.Radiobutton(
            sel_frame,
            text="Registrar Nueva Seña",
            variable=self.var_modo_seleccion,
            value="NUEVA",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK,
            activeforeground=COLOR_TEXT_LIGHT,
            command=self._actualizar_estado_formulario_captura
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Radiobutton(
            sel_frame,
            text="Agregar a Existente",
            variable=self.var_modo_seleccion,
            value="EXISTENTE",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK,
            activeforeground=COLOR_TEXT_LIGHT,
            command=self._actualizar_estado_formulario_captura
        ).pack(side=tk.LEFT)

        # Campos del formulario
        form_frame = tk.Frame(right_panel, bg=COLOR_PANEL_DARK)
        form_frame.pack(fill=tk.X, padx=16, pady=4)

        # Nombre de la seña
        tk.Label(form_frame, text="Nombre de la seña:", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w")
        self.ent_nombre_sena = tk.Entry(form_frame, font=FONT_MAIN, bg=COLOR_INPUT_DARK, fg=COLOR_TEXT_LIGHT, bd=0, insertbackground="white")
        self.ent_nombre_sena.pack(fill=tk.X, pady=(4, 10), ipady=6)

        # Selector de seña existente
        tk.Label(form_frame, text="O seleccionar existente:", font=FONT_MAIN, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_MUTED).pack(anchor="w")
        self.combo_senas_existentes = ttk.Combobox(form_frame, state="readonly")
        self.combo_senas_existentes.pack(fill=tk.X, pady=(4, 10))
        self.combo_senas_existentes.bind("<<ComboboxSelected>>", self._al_seleccionar_sena_existente)

        # Tipo de seña
        tk.Label(form_frame, text="Tipo de movimiento:", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w")
        self.var_tipo_sena = tk.StringVar(value="estatica")
        tipo_frame = tk.Frame(form_frame, bg=COLOR_PANEL_DARK)
        tipo_frame.pack(fill=tk.X, pady=(4, 10))

        tk.Radiobutton(
            tipo_frame,
            text="Estática (Letra o postura fija)",
            variable=self.var_tipo_sena,
            value="estatica",
            font=FONT_MAIN,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK
        ).pack(anchor="w")

        tk.Radiobutton(
            tipo_frame,
            text="Dinámica (Gesto con movimiento continuo)",
            variable=self.var_tipo_sena,
            value="dinamica",
            font=FONT_MAIN,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK
        ).pack(anchor="w")

        # Conjunto de destino (Train / Val / Test)
        tk.Label(form_frame, text="Conjunto de datos:", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w")
        self.var_conjunto = tk.StringVar(value="entrenamiento")
        conj_frame = tk.Frame(form_frame, bg=COLOR_PANEL_DARK)
        conj_frame.pack(fill=tk.X, pady=(4, 14))

        for opt in ["entrenamiento", "validacion", "prueba"]:
            tk.Radiobutton(
                conj_frame,
                text=opt.capitalize(),
                variable=self.var_conjunto,
                value=opt,
                font=FONT_MAIN,
                bg=COLOR_PANEL_DARK,
                fg=COLOR_TEXT_LIGHT,
                selectcolor=COLOR_BG_DARK,
                activebackground=COLOR_PANEL_DARK
            ).pack(side=tk.LEFT, padx=(0, 10))

        # Tarjeta de Conteo de Muestras
        self.box_conteo = tk.Frame(right_panel, bg=COLOR_INPUT_DARK, padx=12, pady=10)
        self.box_conteo.pack(fill=tk.X, padx=16, pady=(0, 14))

        self.lbl_conteo_muestras = tk.Label(
            self.box_conteo,
            text="Muestras: Train: 0 | Val: 0 | Total: 0",
            font=FONT_BOLD,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_INFO
        )
        self.lbl_conteo_muestras.pack(anchor="w")

        # Botones de Captura
        self.btn_capturar = tk.Button(
            right_panel,
            text="📷 CAPTURAR MUESTRA",
            font=FONT_BOLD,
            bg=COLOR_ACCENT,
            fg=COLOR_TEXT_LIGHT,
            activebackground=COLOR_ACCENT_HOVER,
            relief=tk.FLAT,
            pady=10,
            cursor="hand2",
            command=self.accion_capturar_muestra
        )
        self.btn_capturar.pack(fill=tk.X, padx=16, pady=(0, 8))

        btn_deshacer = tk.Button(
            right_panel,
            text="↩ Borrar Última Muestra",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_DANGER,
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.accion_borrar_ultima_muestra
        )
        btn_deshacer.pack(fill=tk.X, padx=16, pady=(0, 12))

        # Cargar catálogo de señas existentes
        self._refrescar_combo_senas_existentes()

    # =========================================================================
    # 3. PESTAÑA: ESTUDIO DE ENTRENAMIENTO
    # =========================================================================
    def _construir_tab_entrenamiento(self):
        container = tk.Frame(self.tab_entrenamiento, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Panel izquierdo: Resumen del Catálogo
        left_panel = tk.Frame(container, bg=COLOR_PANEL_DARK, width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(
            left_panel,
            text="CATÁLOGO DE SEÑAS Y REQUISITOS",
            font=FONT_TITLE,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(anchor="w", padx=14, pady=(14, 8))

        # Tabla de clases
        cols = ("Nombre", "Tipo", "Estáticas", "Dinámicas", "Estado")
        self.tree_catalogo = ttk.Treeview(left_panel, columns=cols, show="headings", height=14)
        for col in cols:
            self.tree_catalogo.heading(col, text=col)
            self.tree_catalogo.column(col, width=70, anchor="center")
        self.tree_catalogo.column("Nombre", width=95, anchor="w")
        self.tree_catalogo.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        btn_refrescar_cat = tk.Button(
            left_panel,
            text="🔄 Actualizar Lista de Señas",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self._refrescar_tabla_entrenamiento
        )
        btn_refrescar_cat.pack(fill=tk.X, padx=14, pady=(0, 14))

        # Panel derecho: Parámetros y Consola de Entrenamiento
        right_panel = tk.Frame(container, bg=COLOR_PANEL_DARK)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_panel,
            text="ENTRENAMIENTO DEL MODELO HÍBRIDO (ML / DL)",
            font=FONT_TITLE,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(anchor="w", padx=16, pady=(14, 8))

        # Parámetros rápidos
        param_frame = tk.Frame(right_panel, bg=COLOR_PANEL_DARK)
        param_frame.pack(fill=tk.X, padx=16, pady=(0, 10))

        self.var_aug = tk.BooleanVar(value=True)
        tk.Checkbutton(
            param_frame,
            text="Habilitar Data Augmentation (Ruido, escala y traslación sintética)",
            variable=self.var_aug,
            font=FONT_MAIN,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK
        ).pack(anchor="w")

        # Botón para Iniciar Entrenamiento
        self.btn_iniciar_entrenamiento = tk.Button(
            right_panel,
            text="🚀 INICIAR ENTRENAMIENTO DEL MODELO",
            font=FONT_BOLD,
            bg=COLOR_ACCENT,
            fg=COLOR_TEXT_LIGHT,
            activebackground=COLOR_ACCENT_HOVER,
            relief=tk.FLAT,
            pady=10,
            cursor="hand2",
            command=self.accion_iniciar_entrenamiento
        )
        self.btn_iniciar_entrenamiento.pack(fill=tk.X, padx=16, pady=(6, 10))

        self.progreso_entrenamiento = ttk.Progressbar(right_panel, mode="indeterminate")
        self.progreso_entrenamiento.pack(fill=tk.X, padx=16, pady=(0, 10))

        # Consola de logs
        tk.Label(
            right_panel,
            text="Consola de Progreso en Vivo:",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_MUTED
        ).pack(anchor="w", padx=16, pady=(0, 4))

        self.txt_log_entrenamiento = tk.Text(
            right_panel,
            bg="#0D0E17",
            fg="#A7F3D0",
            font=FONT_CONSOLE,
            bd=0,
            padx=10,
            pady=8
        )
        self.txt_log_entrenamiento.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 14))

        self._refrescar_tabla_entrenamiento()

    # =========================================================================
    # 4. PESTAÑA: HISTORIAL Y BASE DE DATOS
    # =========================================================================
    def _construir_tab_historial(self):
        container = tk.Frame(self.tab_historial, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        header_hist = tk.Frame(container, bg=COLOR_PANEL_DARK)
        header_hist.pack(fill=tk.X, padx=0, pady=(0, 10))

        tk.Label(
            header_hist,
            text="HISTORIAL DE TRADUCCIONES GUARDADAS (SQLite)",
            font=FONT_TITLE,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(side=tk.LEFT, padx=14, pady=12)

        # Botones de exportación
        btn_exp_csv = tk.Button(
            header_hist,
            text="📥 Exportar a CSV",
            font=FONT_MAIN,
            bg=COLOR_ACCENT,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.accion_exportar_csv
        )
        btn_exp_csv.pack(side=tk.RIGHT, padx=(6, 14), pady=12)

        btn_exp_txt = tk.Button(
            header_hist,
            text="📄 Exportar a Reporte TXT",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.accion_exportar_txt
        )
        btn_exp_txt.pack(side=tk.RIGHT, padx=6, pady=12)

        # Tabla de traducciones
        cols = ("ID", "Fecha y Hora", "Texto Traducido")
        self.tree_historial = ttk.Treeview(container, columns=cols, show="headings")
        self.tree_historial.heading("ID", text="ID")
        self.tree_historial.heading("Fecha y Hora", text="Fecha y Hora")
        self.tree_historial.heading("Texto Traducido", text="Texto Traducido")

        self.tree_historial.column("ID", width=60, anchor="center")
        self.tree_historial.column("Fecha y Hora", width=180, anchor="w")
        self.tree_historial.column("Texto Traducido", width=700, anchor="w")
        self.tree_historial.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 10))

        # Barra de acciones inferiores
        bottom_bar = tk.Frame(container, bg=COLOR_PANEL_DARK)
        bottom_bar.pack(fill=tk.X, padx=0, pady=0)

        btn_refrescar_hist = tk.Button(
            bottom_bar,
            text="🔄 Actualizar Historial",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._refrescar_historial
        )
        btn_refrescar_hist.pack(side=tk.LEFT, padx=14, pady=10)

        btn_eliminar_sel = tk.Button(
            bottom_bar,
            text="🗑️ Eliminar Registro Seleccionado",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_DANGER,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.accion_eliminar_traduccion_seleccionada
        )
        btn_eliminar_sel.pack(side=tk.LEFT, padx=6, pady=10)

        btn_vaciar_hist = tk.Button(
            bottom_bar,
            text="⚠️ Vaciar Historial Completo",
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_DANGER,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.accion_vaciar_historial
        )
        btn_vaciar_hist.pack(side=tk.RIGHT, padx=14, pady=10)

        self._refrescar_historial()

    # =========================================================================
    # 5. PESTAÑA: DICCIONARIO VISUAL (TEXTO A SEÑAS)
    # =========================================================================
    def _construir_tab_diccionario(self):
        container = tk.Frame(self.tab_diccionario, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Panel izquierdo: Lista de Señas
        left_panel = tk.Frame(container, bg=COLOR_PANEL_DARK, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(
            left_panel,
            text="DICCIONARIO DE SEÑAS",
            font=FONT_TITLE,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(anchor="w", padx=14, pady=(14, 8))

        # Buscador
        self.ent_buscar_dic = tk.Entry(
            left_panel,
            font=FONT_MAIN,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            bd=0,
            insertbackground="white"
        )
        self.ent_buscar_dic.pack(fill=tk.X, padx=14, pady=(0, 8), ipady=5)
        self.ent_buscar_dic.bind("<KeyRelease>", self._filtrar_diccionario)

        self.listbox_dic = tk.Listbox(
            left_panel,
            bg=COLOR_INPUT_DARK,
            fg=COLOR_TEXT_LIGHT,
            font=FONT_BOLD,
            bd=0,
            selectbackground=COLOR_ACCENT,
            highlightthickness=0
        )
        self.listbox_dic.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self.listbox_dic.bind("<<ListboxSelect>>", self._al_seleccionar_seña_diccionario)

        # Panel derecho: Visualizador de la Seña
        right_panel = tk.Frame(container, bg=COLOR_PANEL_DARK)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_dic_titulo = tk.Label(
            right_panel,
            text="SELECCIONE UNA SEÑA DEL CATÁLOGO",
            font=FONT_HEADER,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_ACCENT
        )
        self.lbl_dic_titulo.pack(anchor="w", padx=18, pady=(16, 4))

        self.lbl_dic_tipo = tk.Label(
            right_panel,
            text="Tipo de Seña: ---",
            font=FONT_BOLD,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_dic_tipo.pack(anchor="w", padx=18, pady=(0, 14))

        # Canvas para mostrar imagen de referencia si existe
        self.canvas_dic_foto = tk.Canvas(right_panel, bg="#0D0E17", width=420, height=360, highlightthickness=0)
        self.canvas_dic_foto.pack(anchor="w", padx=18, pady=(0, 14))

        self.lbl_dic_descripcion = tk.Label(
            right_panel,
            text="Aquí se visualiza la postura o gesto de referencia para practicar la comunicación bidireccional.",
            font=FONT_MAIN,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            wraplength=600,
            justify="left"
        )
        self.lbl_dic_descripcion.pack(anchor="w", padx=18, pady=4)

        self._refrescar_lista_diccionario()

    # =========================================================================
    # 6. PESTAÑA: CONFIGURACIÓN Y CALIBRACIÓN
    # =========================================================================
    def _construir_tab_configuracion(self):
        container = tk.Frame(self.tab_configuracion, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        card = tk.Frame(container, bg=COLOR_PANEL_DARK, padx=20, pady=20)
        card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            card,
            text="CALIBRACIÓN Y CONFIGURACIÓN DEL SISTEMA",
            font=FONT_HEADER,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT
        ).pack(anchor="w", pady=(0, 16))

        # Hardware de Cámara
        tk.Label(card, text="Índice de Cámara (0: integrada, 1: USB externa):", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w", pady=(8, 2))
        self.spn_camara = tk.Spinbox(card, from_=0, to=10, width=8, font=FONT_MAIN, bg=COLOR_INPUT_DARK, fg=COLOR_TEXT_LIGHT, bd=0)
        self.spn_camara.delete(0, "end")
        self.spn_camara.insert(0, str(config.CAMARA_INDEX))
        self.spn_camara.pack(anchor="w", pady=(0, 12))

        # Umbral de Confianza
        tk.Label(card, text="Umbral Mínimo de Confianza (Porcentaje para aceptar predicción):", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w", pady=(8, 2))
        self.scale_confianza = tk.Scale(
            card,
            from_=0.40,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=350,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            highlightthickness=0,
            troughcolor=COLOR_INPUT_DARK
        )
        self.scale_confianza.set(config.UMBRAL_CONFIANZA)
        self.scale_confianza.pack(anchor="w", pady=(0, 12))

        # Tiempos de Cooldown y Hold
        tk.Label(card, text="Tiempo de Confirmación Estática (Segundos con mano quieta):", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w", pady=(8, 2))
        self.scale_estatica = tk.Scale(
            card,
            from_=0.30,
            to=1.50,
            resolution=0.10,
            orient=tk.HORIZONTAL,
            length=350,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            highlightthickness=0,
            troughcolor=COLOR_INPUT_DARK
        )
        self.scale_estatica.set(config.TIEMPO_CONFIRMACION_ESTATICA)
        self.scale_estatica.pack(anchor="w", pady=(0, 12))

        tk.Label(card, text="Tiempo de Cooldown Post-Detección (Espera tras confirmar una seña):", font=FONT_BOLD, bg=COLOR_PANEL_DARK, fg=COLOR_TEXT_LIGHT).pack(anchor="w", pady=(8, 2))
        self.scale_cooldown = tk.Scale(
            card,
            from_=0.50,
            to=3.00,
            resolution=0.25,
            orient=tk.HORIZONTAL,
            length=350,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            highlightthickness=0,
            troughcolor=COLOR_INPUT_DARK
        )
        self.scale_cooldown.set(config.TIEMPO_COOLDOWN_POST_DETECCION)
        self.scale_cooldown.pack(anchor="w", pady=(0, 14))

        # Opciones avanzadas
        self.var_clahe = tk.BooleanVar(value=config.USAR_FILTRO_CLAHE)
        tk.Checkbutton(
            card,
            text="Filtro Adaptativo de Contraste e Iluminación (CLAHE para habitaciones con poca luz)",
            variable=self.var_clahe,
            font=FONT_MAIN,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK
        ).pack(anchor="w", pady=4)

        self.var_voz = tk.BooleanVar(value=config.VOZ_HABILITADA)
        tk.Checkbutton(
            card,
            text="Habilitar Síntesis de Voz Automática (Lectura de señas traducidas)",
            variable=self.var_voz,
            font=FONT_MAIN,
            bg=COLOR_PANEL_DARK,
            fg=COLOR_TEXT_LIGHT,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_DARK
        ).pack(anchor="w", pady=4)

        btn_guardar_cfg = tk.Button(
            card,
            text="💾 Guardar Cambios de Configuración",
            font=FONT_BOLD,
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            relief=tk.FLAT,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.accion_guardar_configuracion
        )
        btn_guardar_cfg.pack(anchor="w", pady=(18, 0))

    # =========================================================================
    # BARRA DE ESTADO INFERIOR
    # =========================================================================
    def _construir_barra_estado(self):
        status_bar = tk.Frame(self.root, bg="#0C0D15", height=28)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status_global = tk.Label(
            status_bar,
            text="Sistema Listo. Presione 'Iniciar Cámara' para comenzar.",
            font=("Segoe UI", 9),
            bg="#0C0D15",
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_status_global.pack(side=tk.LEFT, padx=12, pady=4)

        self.lbl_version_status = tk.Label(
            status_bar,
            text="Versión 2.0 (GUI Tkinter) · MediaPipe & SQLite",
            font=("Segoe UI", 9),
            bg="#0C0D15",
            fg=COLOR_TEXT_MUTED
        )
        self.lbl_version_status.pack(side=tk.RIGHT, padx=12, pady=4)

    # =========================================================================
    # CONTROL DE CÁMARA Y HILO DE ADQUISICIÓN
    # =========================================================================
    def alternar_camara(self):
        if not self.camara_activa:
            self.iniciar_camara()
        else:
            self.detener_camara()

    def iniciar_camara(self):
        try:
            indice = int(self.spn_camara.get())
        except Exception:
            indice = config.CAMARA_INDEX

        from utils import abrir_camara_robusta
        self.cap, idx_real = abrir_camara_robusta(indice, config.FRAME_ANCHO, config.FRAME_ALTO)
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror(
                "Error de Cámara",
                f"No se detectó ninguna cámara web activa (intentado índice #{indice} y escaneo automático).\n\n"
                "Verifique que su cámara web esté conectada y que no esté en uso por otra aplicación."
            )
            return

        # Sincronizar índice si se detectó en otro puerto
        if idx_real != indice:
            try:
                self.spn_camara.delete(0, "end")
                self.spn_camara.insert(0, str(idx_real))
            except Exception:
                pass
            config.CAMARA_INDEX = idx_real

        self.camara_activa = True
        self.btn_camara_master.configure(text="⏹ DETENER CÁMARA", bg=COLOR_DANGER, activebackground="#DC2626")
        self.lbl_indicador_camara.configure(text=f"🟢 Cámara Activa (#{idx_real})", fg=COLOR_SUCCESS)
        if hasattr(self, "lbl_status_global"):
            self.lbl_status_global.configure(text=f"Cámara #{idx_real} conectada y transmitiendo a máxima fluidez.")

        self.hilo_camara = threading.Thread(target=self._bucle_captura_frames, daemon=True)
        self.hilo_camara.start()

    def detener_camara(self):
        self.camara_activa = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_camara_master.configure(text="▶ INICIAR CÁMARA", bg=COLOR_SUCCESS, activebackground="#059669")
        self.lbl_indicador_camara.configure(text="🔴 Cámara Detenida", fg=COLOR_DANGER)
        if hasattr(self, "lbl_status_global"):
            self.lbl_status_global.configure(text="Cámara detenida.")

    def _bucle_captura_frames(self):
        """Hilo de segundo plano para leer la cámara continuamente sin lag en la GUI.
        No usa sleep — deja que la cámara marque su propio ritmo de FPS."""
        while self.camara_activa and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Un solo intento extra antes de rendirse; evita busy-loop en fallo total
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)

            # Procesamiento de landmarks y visión
            vector, results = self.extractor.procesar_frame(frame)
            calidad = validar_calidad_landmarks(vector)

            with self.bloqueo_frame:
                self.frame_bgr_reciente = frame
                self.vector_caracteristicas_reciente = vector
                self.results_mediapipe_reciente = results
                self.calidad_muestra_reciente = calidad
            # Sin sleep: la cámara controla el ritmo; cap.read() bloquea hasta el siguiente frame

    # =========================================================================
    # BUCLE PRINCIPAL DE ACTUALIZACIÓN DE LA GUI (30 FPS)
    # =========================================================================
    def _bucle_actualizar_gui(self):
        """Se ejecuta en el hilo principal de Tkinter cada ~30ms.
        Si el hilo de cámara aún no entregó frames (p.ej. primer frame lento),
        intenta una lectura no bloqueante directamente en el hilo GUI."""
        pestana_actual = self.notebook.index(self.notebook.select())

        if self.camara_activa and self.cap and self.cap.isOpened():
            # --- Modo principal: leer del buffer compartido que llena el hilo de cámara ---
            with self.bloqueo_frame:
                frame_bgr = self.frame_bgr_reciente.copy() if self.frame_bgr_reciente is not None else None
                vector = self.vector_caracteristicas_reciente.copy() if self.vector_caracteristicas_reciente is not None else None
                results = self.results_mediapipe_reciente
                calidad_ok, calidad_msg = self.calidad_muestra_reciente

            # --- Fallback: si el hilo aún no proveyó ningún frame, leer directamente ---
            if frame_bgr is None:
                try:
                    ret, frame_directo = self.cap.read()
                    if ret and frame_directo is not None:
                        frame_directo = cv2.flip(frame_directo, 1)
                        v_dir, r_dir = self.extractor.procesar_frame(frame_directo)
                        c_dir = validar_calidad_landmarks(v_dir)
                        with self.bloqueo_frame:
                            self.frame_bgr_reciente = frame_directo
                            self.vector_caracteristicas_reciente = v_dir
                            self.results_mediapipe_reciente = r_dir
                            self.calidad_muestra_reciente = c_dir
                        frame_bgr = frame_directo
                        vector = v_dir
                        results = r_dir
                        calidad_ok, calidad_msg = c_dir
                except Exception:
                    pass

            if frame_bgr is not None:
                # Procesar según pestaña activa
                if pestana_actual == 0:  # Traducción en vivo
                    self._procesar_frame_traduccion_en_vivo(frame_bgr, vector, results)
                elif pestana_actual == 1:  # Captura
                    self._procesar_frame_captura(frame_bgr, vector, results, calidad_ok, calidad_msg)

        # Reprogramar siguiente frame
        self.root.after(30, self._bucle_actualizar_gui)

    # =========================================================================
    # LÓGICA DE PROCESAMIENTO: TRADUCCIÓN EN VIVO
    # =========================================================================
    def _procesar_frame_traduccion_en_vivo(self, frame_bgr: np.ndarray, vector: np.ndarray, results: Any):
        ahora = time.time()
        geometria = self.extractor.obtener_geometria_mano_rostro(results)
        hay_mano = geometria["hay_mano"]
        hay_rostro = geometria["hay_rostro"]
        mano_activa = geometria["mano_en_zona_activa"]
        centroide_mano = geometria["centroide_mano"]

        # Evaluar velocidad de movimiento
        landmarks_manos = vector[:config.CANTIDAD_LANDMARKS_MANO * 2]
        self.buffer_manos_previas.append(landmarks_manos)
        if len(self.buffer_manos_previas) > 8:
            self.buffer_manos_previas.pop(0)

        velocidad = 0.0
        if len(self.buffer_manos_previas) >= 2:
            diffs = [np.mean(np.abs(self.buffer_manos_previas[i] - self.buffer_manos_previas[i - 1]))
                     for i in range(1, len(self.buffer_manos_previas))]
            velocidad = float(np.mean(diffs))

        # ---------------------------------------------------------------------
        # MÁQUINA DE ESTADOS
        # ---------------------------------------------------------------------
        # 1. COOLDOWN
        if self.estado_en_vivo == "COOLDOWN":
            tiempo_transcurrido = ahora - self.tiempo_inicio_cooldown
            progreso = min(1.0, tiempo_transcurrido / config.TIEMPO_COOLDOWN_POST_DETECCION)
            self.barra_confianza["value"] = (1.0 - progreso) * 100
            self.lbl_confianza_val.configure(text=f"Espera: {max(0.0, config.TIEMPO_COOLDOWN_POST_DETECCION - tiempo_transcurrido):.1f}s")
            self.lbl_estado_badge.configure(text=f"CONFIRMADO: {self.ultima_seña_detectada}", bg=COLOR_SUCCESS, fg=COLOR_TEXT_LIGHT)

            if tiempo_transcurrido >= config.TIEMPO_COOLDOWN_POST_DETECCION:
                self.estado_en_vivo = "LISTO" if hay_mano else "REPOSO"
                self.postura_en_observacion = None
                self.buffer_secuencia_dinamica.clear()
                self.trayectoria_centroides_mano.clear()

        # 2. REPOSO (Sin mano en encuadre)
        elif not hay_mano:
            self.estado_en_vivo = "REPOSO"
            self.traductor.notificar_reposo()
            self.barra_confianza["value"] = 0
            self.lbl_confianza_val.configure(text="Sin mano")
            self.lbl_estado_badge.configure(
                text="REPOSO: Muestre su mano" if not hay_rostro else "ROSTRO DETECTADO: Levante su mano",
                bg=COLOR_INPUT_DARK,
                fg=COLOR_TEXT_MUTED
            )
            self.buffer_secuencia_dinamica.clear()
            self.trayectoria_centroides_mano.clear()

        # 3. MANO PRESENTE
        else:
            modo = self.modo_reconocimiento
            es_estatica = (modo == config.MODO_SOLO_ESTATICO) or (modo == config.MODO_AUTO and velocidad <= config.UMBRAL_MOVIMIENTO_DINAMICO)
            es_dinamica = (modo == config.MODO_SOLO_DINAMICO) or (modo == config.MODO_AUTO and velocidad > config.UMBRAL_MOVIMIENTO_DINAMICO)

            # RAMA DINÁMICA
            if es_dinamica and self.clases_dinamicas:
                self.estado_en_vivo = "DINAMICA"
                self.buffer_secuencia_dinamica.append(vector)
                if centroide_mano:
                    self.trayectoria_centroides_mano.append(centroide_mano)

                progreso = min(1.0, len(self.buffer_secuencia_dinamica) / config.FRAMES_POR_SECUENCIA)
                self.barra_confianza["value"] = progreso * 100
                self.lbl_confianza_val.configure(text=f"Gesto: {int(progreso * 100)}%")
                self.lbl_estado_badge.configure(text="MOVIMIENTO DETECTADO", bg=COLOR_INFO, fg=COLOR_TEXT_LIGHT)

                if len(self.buffer_secuencia_dinamica) >= config.FRAMES_POR_SECUENCIA:
                    # Clasificar secuencia
                    seña, conf = self._clasificar_secuencia_dinamica(self.buffer_secuencia_dinamica)
                    if seña and not seña.startswith("_") and conf >= config.UMBRAL_CONFIANZA:
                        palabra = self.traductor.procesar_gesto_dinamico(seña, conf)
                        if palabra:
                            self._al_confirmar_seña(seña, palabra, conf, "dinamica")
                    self.buffer_secuencia_dinamica.clear()
                    self.trayectoria_centroides_mano.clear()

            # RAMA ESTÁTICA
            elif es_estatica and self.clases_estaticas:
                self.buffer_secuencia_dinamica.clear()
                self.trayectoria_centroides_mano.clear()

                seña, conf = self._clasificar_frame_estatico(vector)
                if seña and conf >= config.UMBRAL_CONFIANZA:
                    if seña != self.postura_en_observacion:
                        self.postura_en_observacion = seña
                        self.tiempo_inicio_postura = ahora

                    duracion = ahora - self.tiempo_inicio_postura
                    progreso = min(1.0, duracion / config.TIEMPO_CONFIRMACION_ESTATICA)
                    self.barra_confianza["value"] = progreso * 100
                    self.lbl_confianza_val.configure(text=f"{seña} ({int(conf * 100)}%)")
                    self.lbl_estado_badge.configure(text=f"MANTENIENDO '{seña}'...", bg=COLOR_ACCENT, fg=COLOR_TEXT_LIGHT)

                    if duracion >= config.TIEMPO_CONFIRMACION_ESTATICA:
                        palabra = self.traductor.procesar_seña_estatica(seña, conf)
                        if palabra:
                            self._al_confirmar_seña(seña, palabra, conf, "estatica")
                else:
                    self.postura_en_observacion = None
                    self.barra_confianza["value"] = 0
                    self.lbl_confianza_val.configure(text="Analizando postura...")
                    self.lbl_estado_badge.configure(text="LISTO PARA SIGNAR", bg=COLOR_INPUT_DARK, fg=COLOR_TEXT_LIGHT)

        # Renderizar en canvas
        frame_para_mostrar = self._generar_frame_segun_vista(frame_bgr, results)
        self._dibujar_frame_en_canvas(self.canvas_video, frame_para_mostrar)

    def _al_confirmar_seña(self, seña: str, palabra: str, confianza: float, tipo: str):
        """Dispara las acciones cuando una seña es aceptada por la máquina de estados."""
        self.ultima_seña_detectada = seña
        self.confianza_actual = confianza
        self.lbl_sena_grande.configure(text=seña)

        # Actualizar caja de texto con texto inteligente
        oracion = self.traductor.obtener_oracion_actual()
        self.txt_traduccion.delete("1.0", tk.END)
        self.txt_traduccion.insert(tk.END, oracion)

        # Guardar en registro temporal de la sesión
        self.predicciones_sesion.append({
            "sena": seña,
            "palabra": palabra,
            "confianza": round(confianza, 3),
            "tipo": tipo,
            "timestamp": time.time()
        })

        # Entrar en cooldown
        self.estado_en_vivo = "COOLDOWN"
        self.tiempo_inicio_cooldown = time.time()

        # Actualizar métricas
        cant_palabras = len(self.traductor.palabras_acumuladas)
        self.lbl_metricas.configure(text=f"Última: {seña} | Confianza: {int(confianza * 100)}% | Total Palabras: {cant_palabras}")

    # =========================================================================
    # CLASIFICACIÓN DE INFERENCIA
    # =========================================================================
    def _clasificar_frame_estatico(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.modelo_estatico or not self.clases_estaticas:
            return None, 0.0
        try:
            proba = self.modelo_estatico.predict_proba([vector])[0]
            idx = int(np.argmax(proba))
            return self.clases_estaticas[idx], float(proba[idx])
        except Exception:
            return None, 0.0

    def _clasificar_secuencia_dinamica(self, secuencia: List[np.ndarray]) -> Tuple[Optional[str], float]:
        if not self.modelo_dinamico or not self.clases_dinamicas:
            return None, 0.0
        try:
            arr_sec = np.array(secuencia, dtype=np.float32)
            if len(arr_sec) < config.FRAMES_POR_SECUENCIA:
                arr_sec = np.pad(arr_sec, ((0, config.FRAMES_POR_SECUENCIA - len(arr_sec)), (0, 0)), mode="edge")
            else:
                arr_sec = arr_sec[-config.FRAMES_POR_SECUENCIA:]
            features = extraer_caracteristicas_dinamicas(arr_sec)
            proba = self.modelo_dinamico.predict_proba([features])[0]
            idx = int(np.argmax(proba))
            return self.clases_dinamicas[idx], float(proba[idx])
        except Exception:
            return None, 0.0

    # =========================================================================
    # LÓGICA DE CAPTURA DE MUESTRAS
    # =========================================================================
    def _procesar_frame_captura(self, frame_bgr: np.ndarray, vector: np.ndarray, results: Any, calidad_ok: bool, calidad_msg: str):
        # Actualizar badge de calidad
        if calidad_ok:
            self.lbl_calidad_badge.configure(text="CALIDAD: ÓPTIMA ✅", bg="#064E3B", fg="#6EE7B7")
        else:
            self.lbl_calidad_badge.configure(text=f"CALIDAD: {calidad_msg.upper()}", bg="#7F1D1D", fg="#FCA5A5")

        # Dibujar landmarks en el frame
        frame_dibujado = self.extractor.dibujar_landmarks(frame_bgr, results)

        # Si hay grabación dinámica en curso
        if self.grabando_dinamica:
            self.frames_grabados_dinamicos.append(vector)
            total_grabados = len(self.frames_grabados_dinamicos)
            cv2.putText(
                frame_dibujado,
                f"GRABANDO: {total_grabados}/{config.FRAMES_POR_SECUENCIA}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            if total_grabados >= config.FRAMES_POR_SECUENCIA:
                self._finalizar_grabacion_dinamica()

        self._dibujar_frame_en_canvas(self.canvas_captura, frame_dibujado)

    def accion_capturar_muestra(self):
        """Toma una muestra estática o inicia la grabación dinámica."""
        if not self.camara_activa:
            messagebox.showwarning("Cámara Inactiva", "Inicie la cámara primero para capturar muestras.")
            return

        nombre_sena = self._obtener_nombre_sena_actual()
        if not nombre_sena:
            messagebox.showwarning("Nombre Requerido", "Ingrese o seleccione un nombre de seña válido.")
            return

        tipo = self.var_tipo_sena.get()
        conjunto = self.var_conjunto.get()

        # Asegurar registro en la base de datos
        sena_id = db.obtener_o_crear_sena(nombre_sena, tipo)

        with self.bloqueo_frame:
            frame = self.frame_bgr_reciente.copy() if self.frame_bgr_reciente is not None else None
            vector = self.vector_caracteristicas_reciente.copy() if self.vector_caracteristicas_reciente is not None else None
            calidad_ok, msg = self.calidad_muestra_reciente

        if frame is None or vector is None:
            messagebox.showwarning("Error", "No hay fotograma disponible de la cámara.")
            return

        if not calidad_ok:
            if not messagebox.askyesno("Advertencia de Calidad", f"{msg}.\n¿Desea guardarla de todas formas?"):
                return

        if tipo == "estatica":
            # Guardar imagen y .npy
            dir_guardar = config.DIR_SENAS / nombre_sena / conjunto
            dir_guardar.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)

            ruta_img = dir_guardar / f"img_{nombre_sena}_{timestamp}.jpg"
            ruta_npy = dir_guardar / f"lm_{nombre_sena}_{timestamp}.npy"

            cv2.imwrite(str(ruta_img), frame)
            np.save(str(ruta_npy), vector)

            db.guardar_muestra(sena_id, conjunto, str(ruta_img), str(ruta_npy))
            self.total_muestras_guardadas += 1
            self._actualizar_conteo_muestras_sena(sena_id)
            self.lbl_status_global.configure(text=f"Muestra estática guardada para '{nombre_sena}' en {conjunto}.")

        elif tipo == "dinamica":
            # Iniciar grabación dinámica
            self.frames_grabados_dinamicos.clear()
            self.grabando_dinamica = True
            self.btn_capturar.configure(text="GRABANDO GESTO... 🔴", bg=COLOR_DANGER)

    def _finalizar_grabacion_dinamica(self):
        """Guarda la secuencia de 30 frames en disco y SQLite."""
        self.grabando_dinamica = False
        self.btn_capturar.configure(text="📷 CAPTURAR MUESTRA", bg=COLOR_ACCENT)

        nombre_sena = self._obtener_nombre_sena_actual()
        conjunto = self.var_conjunto.get()
        sena_id = db.obtener_o_crear_sena(nombre_sena, "dinamica")

        dir_guardar = config.DIR_SECUENCIAS / nombre_sena / conjunto
        dir_guardar.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        ruta_npy = dir_guardar / f"seq_{nombre_sena}_{timestamp}.npy"

        arr_seq = np.array(self.frames_grabados_dinamicos[:config.FRAMES_POR_SECUENCIA], dtype=np.float32)
        np.save(str(ruta_npy), arr_seq)

        db.guardar_secuencia(sena_id, conjunto, str(ruta_npy), len(arr_seq))
        self.total_muestras_guardadas += 1
        self._actualizar_conteo_muestras_sena(sena_id)
        self.lbl_status_global.configure(text=f"Secuencia dinámica de 30 frames guardada para '{nombre_sena}'.")
        messagebox.showinfo("Gesto Guardado", f"¡Secuencia dinámica para '{nombre_sena}' guardada exitosamente!")

    def accion_borrar_ultima_muestra(self):
        """Elimina la última muestra agregada de la base de datos y disco."""
        nombre_sena = self._obtener_nombre_sena_actual()
        if not nombre_sena:
            return

        sena_info = db.obtener_sena_por_nombre(nombre_sena)
        if not sena_info:
            return

        tipo = sena_info["tipo"]
        eliminado = False

        if tipo == "estatica":
            # Buscar última muestra en SQLite
            with db.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, ruta_imagen, ruta_landmarks FROM muestras WHERE sena_id = ? ORDER BY id DESC LIMIT 1", (sena_info["id"],))
                fila = cursor.fetchone()
                if fila:
                    try:
                        Path(fila["ruta_imagen"]).unlink(missing_ok=True)
                        Path(fila["ruta_landmarks"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                    cursor.execute("DELETE FROM muestras WHERE id = ?", (fila["id"],))
                    conn.commit()
                    eliminado = True
        else:
            with db.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, ruta_npy FROM secuencias WHERE sena_id = ? ORDER BY id DESC LIMIT 1", (sena_info["id"],))
                fila = cursor.fetchone()
                if fila:
                    try:
                        Path(fila["ruta_npy"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                    cursor.execute("DELETE FROM secuencias WHERE id = ?", (fila["id"],))
                    conn.commit()
                    eliminado = True

        if eliminado:
            self._actualizar_conteo_muestras_sena(sena_info["id"])
            messagebox.showinfo("Muestra Eliminada", "Se eliminó la última muestra correctamente.")
        else:
            messagebox.showinfo("Sin Muestras", "No hay muestras previas para eliminar de esta seña.")

    def _obtener_nombre_sena_actual(self) -> str:
        if self.var_modo_seleccion.get() == "NUEVA":
            return self.ent_nombre_sena.get().strip().upper()
        else:
            return self.combo_senas_existentes.get().strip().upper()

    def _actualizar_estado_formulario_captura(self):
        modo = self.var_modo_seleccion.get()
        if modo == "NUEVA":
            self.ent_nombre_sena.configure(state="normal")
            self.combo_senas_existentes.configure(state="disabled")
        else:
            self.ent_nombre_sena.configure(state="disabled")
            self.combo_senas_existentes.configure(state="readonly")

    def _refrescar_combo_senas_existentes(self):
        senas = db.obtener_todas_las_senas()
        nombres = [s["nombre"] for s in senas]
        self.combo_senas_existentes["values"] = nombres
        if nombres:
            self.combo_senas_existentes.set(nombres[0])
            self._al_seleccionar_sena_existente(None)

    def _al_seleccionar_sena_existente(self, _event):
        nombre = self.combo_senas_existentes.get()
        if nombre:
            sena = db.obtener_sena_por_nombre(nombre)
            if sena:
                self.var_tipo_sena.set(sena["tipo"])
                self._actualizar_conteo_muestras_sena(sena["id"])

    def _actualizar_conteo_muestras_sena(self, sena_id: int):
        conteo = db.obtener_conteo_muestras_por_sena(sena_id)
        self.lbl_conteo_muestras.configure(
            text=f"Muestras: Train: {conteo['entrenamiento']} | Val: {conteo['validacion']} | Total: {conteo['total']}"
        )

    # =========================================================================
    # ACCIONES DE ENTRENAMIENTO EN SEGUNDO PLANO
    # =========================================================================
    def accion_iniciar_entrenamiento(self):
        self.btn_iniciar_entrenamiento.configure(state="disabled", text="ENTRENANDO... ESPERE POR FAVOR")
        self.progreso_entrenamiento.start(10)
        self.txt_log_entrenamiento.delete("1.0", tk.END)

        def worker():
            def log_callback(msg: str):
                self.root.after(0, lambda: self._agregar_log_entrenamiento(msg))

            resultado = ejecutar_entrenamiento(callback_progreso=log_callback)
            self.root.after(0, lambda: self._finalizar_entrenamiento(resultado))

        threading.Thread(target=worker, daemon=True).start()

    def _agregar_log_entrenamiento(self, msg: str):
        self.txt_log_entrenamiento.insert(tk.END, msg + "\n")
        self.txt_log_entrenamiento.see(tk.END)

    def _finalizar_entrenamiento(self, resultado: Dict[str, Any]):
        self.progreso_entrenamiento.stop()
        self.btn_iniciar_entrenamiento.configure(state="normal", text="🚀 INICIAR ENTRENAMIENTO DEL MODELO")

        if resultado.get("exito"):
            acc = resultado.get("acc_promedio", 1.0) * 100
            messagebox.showinfo(
                "Entrenamiento Exitoso",
                f"¡Modelo reentrenado con éxito!\nPrecisión estimada: {acc:.2f}%\n"
                f"Clases Estáticas: {len(resultado.get('clases_estaticas', []))}\n"
                f"Clases Dinámicas: {len(resultado.get('clases_dinamicas', []))}"
            )
            # Recargar modelos en memoria inmediatamente
            self._cargar_modelos()
            self._refrescar_tabla_entrenamiento()
        else:
            err = resultado.get("error", "Error desconocido")
            messagebox.showerror("Error en Entrenamiento", f"Ocurrió un error al entrenar:\n{err}")

    def _refrescar_tabla_entrenamiento(self):
        for item in self.tree_catalogo.get_children():
            self.tree_catalogo.delete(item)

        stats = db.obtener_estadisticas_generales()
        for s in stats["detalle_senas"]:
            total = s["estaticas"] if s["tipo"] == "estatica" else s["dinamicas"]
            estado = "✅ Lista" if total >= config.MIN_MUESTRAS_POR_CLASE else f"⚠️ ({total}/{config.MIN_MUESTRAS_POR_CLASE})"
            self.tree_catalogo.insert("", tk.END, values=(s["nombre"], s["tipo"], s["estaticas"], s["dinamicas"], estado))

    # =========================================================================
    # ACCIONES DE HISTORIAL Y EXPORTACIÓN
    # =========================================================================
    def _refrescar_historial(self):
        for item in self.tree_historial.get_children():
            self.tree_historial.delete(item)

        traducciones = db.obtener_todas_las_traducciones(limite=500)
        for t in traducciones:
            self.tree_historial.insert("", tk.END, values=(t["id"], t["fecha"], t["texto"]))

    def accion_exportar_csv(self):
        ruta = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv")],
            title="Guardar Historial de Traducciones como CSV"
        )
        if ruta:
            cant = db.exportar_traducciones_csv(ruta)
            messagebox.showinfo("Exportación Exitosa", f"Se exportaron {cant} traducciones al archivo:\n{ruta}")

    def accion_exportar_txt(self):
        ruta = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de Texto", "*.txt")],
            title="Guardar Reporte de Traducción como TXT"
        )
        if ruta:
            cant = db.exportar_traducciones_txt(ruta)
            messagebox.showinfo("Reporte Exportado", f"Se exportaron {cant} registros de traducción al archivo:\n{ruta}")

    def accion_eliminar_traduccion_seleccionada(self):
        seleccion = self.tree_historial.selection()
        if not seleccion:
            messagebox.showwarning("Selección Requerida", "Seleccione un registro de la tabla para eliminar.")
            return

        item = self.tree_historial.item(seleccion[0])
        traduccion_id = item["values"][0]

        if messagebox.askyesno("Confirmar Eliminación", f"¿Eliminar la traducción ID #{traduccion_id}?"):
            db.eliminar_traduccion(int(traduccion_id))
            self._refrescar_historial()

    def accion_vaciar_historial(self):
        if messagebox.askyesno("Confirmar Vaciado", "¿Está seguro de eliminar TODO el historial de traducciones? Esta acción no se puede deshacer."):
            db.limpiar_todas_las_traducciones()
            self._refrescar_historial()
            messagebox.showinfo("Historial Vaciado", "El historial ha sido vaciado por completo.")

    # =========================================================================
    # DICCIONARIO VISUAL (TEXTO A SEÑAS)
    # =========================================================================
    def _refrescar_lista_diccionario(self):
        self.listbox_dic.delete(0, tk.END)
        senas = db.obtener_todas_las_senas()
        for s in senas:
            self.listbox_dic.insert(tk.END, f"{s['nombre']} ({s['tipo'].capitalize()})")

    def _filtrar_diccionario(self, _event):
        filtro = self.ent_buscar_dic.get().strip().upper()
        self.listbox_dic.delete(0, tk.END)
        senas = db.obtener_todas_las_senas()
        for s in senas:
            if filtro in s["nombre"].upper():
                self.listbox_dic.insert(tk.END, f"{s['nombre']} ({s['tipo'].capitalize()})")

    def _al_seleccionar_seña_diccionario(self, _event):
        seleccion = self.listbox_dic.curselection()
        if not seleccion:
            return

        texto = self.listbox_dic.get(seleccion[0])
        nombre = texto.split(" ")[0]

        sena = db.obtener_sena_por_nombre(nombre)
        if not sena:
            return

        self.lbl_dic_titulo.configure(text=f"SEÑA: {sena['nombre']}")
        self.lbl_dic_tipo.configure(text=f"Tipo: {sena['tipo'].upper()} | Registrada: {sena['fecha']}")

        # Buscar si existe foto de referencia en datos/senas
        dir_sena = config.DIR_SENAS / sena["nombre"] / "entrenamiento"
        foto_encontrada = None
        if dir_sena.exists():
            fotos = list(dir_sena.glob("*.jpg"))
            if fotos:
                foto_encontrada = fotos[0]

        if foto_encontrada and foto_encontrada.exists():
            try:
                img_bgr = cv2.imread(str(foto_encontrada))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb).resize((420, 360), Image.Resampling.LANCZOS)
                self.img_dic_ref = ImageTk.PhotoImage(img_pil)
                self.canvas_dic_foto.delete("all")
                self.canvas_dic_foto.create_image(210, 180, image=self.img_dic_ref)
                self.lbl_dic_descripcion.configure(
                    text=f"Muestra fotográfica de referencia para la postura '{sena['nombre']}'. Mantenga la mano en esta posición frente a la cámara."
                )
                return
            except Exception:
                pass

        # Si no hay imagen o es dinámica
        self.canvas_dic_foto.delete("all")
        self.canvas_dic_foto.create_text(
            210, 180,
            text=f"Gesto {sena['tipo'].capitalize()}\n'{sena['nombre']}'\n\n(Consulte o grabe muestras en Captura)",
            fill=COLOR_TEXT_MUTED,
            font=FONT_BOLD,
            justify="center"
        )
        self.lbl_dic_descripcion.configure(
            text=f"Seña registrada como '{sena['tipo']}'. En este modo, el traductor evalúa el movimiento temporal en 30 frames."
        )

    # =========================================================================
    # ACCIONES DE TRADUCCIÓN (TEXTO Y VOZ)
    # =========================================================================
    def accion_hablar_oracion(self):
        texto = self.txt_traduccion.get("1.0", tk.END).strip()
        if texto:
            self.traductor.motor_voz.hablar(texto)

    def accion_guardar_oracion(self):
        texto = self.txt_traduccion.get("1.0", tk.END).strip()
        if not texto:
            messagebox.showwarning("Texto Vacío", "No hay ninguna oración traducida para guardar.")
            return

        db.guardar_traduccion(texto, self.predicciones_sesion)
        self._refrescar_historial()
        messagebox.showinfo("Guardado Exitoso", "La oración ha sido guardada en la base de datos.")
        self.predicciones_sesion.clear()

    def accion_borrar_ultima_palabra(self):
        if self.traductor.borrar_ultimo():
            oracion = self.traductor.obtener_oracion_actual()
            self.txt_traduccion.delete("1.0", tk.END)
            self.txt_traduccion.insert(tk.END, oracion)

    def accion_limpiar_oracion(self):
        self.traductor.limpiar()
        self.txt_traduccion.delete("1.0", tk.END)
        self.lbl_sena_grande.configure(text="---")
        self.predicciones_sesion.clear()

    # =========================================================================
    # CONFIGURACIÓN Y GUARDADO DE PARÁMETROS
    # =========================================================================
    def accion_guardar_configuracion(self):
        try:
            config.CAMARA_INDEX = int(self.spn_camara.get())
            config.UMBRAL_CONFIANZA = float(self.scale_confianza.get())
            config.TIEMPO_CONFIRMACION_ESTATICA = float(self.scale_estatica.get())
            config.TIEMPO_COOLDOWN_POST_DETECCION = float(self.scale_cooldown.get())
            config.USAR_FILTRO_CLAHE = self.var_clahe.get()
            config.VOZ_HABILITADA = self.var_voz.get()

            self.extractor.usar_clahe = config.USAR_FILTRO_CLAHE
            self.traductor.motor_voz.set_habilitado(config.VOZ_HABILITADA)

            # Persistir en base de datos SQLite
            db.guardar_configuracion("CAMARA_INDEX", str(config.CAMARA_INDEX))
            db.guardar_configuracion("UMBRAL_CONFIANZA", str(config.UMBRAL_CONFIANZA))
            db.guardar_configuracion("USAR_FILTRO_CLAHE", str(config.USAR_FILTRO_CLAHE))
            db.guardar_configuracion("VOZ_HABILITADA", str(config.VOZ_HABILITADA))

            messagebox.showinfo("Configuración", "¡Ajustes guardados y aplicados correctamente!")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la configuración: {e}")

    # =========================================================================
    # RENDERIZADO Y MODOS DE VISTA
    # =========================================================================
    def _al_cambiar_vista(self, _event):
        self.modo_visualizacion = self.combo_vista.get()

    def _al_cambiar_modo(self, _event):
        self.modo_reconocimiento = self.combo_modo.get()

    def _generar_frame_segun_vista(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        vista = self.modo_visualizacion
        if vista == "NORMAL":
            return frame_bgr
        elif vista == "LANDMARKS":
            return self.extractor.dibujar_landmarks(frame_bgr, results)
        elif vista == "CANNY":
            return generar_panel_derecho(frame_bgr, results, self.extractor, usar_canny=True)
        elif vista == "ESQUELETO":
            return generar_panel_derecho(frame_bgr, results, self.extractor, usar_canny=False)
        return frame_bgr

    def _dibujar_frame_en_canvas(self, canvas: tk.Canvas, frame_bgr: np.ndarray):
        """Redimensiona y plasma el frame OpenCV directamente en el Canvas de Tkinter sin parpadeo."""
        alto_canvas = canvas.winfo_height()
        ancho_canvas = canvas.winfo_width()

        if alto_canvas < 50 or ancho_canvas < 50:
            ancho_canvas, alto_canvas = 640, 480

        # Mantener proporción de aspecto
        h_orig, w_orig = frame_bgr.shape[:2]
        escala = min(ancho_canvas / w_orig, alto_canvas / h_orig)
        nuevo_w = max(1, int(w_orig * escala))
        nuevo_h = max(1, int(h_orig * escala))

        frame_resized = cv2.resize(frame_bgr, (nuevo_w, nuevo_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        pos_x = ancho_canvas // 2
        pos_y = alto_canvas // 2

        # Reutilizar el elemento gráfico del canvas para máxima fluidez y evitar parpadeos
        if not hasattr(canvas, "_canvas_img_id") or canvas._canvas_img_id is None:
            canvas.delete("all")
            canvas._canvas_img_id = canvas.create_image(pos_x, pos_y, image=img_tk)
        else:
            canvas.coords(canvas._canvas_img_id, pos_x, pos_y)
            canvas.itemconfig(canvas._canvas_img_id, image=img_tk)

        # Prevenir recolección de basura de PhotoImage
        canvas.image = img_tk

    # =========================================================================
    # CIERRE SEGURO DE LA APLICACIÓN
    # =========================================================================
    def cerrar_aplicacion(self):
        self.camara_activa = False
        if self.cap:
            self.cap.release()
        self.extractor.cerrar()
        self.root.destroy()


def iniciar_interfaz_grafica():
    """Punto de entrada de la GUI."""
    root = tk.Tk()
    app = InterfazTraductorApp(root)
    root.mainloop()


if __name__ == "__main__":
    iniciar_interfaz_grafica()

