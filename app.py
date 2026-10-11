import sys
import os
import json
import time
import shutil
import logging
import math
import csv
import gc
import subprocess
import threading
import inspect
import platform
import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque
from logging.handlers import RotatingFileHandler
from enum import Enum

import numpy as np
from scipy.io import wavfile

# Qt Imports
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QUrl, QTimer, QPoint, QSize, QLineF,
    QStandardPaths, QEvent, QByteArray
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSlider, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter,
    QDialog, QFormLayout, QMenu, QStyle, QInputDialog, QPlainTextEdit,
    QSizePolicy, QFrame, QToolButton, QStatusBar, QListWidget, QListWidgetItem,
    QDockWidget, QScrollArea
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import (
    QAction, QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QPen,
    QMouseEvent, QKeySequence, QShortcut, QDesktopServices
)

# ML Imports
try:
    import torch
    from diffusers import AudioLDM2Pipeline
    import transformers
    import diffusers
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# --- Configuration & Constants ---
APP_NAME = "AudioLDM2 Studio"
APP_VERSION = "3.0.5"
SETTINGS_ORG = "AudioLDM2"
SETTINGS_APP = "Studio"

# Set default cache to E:\cacheAI as requested
DEFAULT_CACHE_DIR = "E:\\cacheAI"

# Keep state and logs in the Current Working Directory (CWD)
CWD = os.getcwd()
BATCH_STATE_FILE = os.path.join(CWD, "batch_state.json")
LOG_FILE = os.path.join(CWD, f"{APP_NAME}.log")

DEFAULT_MODELS = [
    "cvssp/audioldm2",
    "cvssp/audioldm2-music",
    "cvssp/audioldm2-gigaspeech",
    "cvssp/audioldm2-ljspeech",
]

DEFAULT_PROMPT_PRESETS = [
    "Musical constellations twinkling in the night sky",
    "Rain drops falling on a tin roof with distant thunder",
    "A cinematic orchestral swell with epic drums",
    "Birds chirping in a peaceful forest at dawn",
    "Electronic synth wave with retro 80s vibe",
    "Ocean waves crashing on a rocky shore",
    "A jazz saxophone playing in a smoky bar",
    "Wind howling through mountain peaks",
]

NEGATIVE_PROMPT_PRESETS = [
    "noise, distortion, artifacts, low quality",
    "static, hum, buzzing, interference",
    "echo, reverb, muffled sound",
    "clipping, peaking, distortion",
]

# App Constants
OVERLAP_S = 1.0
MIN_CHUNK_S = 2.0
MAX_HISTORY = 500
MAX_PROMPT_HISTORY = 50
FADE_DURATION_S = 0.05
TRIM_THRESHOLD = 0.01
ETA_SMOOTHING_ALPHA = 0.3
AUTO_SAVE_INTERVAL_MS = 60_000
GPU_MEM_POLL_MS = 2_000
NOTIFY_INTERVAL_MS = 50

# --- Professional Stylesheet ---
STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }

QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #cccccc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #2b2b2b;
}

QTabWidget::pane {
    border: 1px solid #3c3c3c;
    border-radius: 2px;
    background-color: #2b2b2b;
    top: -1px;
}
QTabBar::tab {
    background: #3c3c3c;
    color: #aaaaaa;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    min-width: 70px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #2A82DA;
    color: white;
}
QTabBar::tab:hover:!selected {
    background: #4a4a4a;
    color: white;
}

QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    color: #e0e0e0;
    padding: 6px 12px;
    border-radius: 3px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #2A82DA;
}
QPushButton:pressed { background-color: #2b2b2b; }
QPushButton:disabled {
    background-color: #333333;
    color: #666666;
    border-color: #333333;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background-color: #333333;
    border: 1px solid #3c3c3c;
    padding: 4px 6px;
    border-radius: 2px;
    color: #e0e0e0;
    selection-background-color: #2A82DA;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #2A82DA;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #333333;
    border: 1px solid #3c3c3c;
    selection-background-color: #2A82DA;
    color: #e0e0e0;
}

QTableWidget {
    gridline-color: #3c3c3c;
    background-color: #2b2b2b;
    alternate-background-color: #333333;
    border: 1px solid #3c3c3c;
    border-radius: 2px;
}
QHeaderView::section {
    background-color: #3c3c3c;
    padding: 6px;
    border: none;
    font-weight: bold;
    color: #cccccc;
}

QProgressBar {
    border: 1px solid #3c3c3c;
    border-radius: 2px;
    text-align: center;
    background-color: #333333;
    color: white;
    min-height: 22px;
}
QProgressBar::chunk {
    background-color: #2A82DA;
    border-radius: 1px;
}

QScrollBar:vertical {
    background: #2b2b2b;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4a4a4a;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #2A82DA; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #2b2b2b;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #4a4a4a;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover { background: #2A82DA; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QScrollArea { border: none; background-color: transparent; }

QCheckBox { color: #e0e0e0; spacing: 6px; }
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 2px;
    border: 1px solid #4a4a4a;
    background: #333333;
}
QCheckBox::indicator:checked {
    background: #2A82DA;
    border-color: #2A82DA;
}

QLabel { color: #d0d0d0; }

QToolButton {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 4px;
    color: #e0e0e0;
}
QToolButton:hover {
    background-color: #4a4a4a;
    border-color: #2A82DA;
}

QStatusBar {
    background-color: #2b2b2b;
    color: #aaaaaa;
}
QStatusBar::item { border: none; }

QMenuBar {
    background-color: #2b2b2b;
    color: #d0d0d0;
    border-bottom: 1px solid #3c3c3c;
}
QMenuBar::item:selected { background-color: #3c3c3c; }
QMenu {
    background-color: #333333;
    border: 1px solid #3c3c3c;
    color: #d0d0d0;
}
QMenu::item:selected { background-color: #2A82DA; }

QSlider::groove:horizontal {
    border: 1px solid #3c3c3c;
    height: 4px;
    background: #333333;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #2A82DA;
    border: 1px solid #2A82DA;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #3a92ea; }

QSplitter::handle { background-color: #3c3c3c; }
QSplitter::handle:horizontal { width: 2px; }
"""

# --- JSON Settings Engine ---
class JsonSettings:
    """Mimics QSettings API but saves directly to a JSON file."""
    def __init__(self, path):
        self.path = path
        self.data = {}
        self._dirty = False
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load JSON settings: {e}")
                self.data = {}

    def save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
            self._dirty = False
        except Exception as e:
            logging.error(f"Failed to save JSON settings: {e}")

    def value(self, key, default=None, type=None):
        val = self.data.get(key, default)
        if val is None:
            return default
        
        # Handle QByteArray (for window geometry/state)
        if isinstance(val, str) and val.startswith("base64:"):
            return QByteArray.fromBase64(val[7:].encode('utf-8'))
            
        if type == bool:
            if isinstance(val, str): return val.lower() in ('true', '1', 'yes')
            return bool(val)
        if type == int:
            try: return int(val)
            except: return default
        if type == float:
            try: return float(val)
            except: return default
        return val

    def setValue(self, key, value):
        # Encode QByteArray to base64 string so it can be stored in JSON
        if isinstance(value, QByteArray):
            value = "base64:" + base64.b64encode(value.data()).decode('utf-8')
        # Only mark dirty and update if value changed
        if self.data.get(key) != value:
            self.data[key] = value
            self._dirty = True

    def remove(self, key):
        if key in self.data:
            del self.data[key]
            self._dirty = True
            self.save()

    def sync(self):
        if self._dirty:
            self.save()

    def clear(self):
        self.data = {}
        self._dirty = True
        self.save()


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(APP_NAME)

try:
    # Enforce mode='w' to ALWAYS overwrite the log file on startup
    handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
except Exception:
    handler = logging.FileHandler("audioldm2_studio.log", mode='w', encoding='utf-8')

handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logger.addHandler(handler)

def log_system_info():
    logger.info("=" * 60)
    logger.info(f"{APP_NAME} v{APP_VERSION} starting")
    logger.info(f"Python: {sys.version}")
    logger.info(f"OS: {platform.platform()}")
    if ML_AVAILABLE:
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Diffusers: {diffusers.__version__}")
        logger.info(f"Transformers: {transformers.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 60)


# --- Utility Functions ---
def sanitize_filename(text: str, max_length: int = 30) -> str:
    safe = "".join(c for c in text if c.isalnum() or c in (' ', '_', '-')).rstrip()
    safe = safe[:max_length].replace(' ', '_') if safe else "audio"
    return safe

def format_time(seconds) -> str:
    try:
        total_secs = int(float(seconds))
        if total_secs < 0:
            total_secs = 0
        minutes, secs = divmod(total_secs, 60)
        return f"{minutes:02d}:{secs:02d}"
    except Exception:
        return "00:00"

def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    if len(audio) == 0:
        return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_peak = 10 ** (target_db / 20.0)
    return audio * (target_peak / peak)

def apply_fade(audio: np.ndarray, sample_rate: int, fade_in_s: float = 0.05, fade_out_s: float = 0.05) -> np.ndarray:
    audio = audio.astype(np.float32, copy=True)
    if len(audio) == 0:
        return audio
    fade_in_samples = int(fade_in_s * sample_rate)
    fade_out_samples = int(fade_out_s * sample_rate)
    
    if fade_in_samples + fade_out_samples > len(audio):
        total_fade = len(audio)
        fade_in_samples = total_fade // 2
        fade_out_samples = total_fade - fade_in_samples
    
    if fade_in_samples > 0:
        audio[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    if fade_out_samples > 0:
        audio[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
    return audio

def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if len(audio) == 0:
        return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    above_threshold = np.where(np.abs(audio) > threshold)[0]
    if len(above_threshold) == 0:
        return audio
    start = above_threshold[0]
    end = above_threshold[-1] + 1
    return audio[start:end]


# --- State Machine & Data Models ---
class BatchState(Enum):
    IDLE = 0
    RUNNING = 1
    CANCELLED = 2
    COMPLETED = 3

@dataclass
class GenerationParams:
    prompt: str
    negative_prompt: str
    model_name: str
    duration: float
    steps: int
    guidance: float
    seed: int
    device: str
    cache_dir: str
    output_dir: str
    num_variations: int = 1
    normalize: bool = True
    fade_in: bool = True
    fade_out: bool = True
    trim: bool = True
    chunk_size: float = 10.0
    use_cpu_offload: bool = False
    section: str = ""


# --- Worker Threads ---
class AudioGenerationWorker(QObject):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    eta_updated = Signal(str)
    generation_completed = Signal(str, dict)
    error_occurred = Signal(str)
    cancelled = Signal()

    _shared_pipe = None
    _shared_pipe_model = None
    _shared_pipe_device = None
    _shared_pipe_offload = None
    _pipe_lock = threading.Lock()

    def __init__(self, params: GenerationParams) -> None:
        super().__init__()
        self.params = params
        self._cancel_event = threading.Event()
        self.pipe = None
        self.ema_step_time = None
        self.last_step_time = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @is_cancelled.setter
    def is_cancelled(self, val: bool):
        if val:
            self._cancel_event.set()
        else:
            self._cancel_event.clear()

    def _load_pipeline(self, actual_device: str):
        with self._pipe_lock:
            # 1. Check if we can reuse the pipeline
            if (AudioGenerationWorker._shared_pipe is not None
                and AudioGenerationWorker._shared_pipe_model == self.params.model_name
                and AudioGenerationWorker._shared_pipe_device == actual_device
                and AudioGenerationWorker._shared_pipe_offload == self.params.use_cpu_offload
                and not self.is_cancelled):
                self.pipe = AudioGenerationWorker._shared_pipe
                logger.info("Reusing cached pipeline in memory.")
                return

            # 2. Unload any existing pipeline to free VRAM
            self._unload_pipeline_unsafe()
            
            self.status_updated.emit("Loading model weights...")
            self.progress_updated.emit(5)

            dtype = torch.float16 if actual_device != "cpu" else torch.float32
            
            load_kwargs = {
                "torch_dtype": dtype,
                "cache_dir": self.params.cache_dir,
            }
            
            # Crucial: Request fp16 variant directly to avoid massive System RAM spike
            if dtype == torch.float16:
                load_kwargs["variant"] = "fp16"

            load_attempts = [
                {"use_safetensors": True},
                {"use_safetensors": False},
            ]
            
            last_err = None
            pipe = None
            
            for attempt in load_attempts:
                try:
                    pipe = AudioLDM2Pipeline.from_pretrained(
                        self.params.model_name,
                        **load_kwargs,
                        **attempt
                    )
                    break
                except Exception as e:
                    # If it failed because the fp16 variant doesn't exist, remove variant and retry
                    if "variant" in str(e).lower() or "fp16" in str(e).lower():
                        logger.warning("fp16 variant not found, falling back to default weights.")
                        load_kwargs.pop("variant", None)
                        try:
                            pipe = AudioLDM2Pipeline.from_pretrained(
                                self.params.model_name,
                                **load_kwargs,
                                **attempt
                            )
                            break
                        except Exception as e2:
                            last_err = e2
                            logger.warning(f"Load attempt failed ({attempt} no variant): {e2}")
                    else:
                        last_err = e
                        logger.warning(f"Load attempt failed ({attempt}): {e}")
                    
                    # Clean up any partial memory before next attempt
                    pipe = None
                    gc.collect()
                    if ML_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if pipe is None:
                raise RuntimeError(f"Failed to load model after multiple attempts. Last error: {last_err}")

            # Check if user cancelled DURING the long load/download
            if self.is_cancelled:
                del pipe
                gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise InterruptedError("User cancelled generation")

            self.status_updated.emit("Configuring model for inference...")
            self.progress_updated.emit(10)

            try:
                if self.params.use_cpu_offload and actual_device != "cpu":
                    if not hasattr(pipe, 'enable_model_cpu_offload'):
                        raise RuntimeError("CPU Offload is requested but the 'accelerate' library is not installed. Please run: pip install accelerate")
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(actual_device)

                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
            except RuntimeError as e:
                # If it fails during configuration (e.g. OOM moving to VRAM), clean up
                del pipe
                gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if "out of memory" in str(e).lower():
                    raise RuntimeError("GPU Out of Memory during model configuration.")
                raise

            AudioGenerationWorker._shared_pipe = pipe
            AudioGenerationWorker._shared_pipe_model = self.params.model_name
            AudioGenerationWorker._shared_pipe_device = actual_device
            AudioGenerationWorker._shared_pipe_offload = self.params.use_cpu_offload
            self.pipe = pipe

    def run(self):
        gen_start_time = time.time()
        try:
            p = self.params
            logger.info(f"Starting generation. Prompt: '{p.prompt}' | Model: {p.model_name}")
            
            cache_dir = os.path.normpath(p.cache_dir or DEFAULT_CACHE_DIR)
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as e:
                # Directly raise error if cache directory cannot be accessed (no fallback)
                raise RuntimeError(f"Cannot create or access cache directory '{cache_dir}':\n{e}")
            
            final_out_dir = p.output_dir
            if p.section:
                safe_section = sanitize_filename(p.section, max_length=50)
                final_out_dir = os.path.join(p.output_dir, safe_section)
            os.makedirs(final_out_dir, exist_ok=True)

            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["DIFFUSERS_CACHE"] = cache_dir

            actual_device = p.device
            if actual_device == "auto":
                if torch.cuda.is_available():
                    actual_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    actual_device = "mps"
                else:
                    actual_device = "cpu"
            
            logger.info(f"Using device: {actual_device} | CPU Offload: {p.use_cpu_offload}")

            self._load_pipeline(actual_device)

            self.status_updated.emit("Generating audio...")
            self.progress_updated.emit(15)

            saved_paths = []
            metadata_list = []
            
            MAX_CHUNK_S = max(MIN_CHUNK_S, p.chunk_size)
            use_chunking = p.duration > MAX_CHUNK_S
            
            for variation in range(p.num_variations):
                if self.is_cancelled:
                    raise InterruptedError("User cancelled generation")

                if p.seed >= 0:
                    base_seed = p.seed + variation
                else:
                    base_seed = int(time.time() * 1000) % (2**31) + variation
                
                logger.info(f"Variation {variation + 1}/{p.num_variations} - Base seed: {base_seed}")
                
                variation_start = time.time()

                if use_chunking:
                    num_chunks = math.ceil((p.duration - OVERLAP_S) / (MAX_CHUNK_S - OVERLAP_S))
                    self.status_updated.emit(f"Variation {variation + 1}/{p.num_variations} - Generating {num_chunks} chunks...")
                else:
                    num_chunks = 1

                final_audio = np.array([], dtype=np.float32)
                sample_rate = getattr(self.pipe.vae.config, 'sample_rate', 16000)
                self.ema_step_time = None
                self.last_step_time = None

                def progress_callback(step, timestep, latents, current_chunk=0):
                    if self.is_cancelled:
                        raise InterruptedError("User cancelled generation")
                    
                    step = int(step)
                    current_chunk = int(current_chunk)
                    num_chunks_int = int(num_chunks)
                    p_steps_int = int(p.steps)
                    variation_int = int(variation)
                    p_num_variations_int = int(p.num_variations)
                    
                    current_time = time.time()
                    base_progress = (step + 1) / p_steps_int
                    chunk_progress = (current_chunk + base_progress) / num_chunks_int
                    total_progress = 15 + int(75 * chunk_progress * (1 / max(p_num_variations_int, 1)))
                    total_progress += int((variation_int / max(p_num_variations_int, 1)) * 75)
                    self.progress_updated.emit(min(total_progress, 95))
                    
                    if self.last_step_time is not None:
                        dt = current_time - self.last_step_time
                        if dt > 0:
                            if self.ema_step_time is None:
                                self.ema_step_time = dt
                            else:
                                self.ema_step_time = ETA_SMOOTHING_ALPHA * dt + (1 - ETA_SMOOTHING_ALPHA) * self.ema_step_time
                            
                            remaining_steps = max(0, p_steps_int - step - 1)
                            remaining_chunks = max(0, num_chunks_int - current_chunk - 1)
                            remaining_variations = max(0, p_num_variations_int - variation_int - 1)
                            
                            eta_seconds = (
                                remaining_steps * self.ema_step_time
                                + remaining_chunks * p_steps_int * self.ema_step_time
                                + remaining_variations * num_chunks_int * p_steps_int * self.ema_step_time
                            )
                            self.eta_updated.emit(format_time(eta_seconds))
                    
                    self.last_step_time = current_time
                    self.status_updated.emit(
                        f"Variation {variation_int + 1}/{p_num_variations_int} - "
                        f"Chunk {current_chunk + 1}/{num_chunks_int} - "
                        f"Step {step + 1}/{p_steps_int}"
                    )

                for chunk_idx in range(num_chunks):
                    if self.is_cancelled:
                        raise InterruptedError("User cancelled generation")

                    if use_chunking:
                        chunk_len = MAX_CHUNK_S if chunk_idx < num_chunks - 1 else max(
                            1.0, p.duration - (num_chunks - 1) * (MAX_CHUNK_S - OVERLAP_S)
                        )
                    else:
                        chunk_len = p.duration
                    
                    chunk_seed = (base_seed + chunk_idx * 7919) % (2**31)
                    
                    gen_device = "cpu" if p.use_cpu_offload else actual_device
                    generator = torch.Generator(device=gen_device).manual_seed(chunk_seed)
                    
                    def cb_old(step, timestep, latents, c=chunk_idx):
                        progress_callback(step, timestep, latents, current_chunk=c)
                        
                    def cb_new(pipe, step, timestep, callback_kwargs, c=chunk_idx):
                        latents = callback_kwargs.get("latents") if isinstance(callback_kwargs, dict) else None
                        progress_callback(step, timestep, latents, current_chunk=c)

                    kwargs = dict(
                        prompt=p.prompt,
                        negative_prompt=p.negative_prompt if p.negative_prompt else None,
                        num_inference_steps=p.steps,
                        audio_length_in_s=chunk_len,
                        guidance_scale=p.guidance,
                        generator=generator,
                    )
                    
                    sig = inspect.signature(self.pipe.__call__)
                    if "callback_on_step_end" in sig.parameters:
                        kwargs["callback_on_step_end"] = cb_new
                        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                    else:
                        kwargs["callback"] = cb_old
                        kwargs["callback_steps"] = 1

                    with torch.inference_mode():
                        audio_chunk = self.pipe(**kwargs).audios[0]

                    if num_chunks > 1:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if num_chunks == 1:
                        final_audio = audio_chunk
                    else:
                        if len(final_audio) == 0:
                            final_audio = audio_chunk
                        else:
                            overlap_samples = min(int(OVERLAP_S * sample_rate), len(final_audio) // 2, len(audio_chunk) // 2)
                            if overlap_samples <= 0:
                                final_audio = np.concatenate([final_audio, audio_chunk])
                            else:
                                fade_out_arr = np.sqrt(np.linspace(1, 0, overlap_samples))
                                fade_in_arr = np.sqrt(np.linspace(0, 1, overlap_samples))
                                mixed_overlap = final_audio[-overlap_samples:] * fade_out_arr + audio_chunk[:overlap_samples] * fade_in_arr
                                final_audio = np.concatenate([
                                    final_audio[:-overlap_samples],
                                    mixed_overlap,
                                    audio_chunk[overlap_samples:]
                                ])

                if self.is_cancelled:
                    raise InterruptedError("User cancelled generation")

                audio = final_audio
                if p.trim:
                    audio = trim_silence(audio)
                if p.normalize:
                    audio = normalize_audio(audio)
                if p.fade_in or p.fade_out:
                    audio = apply_fade(
                        audio, sample_rate,
                        fade_in_s=FADE_DURATION_S if p.fade_in else 0,
                        fade_out_s=FADE_DURATION_S if p.fade_out else 0
                    )
                
                if len(audio) == 0:
                    raise RuntimeError("Generated audio is empty. Model might have failed.")
                
                safe_prompt = sanitize_filename(p.prompt)
                suffix = f"_v{variation + 1}" if p.num_variations > 1 else ""
                filename = f"{int(time.time())}_{safe_prompt}{suffix}.wav"
                out_path = os.path.join(final_out_dir, filename)
                
                audio_int16 = (audio * 32767).astype(np.int16)
                try:
                    wavfile.write(out_path, rate=sample_rate, data=audio_int16)
                except OSError as e:
                    raise RuntimeError(f"Failed to save audio to disk:\n{e}")
                
                metadata = {
                    'prompt': p.prompt,
                    'negative_prompt': p.negative_prompt,
                    'model': p.model_name,
                    'duration': p.duration,
                    'steps': p.steps,
                    'guidance': p.guidance,
                    'seed': base_seed,
                    'sample_rate': sample_rate,
                    'device': actual_device,
                    'variation': variation + 1,
                    'total_variations': p.num_variations,
                    'generation_time': time.time() - variation_start,
                    'timestamp': datetime.now().isoformat(),
                    'chunked_generation': num_chunks > 1,
                    'num_chunks': num_chunks,
                    'path': out_path,
                    'filename': filename
                }
                
                saved_paths.append(out_path)
                metadata_list.append(metadata)
                logger.info(f"Variation {variation + 1} generated: {out_path}")

            # --- Update Manifest ---
            # Save a single manifest.json file in the output directory containing all metadata
            manifest_path = os.path.join(final_out_dir, "manifest.json")
            manifest_data = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest_data = json.load(f)
                except Exception:
                    manifest_data = []
            
            manifest_data.extend(metadata_list)
            
            try:
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save manifest.json: {e}")

            self.progress_updated.emit(100)
            self.status_updated.emit("Generation complete")
            self.eta_updated.emit("")
            
            self.generation_completed.emit(saved_paths[0], {'paths': saved_paths, 'metadata': metadata_list, 'gen_time': time.time() - gen_start_time})

        except InterruptedError:
            logger.warning("Generation interrupted by user.")
            self.status_updated.emit("Cancelled")
            self.progress_updated.emit(0)
            self.eta_updated.emit("")
            self.cancelled.emit()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.critical("GPU Out of Memory.")
                friendly_msg = (
                    "GPU Out of Memory!\n\n"
                    "Your graphics card ran out of VRAM. Try:\n"
                    "- Lowering the Chunk Size (e.g., 5s or 10s)\n"
                    "- Lowering the number of Steps\n"
                    "- Enabling CPU Offload (Low VRAM Mode)\n"
                    "- Closing other GPU-intensive applications"
                )
                self.error_occurred.emit(friendly_msg)
            else:
                logger.error(f"RuntimeError: {e}", exc_info=True)
                self.error_occurred.emit(str(e))

        except AttributeError as e:
            if "_update_model_kwargs_for_generation" in str(e):
                logger.critical("Library version conflict detected.")
                self.error_occurred.emit(
                    "Library Version Conflict Detected!\n\n"
                    "Please run:\npip install --upgrade diffusers transformers"
                )
            else:
                logger.error(f"AttributeError: {e}", exc_info=True)
                self.error_occurred.emit(str(e))

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

        finally:
            self.clear_cache()

    def clear_cache(self):
        """Clears PyTorch's intermediate memory cache without unloading the model."""
        self.pipe = None  # Remove instance reference, class reference (_shared_pipe) keeps model alive
        gc.collect()
        if ML_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")

    @classmethod
    def unload_pipeline(cls):
        """Public method to unload pipeline (acquires lock)."""
        with cls._pipe_lock:
            cls._unload_pipeline_unsafe()

    @classmethod
    def _unload_pipeline_unsafe(cls):
        """Internal method to unload pipeline (assumes lock is already held). Prevents deadlocks."""
        if cls._shared_pipe is not None:
            try:
                logger.info(f"Unloading model '{cls._shared_pipe_model}' from memory...")
                pipe = cls._shared_pipe
                
                # Remove accelerate hooks if CPU offload was used
                if hasattr(pipe, "remove_hooks"):
                    try:
                        pipe.remove_hooks()
                    except Exception:
                        pass
                        
                # Delete the pipeline reference explicitly
                del pipe
                cls._shared_pipe = None
                cls._shared_pipe_model = None
                cls._shared_pipe_device = None
                cls._shared_pipe_offload = None
                
                # Force garbage collection and clear VRAM
                gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                logger.info("Model unloaded successfully.")
            except Exception as e:
                logger.error(f"Error unloading pipeline: {e}")


# --- Custom Widgets ---
class WaveformWidget(QWidget):
    seekRequested = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_audio = None
        self._peaks = None
        self._peaks_width = 0
        self.sample_rate = 16000
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.playback_position = 0.0
        
    def set_audio(self, path: str):
        try:
            with open(path, 'rb') as f:
                f.read(4)
        except OSError as e:
            logger.error(f"Cannot open file: {e}")
            self._raw_audio = None
            self._peaks = None
            self.update()
            return

        try:
            sample_rate, data = wavfile.read(path)
            self.sample_rate = sample_rate
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0
            self._raw_audio = data
            self._peaks = None
            self._ensure_peaks(self.width())
            self.update()
        except Exception as e:
            logger.error(f"Failed to load waveform: {e}")
            self._raw_audio = None
            self._peaks = None
            self.update()
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._peaks = None
        self.update()
        
    def _ensure_peaks(self, width: int):
        if self._peaks is not None and self._peaks_width == width:
            return
        if self._raw_audio is None or width <= 0:
            self._peaks = None
            return
            
        n = len(self._raw_audio)
        if n == 0:
            self._peaks = np.zeros(width, dtype=np.float32)
            self._peaks_width = width
            return
            
        abs_audio = np.abs(self._raw_audio)
        if n <= width:
            self._peaks = np.pad(abs_audio, (0, width - n), 'constant', constant_values=0).astype(np.float32)
        else:
            chunk_size = n // width
            trimmed = abs_audio[:width * chunk_size].reshape(width, chunk_size)
            self._peaks = trimmed.max(axis=1).astype(np.float32)
        self._peaks_width = width
        
    def set_playback_position(self, position: float):
        self.playback_position = max(0.0, min(1.0, position))
        self.update()
        
    def clear(self):
        self._raw_audio = None
        self._peaks = None
        self.update()
        
    def mousePressEvent(self, event: QMouseEvent):
        if self._raw_audio is not None and event.button() == Qt.LeftButton:
            pos = event.position().x() / max(self.width(), 1)
            self.seekRequested.emit(pos)
            self.set_playback_position(pos)
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        if self._peaks is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "No audio loaded")
            return
            
        pen = QPen(QColor(42, 130, 218))
        pen.setWidth(1)
        painter.setPen(pen)
        mid = height // 2
        bar_heights = (self._peaks * (height * 0.45)).astype(int)
        lines = [QLineF(x, mid - bh, x, mid + bh) for x, bh in enumerate(bar_heights)]
        painter.drawLines(lines)
            
        if self.playback_position >= 0:
            pos_x = int(self.playback_position * (width - 1))
            pen = QPen(QColor(255, 100, 100))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(pos_x, 0, pos_x, height)
            
        pen = QPen(QColor(60, 60, 60))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, mid, width, mid)


class ClickableLabel(QLabel):
    clicked = Signal()
    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.clicked.emit()


class PromptHistoryLineEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_index = -1
        self._saved_text = ""
        self.setPlaceholderText("Describe the audio... (Ctrl+Enter to generate, Ctrl+Up/Down for history)")
        self.setMaximumHeight(70)
        self.setTabChangesFocus(True)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up and (event.modifiers() & Qt.ControlModifier):
            self.navigate_history(-1)
        elif event.key() == Qt.Key_Down and (event.modifiers() & Qt.ControlModifier):
            self.navigate_history(1)
        elif event.key() == Qt.Key_Return and (event.modifiers() & Qt.ControlModifier):
            self.window().start_generation()
        else:
            super().keyPressEvent(event)
        
    def navigate_history(self, direction):
        if not self.history:
            return
        if self.history_index == -1:
            self._saved_text = self.toPlainText()
        new_index = max(-1, min(len(self.history) - 1, self.history_index + direction))
        if new_index == self.history_index:
            return
        self.history_index = new_index
        self.setPlainText(self._saved_text if self.history_index == -1 else self.history[self.history_index])
            
    def add_to_history(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > MAX_PROMPT_HISTORY:
                self.history.pop(0)
            self.history_index = -1


# --- Main Window ---
class AudioLDM2Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1200, 800)
        self.setMinimumSize(1000, 650)
        log_system_info()

        if not ML_AVAILABLE:
            QMessageBox.critical(self, "Missing Dependencies",
                "ML libraries (torch, diffusers, transformers) are not installed.\n"
                "Please install them with:\n"
                "pip install torch diffusers transformers")
            sys.exit(1)

        # Initialize custom JSON settings (config.json in CWD)
        self.settings = JsonSettings(os.path.join(CWD, "config.json"))
        
        self.worker_thread = None
        self.worker = None
        self.current_audio_path = None
        self.current_audio_paths = []
        
        self.batch_state = BatchState.IDLE
        self.batch_prompts = []
        self.batch_index = 0
        
        self.prompt_history = []
        self.setAcceptDrops(True)
        
        self.check_library_versions()
        self.setup_ui()
        self.setup_audio()
        self.setup_statusbar()
        self.load_settings()
        self.setup_shortcuts()
        self.check_interrupted_batch()
        
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.auto_save_state)
        self.autosave_timer.start(AUTO_SAVE_INTERVAL_MS)

    def check_library_versions(self):
        try:
            t_ver = transformers.__version__.split('.')
            d_ver = diffusers.__version__.split('.')
            if int(t_ver[0]) >= 4 and int(t_ver[1]) >= 40:
                if int(d_ver[0]) == 0 and int(d_ver[1]) < 29:
                    QMessageBox.warning(self, "Potential Version Conflict",
                        "You may experience a crash due to a library bug.\n"
                        "If generation fails, run:\n"
                        "pip install --upgrade diffusers transformers")
        except Exception:
            pass

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        self.create_left_panel(splitter)
        self.create_right_panel(splitter)
        splitter.setSizes([420, 780])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.create_menus()
        
        try:
            # JsonSettings automatically returns QByteArray if it was saved as one
            geo = self.settings.value("geometry")
            if geo:
                self.restoreGeometry(geo)
            state = self.settings.value("windowState")
            if state:
                self.restoreState(state)
        except Exception as e:
            logger.warning(f"Failed to restore window geometry: {e}")

    def create_left_panel(self, parent):
        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.mode_tabs = QTabWidget()
        left_layout.addWidget(self.mode_tabs)

        # --- Tab 1: Generate ---
        generate_widget = QWidget()
        gen_outer_layout = QVBoxLayout(generate_widget)
        gen_outer_layout.setContentsMargins(0, 0, 0, 0)

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setFrameShape(QFrame.NoFrame)
        gen_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        gen_content = QWidget()
        gen_layout = QVBoxLayout(gen_content)
        gen_layout.setContentsMargins(8, 8, 8, 8)
        gen_layout.setSpacing(8)

        prompt_group = QGroupBox("Prompt")
        p_layout = QVBoxLayout(prompt_group)
        p_layout.setSpacing(6)
        
        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Presets:")
        preset_lbl.setFixedWidth(50)
        preset_row.addWidget(preset_lbl)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([""] + DEFAULT_PROMPT_PRESETS)
        self.preset_combo.currentTextChanged.connect(self.on_preset_selected)
        preset_row.addWidget(self.preset_combo, 1)
        p_layout.addLayout(preset_row)
        
        self.prompt_input = PromptHistoryLineEdit()
        p_layout.addWidget(self.prompt_input)

        neg_row = QHBoxLayout()
        neg_lbl = QLabel("Negative:")
        neg_lbl.setFixedWidth(50)
        neg_row.addWidget(neg_lbl)
        self.neg_preset_combo = QComboBox()
        self.neg_preset_combo.addItems([""] + NEGATIVE_PROMPT_PRESETS)
        self.neg_preset_combo.currentTextChanged.connect(self.on_neg_preset_selected)
        neg_row.addWidget(self.neg_preset_combo, 1)
        p_layout.addLayout(neg_row)
        
        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setPlaceholderText("What to avoid...")
        p_layout.addWidget(self.neg_prompt_input)
        gen_layout.addWidget(prompt_group)

        param_group = QGroupBox("Parameters")
        f_layout = QFormLayout(param_group)
        f_layout.setSpacing(6)
        f_layout.setLabelAlignment(Qt.AlignRight)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_MODELS)
        model_row.addWidget(self.model_combo, 1)
        
        save_model_btn = QToolButton()
        save_model_btn.setText("+")
        save_model_btn.setToolTip("Save current model to list")
        save_model_btn.clicked.connect(self.save_current_model)
        model_row.addWidget(save_model_btn)
        f_layout.addRow("Model:", model_row)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 600.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix(" s")
        f_layout.addRow("Duration:", self.duration_spin)

        self.chunk_size_spin = QDoubleSpinBox()
        self.chunk_size_spin.setRange(MIN_CHUNK_S, 60.0)
        self.chunk_size_spin.setValue(10.0)
        self.chunk_size_spin.setSingleStep(0.5)
        self.chunk_size_spin.setSuffix(" s")
        f_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.chunk_preview_label = QLabel("")
        self.chunk_preview_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        f_layout.addRow("", self.chunk_preview_label)
        
        self.duration_spin.valueChanged.connect(self.update_chunk_preview)
        self.chunk_size_spin.valueChanged.connect(self.update_chunk_preview)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 500)
        self.steps_spin.setValue(200)
        f_layout.addRow("Steps:", self.steps_spin)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(3.5)
        self.guidance_spin.setSingleStep(0.5)
        f_layout.addRow("Guidance:", self.guidance_spin)

        seed_row = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 999999999)
        self.seed_spin.setValue(-1)
        seed_row.addWidget(self.seed_spin, 1)
        
        random_seed_btn = QToolButton()
        random_seed_btn.setText("🎲")
        random_seed_btn.setToolTip("Random seed")
        random_seed_btn.clicked.connect(lambda: self.seed_spin.setValue(-1))
        seed_row.addWidget(random_seed_btn)
        f_layout.addRow("Seed:", seed_row)

        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 10)
        self.variations_spin.setValue(1)
        f_layout.addRow("Variations:", self.variations_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_combo.setCurrentText("mps")
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        f_layout.addRow("Device:", self.device_combo)

        cache_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setPlaceholderText(DEFAULT_CACHE_DIR)
        cache_browse_btn = QPushButton("…")
        cache_browse_btn.setFixedWidth(30)
        cache_browse_btn.clicked.connect(self.browse_cache_dir)
        cache_row.addWidget(self.cache_dir_edit)
        cache_row.addWidget(cache_browse_btn)
        f_layout.addRow("Cache:", cache_row)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("./generated_audio")
        out_browse_btn = QPushButton("…")
        out_browse_btn.setFixedWidth(30)
        out_browse_btn.clicked.connect(self.browse_output_dir)
        out_row.addWidget(out_browse_btn)
        
        open_out_btn = QPushButton("📂")
        open_out_btn.setFixedWidth(30)
        open_out_btn.clicked.connect(self.open_output_folder)
        out_row.addWidget(open_out_btn)
        f_layout.addRow("Output:", out_row)

        self.section_edit = QLineEdit()
        self.section_edit.setPlaceholderText("e.g., SFX, Music (leave empty for root)")
        f_layout.addRow("Section:", self.section_edit)

        self.cpu_offload_check = QCheckBox("Enable CPU Offload (Low VRAM)")
        f_layout.addRow("", self.cpu_offload_check)
        gen_layout.addWidget(param_group)

        post_group = QGroupBox("Post-Processing")
        post_layout = QVBoxLayout(post_group)
        post_layout.setSpacing(4)
        self.normalize_check = QCheckBox("Normalize audio volume")
        self.normalize_check.setChecked(True)
        post_layout.addWidget(self.normalize_check)
        self.trim_check = QCheckBox("Trim silence")
        self.trim_check.setChecked(True)
        post_layout.addWidget(self.trim_check)
        
        fade_row = QHBoxLayout()
        self.fade_in_check = QCheckBox("Fade in")
        self.fade_in_check.setChecked(True)
        fade_row.addWidget(self.fade_in_check)
        self.fade_out_check = QCheckBox("Fade out")
        self.fade_out_check.setChecked(True)
        fade_row.addWidget(self.fade_out_check)
        post_layout.addLayout(fade_row)
        gen_layout.addWidget(post_group)

        gen_btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2A82DA; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3A92EA; }"
            "QPushButton:pressed { background-color: #1A72CA; }"
            "QPushButton:disabled { background-color: #444444; color: #888888; }"
        )
        self.generate_btn.clicked.connect(self.start_generation)
        gen_btn_row.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d0493b; }"
            "QPushButton:pressed { background-color: #b0291b; }"
            "QPushButton:disabled { background-color: #444444; color: #888888; }"
        )
        self.cancel_btn.clicked.connect(self.cancel_generation)
        gen_btn_row.addWidget(self.cancel_btn)
        gen_layout.addLayout(gen_btn_row)
        gen_layout.addStretch()
        
        gen_scroll.setWidget(gen_content)
        gen_outer_layout.addWidget(gen_scroll)
        self.mode_tabs.addTab(generate_widget, "Generate")

        # --- Tab 2: Batch ---
        batch_widget = QWidget()
        batch_outer_layout = QVBoxLayout(batch_widget)
        batch_outer_layout.setContentsMargins(0, 0, 0, 0)

        batch_scroll = QScrollArea()
        batch_scroll.setWidgetResizable(True)
        batch_scroll.setFrameShape(QFrame.NoFrame)
        batch_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        batch_content = QWidget()
        batch_layout = QVBoxLayout(batch_content)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        batch_layout.setSpacing(8)
        
        batch_layout.addWidget(QLabel("Select a .txt file with one prompt per line:"))
        self.batch_file_btn = QPushButton("Select Text File")
        self.batch_file_btn.clicked.connect(self.select_batch_file)
        batch_layout.addWidget(self.batch_file_btn)
        
        self.batch_file_label = QLabel("No file selected")
        self.batch_file_label.setWordWrap(True)
        self.batch_file_label.setStyleSheet("color: #888; font-style: italic;")
        batch_layout.addWidget(self.batch_file_label)
        
        batch_settings_group = QGroupBox("Batch Settings")
        bsl = QFormLayout(batch_settings_group)
        bsl.setSpacing(6)
        bsl.setLabelAlignment(Qt.AlignRight)
        self.batch_variations_spin = QSpinBox()
        self.batch_variations_spin.setRange(1, 10)
        self.batch_variations_spin.setValue(1)
        bsl.addRow("Variations:", self.batch_variations_spin)
        self.batch_delay_spin = QDoubleSpinBox()
        self.batch_delay_spin.setRange(0, 30)
        self.batch_delay_spin.setValue(1.0)
        self.batch_delay_spin.setSuffix(" s")
        bsl.addRow("Delay:", self.batch_delay_spin)
        self.batch_continue_on_error_check = QCheckBox("Continue on Error")
        self.batch_continue_on_error_check.setChecked(True)
        bsl.addRow("", self.batch_continue_on_error_check)
        batch_layout.addWidget(batch_settings_group)
        
        self.batch_start_btn = QPushButton("Start Batch Generation")
        self.batch_start_btn.setMinimumHeight(40)
        self.batch_start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-size: 13px; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #37be70; }"
            "QPushButton:pressed { background-color: #179e50; }"
            "QPushButton:disabled { background-color: #444444; color: #888888; }"
        )
        self.batch_start_btn.clicked.connect(self.start_batch)
        batch_layout.addWidget(self.batch_start_btn)
        
        self.batch_progress_label = QLabel("")
        self.batch_progress_label.setWordWrap(True)
        batch_layout.addWidget(self.batch_progress_label)
        
        batch_layout.addWidget(QLabel("Batch Log:"))
        self.batch_log = QPlainTextEdit()
        self.batch_log.setReadOnly=True
        self.batch_log.setMaximumBlockCount(1000)
        batch_layout.addWidget(self.batch_log)
        batch_layout.addStretch()
        
        batch_scroll.setWidget(batch_content)
        batch_outer_layout.addWidget(batch_scroll)
        self.mode_tabs.addTab(batch_widget, "Batch")

        prog_group = QGroupBox("Status")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setSpacing(4)
        prog_layout.setContentsMargins(8, 16, 8, 8)
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_row.addWidget(self.progress_bar, 1)
        self.eta_label = QLabel("")
        self.eta_label.setMinimumWidth(70)
        self.eta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_row.addWidget(self.eta_label)
        prog_layout.addLayout(prog_row)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        prog_layout.addWidget(self.status_label)
        left_layout.addWidget(prog_group)
        parent.addWidget(left_panel)

    def create_right_panel(self, parent):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs)

        player_widget = QWidget()
        player_layout = QVBoxLayout(player_widget)
        player_layout.setContentsMargins(8, 8, 8, 8)
        player_layout.setSpacing(8)

        wave_group = QGroupBox("Waveform")
        wave_layout = QVBoxLayout(wave_group)
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.seekRequested.connect(self._seek_to)
        wave_layout.addWidget(self.waveform_widget)
        player_layout.addWidget(wave_group)

        play_group = QGroupBox("Playback")
        play_layout = QVBoxLayout(play_group)
        play_layout.setSpacing(6)
        play_btn_row = QHBoxLayout()
        play_btn_row.setSpacing(6)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedHeight(36)
        self.load_btn.clicked.connect(self.open_audio)
        play_btn_row.addWidget(self.load_btn)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedHeight(36)
        self.play_btn.setMinimumWidth(80)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        play_btn_row.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_audio)
        play_btn_row.addWidget(self.save_btn)
        
        play_btn_row.addSpacing(10)
        play_btn_row.addWidget(QLabel("Variation:"))
        self.variation_combo = QComboBox()
        self.variation_combo.setEnabled(False)
        self.variation_combo.currentIndexChanged.connect(self.on_variation_changed)
        play_btn_row.addWidget(self.variation_combo, 1)
        play_layout.addLayout(play_btn_row)

        vol_layout = QHBoxLayout()
        vol_layout.setSpacing(6)
        vol_lbl = QLabel("Volume:")
        vol_lbl.setFixedWidth(50)
        vol_layout.addWidget(vol_lbl)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_slider.valueChanged.connect(self.set_volume)
        vol_layout.addWidget(self.vol_slider)
        self.vol_label = QLabel("50%")
        self.vol_label.setFixedWidth(35)
        vol_layout.addWidget(self.vol_label)
        play_layout.addLayout(vol_layout)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2A82DA;")
        play_layout.addWidget(self.time_label)
        player_layout.addWidget(play_group)
        player_layout.addStretch()
        self.right_tabs.addTab(player_widget, "Player")

        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(8, 8, 8, 8)
        history_layout.setSpacing(6)

        header_row = QHBoxLayout()
        hist_lbl = QLabel("History")
        hist_lbl.setStyleSheet("font-size: 13px; font-weight: bold;")
        header_row.addWidget(hist_lbl)
        header_row.addStretch()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.setFixedWidth(180)
        header_row.addWidget(self.search_edit)
        
        export_hist_btn = QPushButton("Export")
        export_hist_btn.setFixedHeight(28)
        export_hist_btn.clicked.connect(self.export_history)
        header_row.addWidget(export_hist_btn)
        
        clear_hist_btn = QPushButton("Clear")
        clear_hist_btn.setFixedHeight(28)
        clear_hist_btn.clicked.connect(self.clear_history)
        header_row.addWidget(clear_hist_btn)
        history_layout.addLayout(header_row)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(lambda: self._do_filter(self.search_edit.text()))
        self.search_edit.textChanged.connect(self._search_timer.start)

        self.playlist_table = QTableWidget()
        self.playlist_table.setColumnCount(5)
        self.playlist_table.setHorizontalHeaderLabels(["Prompt", "Duration", "Date", "Seed", "Model"])
        self.playlist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 5):
            self.playlist_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.playlist_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.playlist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.playlist_table.setAlternatingRowColors(True)
        self.playlist_table.itemDoubleClicked.connect(self.play_from_playlist)
        self.playlist_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlist_table.customContextMenuRequested.connect(self.show_history_context_menu)
        history_layout.addWidget(self.playlist_table)
        self.right_tabs.addTab(history_widget, "History")
        parent.addWidget(right_panel)

    def setup_audio(self):
        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.mediaStatusChanged.connect(self.on_media_status)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.set_volume(50)

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        device_info = "CPU"
        if torch.cuda.is_available():
            device_info = f"GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "MPS"
        self.device_status_label = QLabel(f"Device: {device_info}")
        self.status_bar.addPermanentWidget(self.device_status_label)
        
        self.gpu_mem_label = ClickableLabel()
        self.gpu_mem_label.clicked.connect(self.show_gpu_memory)
        if torch.cuda.is_available():
            self.update_gpu_memory_label()
            self.gpu_mem_timer = QTimer(self)
            self.gpu_mem_timer.timeout.connect(self.update_gpu_memory_label)
            self.gpu_mem_timer.start(GPU_MEM_POLL_MS)
        self.status_bar.addPermanentWidget(self.gpu_mem_label)
        self.status_bar.showMessage("Ready")

    def update_gpu_memory_label(self):
        if not torch.cuda.is_available(): return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  
        self.gpu_mem_label.setText(f"VRAM: {allocated:.1f}/{total:.1f} GB")

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.start_generation)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_audio)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_audio)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_generation)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(lambda: self.prompt_input.setFocus())
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.regenerate_new_seed)

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Export Session", self.export_session)
        file_menu.addAction("Import Session", self.import_session)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close).setShortcut("Ctrl+Q")

        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction("Show GPU Memory", self.show_gpu_memory)
        tools_menu.addAction("Clear GPU Cache", self.clear_gpu_cache)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("Clear Prompt", lambda: self.prompt_input.clear()).setShortcut("Ctrl+Shift+X")
        edit_menu.addAction("Paste from Clipboard", self.paste_from_clipboard).setShortcut("Ctrl+V")
        edit_menu.addAction("Copy Settings as JSON", self.copy_settings_to_clipboard)
        edit_menu.addSeparator()
        edit_menu.addAction("Save Settings Preset", self.save_preset)
        edit_menu.addAction("Load Settings Preset", self.load_preset)
        edit_menu.addSeparator()
        edit_menu.addAction("Reset All Settings", self.reset_settings)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("Documentation", self.open_docs)
        help_menu.addAction("Prompt Tips", self.show_prompt_tips)
        help_menu.addSeparator()
        help_menu.addAction("About", self.show_about)

    def show_gpu_memory(self):
        if not ML_AVAILABLE or not torch.cuda.is_available():
            QMessageBox.information(self, "GPU Memory", "CUDA is not available.")
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        QMessageBox.information(self, "GPU Memory",
            f"GPU: {torch.cuda.get_device_name(0)}\n\n"
            f"Total VRAM:     {total:.2f} GB\n"
            f"Allocated:      {allocated:.2f} GB\n"
            f"Reserved:       {reserved:.2f} GB\n"
            f"Free (approx):  {total - reserved:.2f} GB")

    def clear_gpu_cache(self):
        if not ML_AVAILABLE or not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        self.update_gpu_memory_label()
        self.status_bar.showMessage("GPU cache cleared", 3000)

    def reset_settings(self):
        reply = QMessageBox.question(self, "Reset Settings", 
            "Reset all UI settings to defaults? (Model cache and generated files are not affected.)",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.settings.clear()
            self.load_settings()
            self.status_bar.showMessage("Settings reset to defaults", 3000)

    def on_preset_selected(self, text):
        if text: self.prompt_input.setPlainText(text)

    def on_neg_preset_selected(self, text):
        if text: self.neg_prompt_input.setText(text)

    def save_current_model(self):
        model = self.model_combo.currentText().strip()
        if model and self.model_combo.findText(model) == -1:
            self.model_combo.addItem(model)
            self.model_combo.setCurrentText(model)
            custom_models = self.settings.value("custom_models", [])
            if model not in custom_models:
                custom_models.append(model)
                self.settings.setValue("custom_models", custom_models)
                self.settings.sync()

    def browse_cache_dir(self):
        cur = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        path = QFileDialog.getExistingDirectory(self, "Select Model Cache Folder", cur if os.path.isdir(cur) else "")
        if path: self.cache_dir_edit.setText(path)

    def browse_output_dir(self):
        cur = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        path = QFileDialog.getExistingDirectory(self, "Select Output Audio Folder", cur if os.path.isdir(cur) else "")
        if path: self.output_dir_edit.setText(path)

    def open_output_folder(self):
        out = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        if self.section_edit.text().strip():
            out = os.path.join(out, sanitize_filename(self.section_edit.text().strip()))
        os.makedirs(out, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    def update_chunk_preview(self):
        duration = self.duration_spin.value()
        chunk = self.chunk_size_spin.value()
        if chunk <= OVERLAP_S:
            self.chunk_preview_label.setText("⚠️ Chunk size must be > 1.0s")
            self.chunk_preview_label.setStyleSheet("color: #e74c3c; font-style: italic; font-size: 10px;")
            return
        if duration > chunk:
            n = math.ceil((duration - OVERLAP_S) / (chunk - OVERLAP_S))
            warn = " ⚠️ too many chunks" if n > 50 else ""
            self.chunk_preview_label.setText(f"→ Will generate {n} chunks of ~{chunk:.1f}s with 1.0s crossfade{warn}")
            self.chunk_preview_label.setStyleSheet(
                "color: #e74c3c; font-style: italic; font-size: 10px;" if n > 50 
                else "color: #888; font-style: italic; font-size: 10px;"
            )
        else:
            self.chunk_preview_label.setText("→ Will generate as a single chunk")
            self.chunk_preview_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")

    def on_device_changed(self, text: str):
        self.cpu_offload_check.setEnabled(text != "cpu")
        if text == "cpu": self.cpu_offload_check.setChecked(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(STYLESHEET + "QMainWindow { border: 3px dashed #2A82DA; }")

    def dragLeaveEvent(self, event):
        self.setStyleSheet(STYLESHEET)

    def dropEvent(self, event):
        self.setStyleSheet(STYLESHEET)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.wav', '.mp3', '.flac')):
                self._load_audio_file(path)
                break
            elif path.lower().endswith('.txt'):
                self._load_batch_file(path)
                break

    def _load_audio_file(self, path):
        try:
            with open(path, 'rb') as f:
                f.read(4)
        except OSError as e:
            QMessageBox.warning(self, "Cannot Open File", f"Cannot read file:\n{e}")
            return
        self.current_audio_path = path
        self.current_audio_paths = [path]
        self.variation_combo.blockSignals(True)
        self.variation_combo.clear()
        self.variation_combo.addItem("Default")
        self.variation_combo.blockSignals(False)
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.waveform_widget.set_audio(path)
        self.right_tabs.setCurrentIndex(0)
        self.settings.setValue("last_audio_path", path)
        self.settings.sync()

    def _load_batch_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            self.batch_prompts = lines
            self.batch_file_label.setText(path)
            self.batch_log.appendPlainText(f"Loaded {len(self.batch_prompts)} prompts from {os.path.basename(path)}")
            self.mode_tabs.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Batch File Error", str(e))

    def start_generation(self) -> bool:
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "Busy", "The engine is still cleaning up from the last generation. Please wait a moment.")
            return False

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
            return False

        model = self.model_combo.currentText().strip()
        if not model or '/' not in model:
            QMessageBox.warning(self, "Invalid Model", "Model name must be in 'org/model' format (e.g., 'cvssp/audioldm2').")
            return False

        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Output Error", f"Cannot create output folder:\n{e}")
            return False

        cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        try:
            # Pre-validate cache directory to prevent thread crashes
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Cache Directory Error", 
                f"Cannot create or access cache directory '{cache_dir}'.\n"
                f"Please change the 'Cache' folder in the UI to a valid path (e.g., ./model_cache).")
            return False

        self.prompt_input.add_to_history(prompt)

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.variation_combo.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.status_label.setText("Starting...")
        self.status_bar.showMessage("Generation in progress...")

        params = GenerationParams(
            prompt=prompt,
            negative_prompt=self.neg_prompt_input.text().strip(),
            model_name=model,
            duration=self.duration_spin.value(),
            steps=self.steps_spin.value(),
            guidance=self.guidance_spin.value(),
            seed=self.seed_spin.value(),
            device=self.device_combo.currentText(),
            cache_dir=cache_dir,
            output_dir=out_dir,
            num_variations=self.variations_spin.value(),
            normalize=self.normalize_check.isChecked(),
            fade_in=self.fade_in_check.isChecked(),
            fade_out=self.fade_out_check.isChecked(),
            trim=self.trim_check.isChecked(),
            chunk_size=self.chunk_size_spin.value(),
            use_cpu_offload=self.cpu_offload_check.isChecked(),
            section=self.section_edit.text().strip()
        )

        self.worker_thread = QThread()
        self.worker = AudioGenerationWorker(params)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.worker.status_updated.connect(self.status_label.setText, Qt.QueuedConnection)
        self.worker.eta_updated.connect(self.eta_label.setText, Qt.QueuedConnection)
        self.worker.generation_completed.connect(self.generation_finished, Qt.QueuedConnection)
        self.worker.error_occurred.connect(self.generation_error, Qt.QueuedConnection)
        self.worker.cancelled.connect(self.generation_cancelled, Qt.QueuedConnection)
        
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
        return True

    def regenerate_new_seed(self):
        self.seed_spin.setValue(-1)
        self.start_generation()

    def cancel_generation(self):
        if self.batch_state == BatchState.RUNNING:
            reply = QMessageBox.question(self, "Cancel Batch",
                "Cancelling will stop the entire batch. Continue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            self.batch_state = BatchState.CANCELLED
            
        if self.worker:
            self.worker.is_cancelled = True
            self.status_label.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.status_bar.showMessage("Cancelling generation...")
            
        if self.batch_state == BatchState.CANCELLED:
            self.batch_progress_label.setText("Batch cancelled.")
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Batch cancelled by user.")

    def generation_cancelled(self):
        self.progress_bar.setVisible(False)
        self.eta_label.setText("")
        self.status_label.setText("Ready")
        if self.batch_state == BatchState.RUNNING:
            self.batch_state = BatchState.CANCELLED
        self.status_bar.showMessage("Generation cancelled", 3000)

    def generation_finished(self, path, metadata):
        self.current_audio_path = path
        self.current_audio_paths = metadata.get('paths', [path])
        self.progress_bar.setVisible(False)
        self.eta_label.setText("")
        self.status_label.setText("Ready")
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        self.variation_combo.blockSignals(True)
        self.variation_combo.clear()
        if len(self.current_audio_paths) > 1:
            for i in range(len(self.current_audio_paths)):
                self.variation_combo.addItem(f"Variation {i+1}")
            self.variation_combo.setEnabled(True)
        else:
            self.variation_combo.addItem("Default")
        self.variation_combo.blockSignals(False)

        self.waveform_widget.set_audio(path)
        meta_list = metadata['metadata']
        
        # Add to internal history list (does not save sidecar file to disk)
        history_data = self.settings.value("history", [])
        
        # Update UI Table and internal history for all variations
        row = self.playlist_table.rowCount()
        for i, meta in enumerate(meta_list):
            history_data.append(meta)
            
            self.playlist_table.insertRow(row + i)
            prompt_item = QTableWidgetItem(meta['prompt'])
            prompt_item.setToolTip(meta['path'])
            if meta['path'] and not os.path.exists(meta['path']):
                prompt_item.setForeground(QColor('red'))
            self.playlist_table.setItem(row + i, 0, prompt_item)
            self.playlist_table.setItem(row + i, 1, QTableWidgetItem(f"{meta['duration']:.1f}s"))
            self.playlist_table.setItem(row + i, 2, QTableWidgetItem(meta['timestamp'][:10]))
            self.playlist_table.setItem(row + i, 3, QTableWidgetItem(str(meta['seed'])))
            self.playlist_table.setItem(row + i, 4, QTableWidgetItem(meta['model'].split('/')[-1]))

        if len(history_data) > MAX_HISTORY:
            history_data = history_data[-MAX_HISTORY:]
            
        self.settings.setValue("history", history_data)
        self.settings.sync()

        self.settings.setValue("last_audio_path", path)
        self.settings.sync()
        
        gen_time = metadata.get('gen_time', 0.0)
        self.right_tabs.setCurrentIndex(0)
        
        if self.settings.value("autoplay", False, type=bool):
            self.toggle_playback()
            
        self.status_bar.showMessage(f"Generated {len(self.current_audio_paths)} audio file(s) in {gen_time:.2f}s", 5000)

    def generation_error(self, msg):
        self.progress_bar.setVisible(False)
        self.eta_label.setText("")
        self.status_label.setText("Error")
        if self.batch_state == BatchState.RUNNING:
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {msg[:100]}...")
            if not self.batch_continue_on_error_check.isChecked():
                self.batch_state = BatchState.CANCELLED
        else:
            QMessageBox.critical(self, "Generation Error", msg)

    def _trigger_next_batch_item(self):
        if self.batch_state == BatchState.CANCELLED:
            return
        if not self.start_generation():
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Failed to start generation. Skipping prompt.")
            self.on_worker_thread_finished()

    def on_worker_thread_finished(self):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if self.batch_state == BatchState.RUNNING:
            self.batch_index += 1
            self.save_batch_state()
            if self.batch_index < len(self.batch_prompts):
                next_prompt = self.batch_prompts[self.batch_index]
                self.prompt_input.setPlainText(next_prompt)
                self.batch_progress_label.setText(f"Batch Progress: {self.batch_index + 1} / {len(self.batch_prompts)}")
                self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Generated: {next_prompt[:50]}...")
                QTimer.singleShot(int(self.batch_delay_spin.value() * 1000), self._trigger_next_batch_item)
            else:
                self.finalize_batch_state()
                self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Batch complete!")
                if os.path.exists(BATCH_STATE_FILE):
                    os.remove(BATCH_STATE_FILE)
                QMessageBox.information(self, "Batch Complete", f"All {len(self.batch_prompts)} prompts processed!")
        else:
            self.finalize_batch_state()
            self.status_label.setText("Ready")

    def on_variation_changed(self, index):
        if 0 <= index < len(self.current_audio_paths):
            self.current_audio_path = self.current_audio_paths[index]
            self.waveform_widget.set_audio(self.current_audio_path)
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))

    def toggle_playback(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path): return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.stop()
        else:
            self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))
            self.media_player.play()

    def on_playback_state_changed(self, state):
        self.play_btn.setText("Stop" if state == QMediaPlayer.PlayingState else "Play")

    def play_from_playlist(self, item):
        if isinstance(item, QTableWidgetItem):
            path = self.playlist_table.item(item.row(), 0).toolTip()
        else:
            path = item
        self._play_path(path)

    def _play_path(self, path):
        if path and os.path.exists(path):
            self._load_audio_file(path)
            self.toggle_playback()
        else:
            QMessageBox.warning(self, "File Missing", "This audio file has been moved or deleted.")

    def show_history_context_menu(self, pos):
        item = self.playlist_table.itemAt(pos)
        if not item: return
        path = self.playlist_table.item(item.row(), 0).toolTip()
        menu = QMenu(self)
        menu.addAction("Play", lambda: self._play_path(path))
        menu.addAction("Open in Folder", lambda: self.open_file_in_folder(path))
        menu.addAction("Copy File Path", lambda: QApplication.clipboard().setText(path))
        menu.addAction("Copy Prompt", lambda: QApplication.clipboard().setText(self.playlist_table.item(item.row(), 0).text()))
        menu.addAction("Reuse Settings", lambda: self.reuse_settings(item.row()))
        menu.addSeparator()
        menu.addAction("Delete Entry", lambda: self.delete_history_row(item.row()))
        menu.exec(self.playlist_table.viewport().mapToGlobal(pos))

    def open_file_in_folder(self, path):
        if not path or not os.path.exists(path): return
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', '/select,', path])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', path])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def reuse_settings(self, row):
        # Pulls from internal history database instead of sidecar file
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            entry = history_data[row]
            self.prompt_input.setPlainText(entry.get("prompt", ""))
            self.neg_prompt_input.setText(entry.get("negative_prompt", ""))
            self.duration_spin.setValue(entry.get("duration", 10.0))
            self.steps_spin.setValue(entry.get("steps", 200))
            self.guidance_spin.setValue(entry.get("guidance", 3.5))
            self.seed_spin.setValue(entry.get("seed", -1))
            self.model_combo.setCurrentText(entry.get("model", "cvssp/audioldm2"))
        self.mode_tabs.setCurrentIndex(0)

    def delete_history_row(self, row):
        self.playlist_table.removeRow(row)
        history_data = self.settings.value("history", [])
        if 0 <= row < len(history_data):
            history_data.pop(row)
            self.settings.setValue("history", history_data)
            self.settings.sync()

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.play_btn.setText("Play")
            self.waveform_widget.set_playback_position(0.0)

    def on_position_changed(self, position):
        duration = self.media_player.duration()
        if duration > 0:
            self.time_label.setText(f"{format_time(position / 1000)} / {format_time(duration / 1000)}")
            self.waveform_widget.set_playback_position(position / duration)

    def on_duration_changed(self, duration):
        if duration > 0:
            self.time_label.setText(f"00:00 / {format_time(duration / 1000)}")

    def _seek_to(self, pos: float):
        if self.media_player.duration() > 0:
            self.media_player.setPosition(int(pos * self.media_player.duration()))

    def set_volume(self, val):
        self.audio_output.setVolume(val / 100.0)
        self.vol_label.setText(f"{val}%")

    def save_audio(self):
        if not self.current_audio_path: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", os.path.basename(self.current_audio_path), "WAV Files (*.wav)")
        if path:
            try:
                shutil.copy2(self.current_audio_path, path)
                self.status_bar.showMessage("Audio saved successfully!", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if path: self._load_audio_file(path)

    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if text: self.prompt_input.setPlainText(text)

    def select_batch_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt)")
        if path: self._load_batch_file(path)

    def start_batch(self):
        if not self.batch_prompts or not os.path.exists(self.batch_file_label.text()):
            QMessageBox.warning(self, "Batch Error", "Please select a valid text file first.")
            return
        reply = QMessageBox.question(self, "Start Batch",
            f"About to generate {len(self.batch_prompts)} prompts with {self.batch_variations_spin.value()} variation(s) each.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return

        self.batch_state = BatchState.RUNNING
        self.batch_index = 0
        self.batch_start_btn.setEnabled(False)
        self.batch_file_btn.setEnabled(False)
        self.variations_spin.setValue(self.batch_variations_spin.value())
        self.prompt_input.setPlainText(self.batch_prompts[0])
        self.batch_progress_label.setText(f"Batch Progress: 1 / {len(self.batch_prompts)}")
        self.batch_log.clear()
        self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting batch...")
        self.save_batch_state()
        self.mode_tabs.setCurrentIndex(0)
        if not self.start_generation():
            self.finalize_batch_state()
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Batch aborted due to start error.")

    def save_batch_state(self):
        if self.batch_state == BatchState.RUNNING and self.batch_prompts:
            try:
                with open(BATCH_STATE_FILE, "w") as f:
                    json.dump({"prompts": self.batch_prompts, "index": self.batch_index, "file": self.batch_file_label.text()}, f)
            except Exception as e:
                logger.error(f"Failed to save batch state: {e}")

    def check_interrupted_batch(self):
        if os.path.exists(BATCH_STATE_FILE):
            reply = QMessageBox.question(self, "Resume Batch", 
                "An interrupted batch generation was detected. Would you like to resume?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    with open(BATCH_STATE_FILE, "r") as f:
                        state = json.load(f)
                    self.batch_prompts = state["prompts"]
                    self.batch_index = state["index"]
                    self.batch_file_label.setText(state["file"])
                    self.batch_state = BatchState.RUNNING
                    self.batch_start_btn.setEnabled(False)
                    self.batch_file_btn.setEnabled(False)
                    self.prompt_input.setPlainText(self.batch_prompts[self.batch_index])
                    self.batch_progress_label.setText(f"Batch Progress: {self.batch_index + 1} / {len(self.batch_prompts)}")
                    self.batch_log.clear()
                    self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming batch...")
                    self.mode_tabs.setCurrentIndex(0)
                    if not self.start_generation():
                        self.finalize_batch_state()
                        os.remove(BATCH_STATE_FILE)
                except Exception as e:
                    logger.error(f"Failed to resume batch: {e}")
                    os.remove(BATCH_STATE_FILE)
            else:
                os.remove(BATCH_STATE_FILE)

    def finalize_batch_state(self):
        self.batch_start_btn.setEnabled(True)
        self.batch_file_btn.setEnabled(True)
        if self.batch_state == BatchState.RUNNING:
            self.batch_state = BatchState.IDLE
            self.batch_progress_label.setText("Batch finished or stopped.")
        elif self.batch_state == BatchState.CANCELLED:
            self.batch_state = BatchState.IDLE

    def copy_settings_to_clipboard(self):
        settings_dict = {
            "prompt": self.prompt_input.toPlainText(),
            "negative_prompt": self.neg_prompt_input.text(),
            "model": self.model_combo.currentText(),
            "duration": self.duration_spin.value(),
            "chunk_size": self.chunk_size_spin.value(),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "seed": self.seed_spin.value(),
            "variations": self.variations_spin.value(),
            "device": self.device_combo.currentText(),
            "cpu_offload": self.cpu_offload_check.isChecked(),
        }
        QApplication.clipboard().setText(json.dumps(settings_dict, indent=2))
        self.status_bar.showMessage("Settings copied as JSON", 3000)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip(): return
        preset = {
            "prompt": self.prompt_input.toPlainText(), "negative_prompt": self.neg_prompt_input.text(),
            "model": self.model_combo.currentText(), "duration": self.duration_spin.value(),
            "chunk_size": self.chunk_size_spin.value(), "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(), "variations": self.variations_spin.value(),
            "normalize": self.normalize_check.isChecked(), "trim": self.trim_check.isChecked(),
            "fade_in": self.fade_in_check.isChecked(), "fade_out": self.fade_out_check.isChecked(),
        }
        presets = self.settings.value("presets", {})
        presets[name] = preset
        self.settings.setValue("presets", presets)
        self.settings.sync()

    def load_preset(self):
        presets = self.settings.value("presets", {})
        if not presets: return
        name, ok = QInputDialog.getItem(self, "Load Preset", "Select preset:", list(presets.keys()), 0, False)
        if not ok or not name: return
        p = presets[name]
        self.prompt_input.setPlainText(p.get("prompt", ""))
        self.neg_prompt_input.setText(p.get("negative_prompt", ""))
        self.model_combo.setCurrentText(p.get("model", "cvssp/audioldm2"))
        self.duration_spin.setValue(p.get("duration", 10.0))
        self.chunk_size_spin.setValue(p.get("chunk_size", 10.0))
        self.steps_spin.setValue(p.get("steps", 200))
        self.guidance_spin.setValue(p.get("guidance", 3.5))
        self.variations_spin.setValue(p.get("variations", 1))
        self.normalize_check.setChecked(p.get("normalize", True))
        self.trim_check.setChecked(p.get("trim", True))
        self.fade_in_check.setChecked(p.get("fade_in", True))
        self.fade_out_check.setChecked(p.get("fade_out", True))

    def export_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Session", "session.json", "JSON Files (*.json)")
        if not path: return
        session = {
            "prompt": self.prompt_input.toPlainText(), "negative_prompt": self.neg_prompt_input.text(),
            "settings": {
                "model": self.model_combo.currentText(), "duration": self.duration_spin.value(),
                "chunk_size": self.chunk_size_spin.value(), "steps": self.steps_spin.value(),
                "guidance": self.guidance_spin.value(), "seed": self.seed_spin.value(),
                "variations": self.variations_spin.value(),
            },
            "current_audio": self.current_audio_path
        }
        try:
            with open(path, "w") as f: json.dump(session, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export session:\n{e}")

    def import_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Session", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path, "r") as f: session = json.load(f)
            self.prompt_input.setPlainText(session.get("prompt", ""))
            self.neg_prompt_input.setText(session.get("negative_prompt", ""))
            s = session.get("settings", {})
            self.model_combo.setCurrentText(s.get("model", "cvssp/audioldm2"))
            self.duration_spin.setValue(s.get("duration", 10.0))
            self.chunk_size_spin.setValue(s.get("chunk_size", 10.0))
            self.steps_spin.setValue(s.get("steps", 200))
            self.guidance_spin.setValue(s.get("guidance", 3.5))
            self.seed_spin.setValue(s.get("seed", -1))
            self.variations_spin.setValue(s.get("variations", 1))
            if session.get("current_audio") and os.path.exists(session["current_audio"]):
                self._load_audio_file(session["current_audio"])
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import session:\n{e}")

    def open_docs(self):
        QMessageBox.information(self, "Documentation", 
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "BATCH PROCESSING:\nCreate a .txt file with one prompt per line.\n\n"
            "LONG AUDIO GENERATION:\nSet Duration > 10s and Chunk Size to 10s.\nApp will crossfade chunks seamlessly.\n\n"
            "MEMORY MANAGEMENT:\n- CPU Offload: Enable if < 8GB VRAM.\n- Chunk Size: Lower to 5s if you get OOM errors.\n\n"
            "KEYBOARD SHORTCUTS:\n- Ctrl+Enter: Generate\n- Space: Play/Stop\n- Ctrl+O: Open audio\n- Ctrl+S: Save audio\n- Escape: Cancel\n- Ctrl+Up/Down: History")

    def show_prompt_tips(self):
        QMessageBox.information(self, "Prompt Tips",
            "1. BE SPECIFIC: 'heavy metal double kick drum at 180 BPM'\n"
            "2. DESCRIBE THE SCENE: 'rain falling on a metal roof in a thunderstorm'\n"
            "3. MENTION QUALITY: 'high quality, professional recording'\n"
            "4. USE NEGATIVE PROMPTS: 'noise, distortion, artifacts, low quality'\n"
            "5. EXPERIMENT WITH GUIDANCE: Low (1-3) = creative, High (5-10) = strict")

    def show_about(self):
        QMessageBox.about(self, "About",
            f"<h2>{APP_NAME}</h2><p>Version: {APP_VERSION}</p>"
            f"<p>A GUI for AudioLDM2 text-to-audio generation.</p>")

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export History", "history.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Prompt', 'Duration', 'Date', 'Seed', 'Model', 'File Path'])
                for row in range(self.playlist_table.rowCount()):
                    writer.writerow([
                        self.playlist_table.item(row, 0).text() if self.playlist_table.item(row, 0) else "",
                        self.playlist_table.item(row, 1).text() if self.playlist_table.item(row, 1) else "",
                        self.playlist_table.item(row, 2).text() if self.playlist_table.item(row, 2) else "",
                        self.playlist_table.item(row, 3).text() if self.playlist_table.item(row, 3) else "",
                        self.playlist_table.item(row, 4).text() if self.playlist_table.item(row, 4) else "",
                        self.playlist_table.item(row, 0).toolTip()
                    ])
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def _do_filter(self, text):
        text_lower = text.lower()
        for row in range(self.playlist_table.rowCount()):
            item = self.playlist_table.item(row, 0)
            if item:
                self.playlist_table.setRowHidden(row, text_lower not in item.text().lower())

    def load_history(self):
        history_data = self.settings.value("history", [])
        if not history_data: return
        
        self.playlist_table.setUpdatesEnabled(False)
        self.playlist_table.setRowCount(0)
        self.playlist_table.setRowCount(len(history_data))
        try:
            for row, entry in enumerate(history_data):
                path = entry.get("path", "")
                prompt_item = QTableWidgetItem(entry.get("prompt", ""))
                prompt_item.setToolTip(path)
                if path and not os.path.exists(path):
                    prompt_item.setForeground(QColor('red'))
                self.playlist_table.setItem(row, 0, prompt_item)
                self.playlist_table.setItem(row, 1, QTableWidgetItem(f"{entry.get('duration', 0.0):.1f}s"))
                self.playlist_table.setItem(row, 2, QTableWidgetItem(entry.get("timestamp", "")[:10]))
                self.playlist_table.setItem(row, 3, QTableWidgetItem(str(entry.get("seed", ""))))
                self.playlist_table.setItem(row, 4, QTableWidgetItem(entry.get("model", "").split('/')[-1]))
        finally:
            self.playlist_table.setUpdatesEnabled(True)

    def clear_history(self):
        if QMessageBox.question(self, "Clear History", "Remove all history entries?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.playlist_table.setRowCount(0)
            self.settings.remove("history")
            self.settings.sync()

    def load_settings(self):
        self.prompt_input.setPlainText(self.settings.value("prompt", "Musical constellations twinkling in the night sky", type=str))
        self.neg_prompt_input.setText(self.settings.value("negative_prompt", "", type=str))
        self.model_combo.setCurrentText(self.settings.value("model", "cvssp/audioldm2", type=str))
        self.duration_spin.setValue(float(self.settings.value("duration", 10.0)))
        self.chunk_size_spin.setValue(float(self.settings.value("chunk_size", 10.0)))
        self.steps_spin.setValue(int(self.settings.value("steps", 200)))
        self.guidance_spin.setValue(float(self.settings.value("guidance", 3.5)))
        self.seed_spin.setValue(int(self.settings.value("seed", -1)))
        self.variations_spin.setValue(int(self.settings.value("variations", 1)))
        
        default_device = "cpu"
        if torch.cuda.is_available(): default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): default_device = "mps"
        self.device_combo.setCurrentText(self.settings.value("device", default_device, type=str))
        
        self.cache_dir_edit.setText(self.settings.value("cache_dir", DEFAULT_CACHE_DIR, type=str))
        self.output_dir_edit.setText(self.settings.value("output_dir", os.path.join(os.getcwd(), "generated_audio"), type=str))
        self.section_edit.setText(self.settings.value("section", "", type=str))
        
        for model in self.settings.value("custom_models", []):
            if model not in DEFAULT_MODELS and self.model_combo.findText(model) == -1:
                self.model_combo.addItem(model)
        
        self.normalize_check.setChecked(self.settings.value("normalize", True, type=bool))
        self.trim_check.setChecked(self.settings.value("trim", True, type=bool))
        self.fade_in_check.setChecked(self.settings.value("fade_in", True, type=bool))
        self.fade_out_check.setChecked(self.settings.value("fade_out", True, type=bool))
        self.cpu_offload_check.setChecked(self.settings.value("cpu_offload", False, type=bool))
        
        self.batch_variations_spin.setValue(int(self.settings.value("batch_variations", 1)))
        self.batch_delay_spin.setValue(float(self.settings.value("batch_delay", 1.0)))
        self.batch_continue_on_error_check.setChecked(self.settings.value("batch_continue_on_error", True, type=bool))
        self.vol_slider.setValue(int(self.settings.value("volume", 50)))
        
        self.mode_tabs.setCurrentIndex(int(self.settings.value("left_tab", 0)))
        self.right_tabs.setCurrentIndex(int(self.settings.value("right_tab", 0)))
        self.prompt_input.history = self.settings.value("prompt_history", [])
            
        self.on_device_changed(self.device_combo.currentText())
        self.update_chunk_preview()
        self.load_history()
        
        last_audio = self.settings.value("last_audio_path", "", type=str)
        if last_audio and os.path.exists(last_audio):
            self._load_audio_file(last_audio)

    def auto_save_state(self):
        self.settings.setValue("prompt", self.prompt_input.toPlainText())
        self.settings.setValue("negative_prompt", self.neg_prompt_input.text())
        self.settings.setValue("steps", self.steps_spin.value())
        self.settings.setValue("guidance", self.guidance_spin.value())
        self.settings.setValue("seed", self.seed_spin.value())
        self.settings.setValue("duration", self.duration_spin.value())
        self.settings.setValue("chunk_size", self.chunk_size_spin.value())
        self.settings.setValue("variations", self.variations_spin.value())
        self.settings.sync()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.isMinimized() and hasattr(self, 'gpu_mem_timer'):
                self.gpu_mem_timer.stop()
            elif hasattr(self, 'gpu_mem_timer'):
                self.gpu_mem_timer.start(GPU_MEM_POLL_MS)
        super().changeEvent(event)

    def closeEvent(self, event):
        # JsonSettings automatically handles QByteArray -> Base64 conversion
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("left_tab", self.mode_tabs.currentIndex())
        self.settings.setValue("right_tab", self.right_tabs.currentIndex())
        self.settings.setValue("prompt", self.prompt_input.toPlainText())
        self.settings.setValue("negative_prompt", self.neg_prompt_input.text())
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("duration", self.duration_spin.value())
        self.settings.setValue("chunk_size", self.chunk_size_spin.value())
        self.settings.setValue("steps", self.steps_spin.value())
        self.settings.setValue("guidance", self.guidance_spin.value())
        self.settings.setValue("seed", self.seed_spin.value())
        self.settings.setValue("variations", self.variations_spin.value())
        self.settings.setValue("device", self.device_combo.currentText())
        self.settings.setValue("cache_dir", self.cache_dir_edit.text().strip())
        self.settings.setValue("output_dir", self.output_dir_edit.text().strip())
        self.settings.setValue("section", self.section_edit.text().strip())
        self.settings.setValue("normalize", self.normalize_check.isChecked())
        self.settings.setValue("trim", self.trim_check.isChecked())
        self.settings.setValue("fade_in", self.fade_in_check.isChecked())
        self.settings.setValue("fade_out", self.fade_out_check.isChecked())
        self.settings.setValue("cpu_offload", self.cpu_offload_check.isChecked())
        self.settings.setValue("batch_variations", self.batch_variations_spin.value())
        self.settings.setValue("batch_delay", self.batch_delay_spin.value())
        self.settings.setValue("batch_continue_on_error", self.batch_continue_on_error_check.isChecked())
        self.settings.setValue("volume", self.vol_slider.value())
        self.settings.setValue("prompt_history", self.prompt_input.history[-MAX_PROMPT_HISTORY:])
        
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            self.settings.setValue("last_audio_path", self.current_audio_path)
        else:
            self.settings.remove("last_audio_path")
            
        try: self.media_player.stop()
        except Exception: pass

        if self.worker:
            self.worker.is_cancelled = True
            
        if self.worker_thread and self.worker_thread.isRunning():
            if not self.worker_thread.wait(5000):
                logger.warning("Worker thread did not finish in time. Forcing exit.")
        
        AudioGenerationWorker.unload_pipeline()
        self.settings.sync()
        event.accept()


# --- Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(SETTINGS_ORG)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
    app.setFont(QFont("Segoe UI", 9))

    window = AudioLDM2Studio()
    window.show()
    sys.exit(app.exec())