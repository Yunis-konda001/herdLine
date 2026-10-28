"""Paths and default settings for detection and counting."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
RESULTS_DIR = OUTPUT_DIR / "results"

DEFAULT_DATASET_YAML = DATA_DIR / "goat.yaml"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"
DEFAULT_ONNX_WEIGHTS = MODELS_DIR / "best.onnx"
FALLBACK_WEIGHTS = "yolov8n.pt"

# Virtual gate line — a goat is counted when its centre crosses this line
DEFAULT_LINE_MODE = "vertical"
DEFAULT_LINE_X = 0.5
DEFAULT_LINE_Y = 0.5
DEFAULT_CROSSING_DIRECTION = "right_to_left"

# Recall-first so crowded clips stay close to the hand count.
# On Render, set HERDLINE_IMGSZ=640 (matches exported ONNX) to save RAM.
CONFIDENCE_THRESHOLD = float(os.getenv("HERDLINE_CONF", "0.18"))
IOU_THRESHOLD = float(os.getenv("HERDLINE_IOU", "0.70"))
DETECT_IMGSZ = int(os.getenv("HERDLINE_IMGSZ", "640"))

GOAT_CLASS_ID = 0
GOAT_MIN_CONFIDENCE = float(os.getenv("HERDLINE_CONF", "0.18"))
GOAT_MAX_HEIGHT_WIDTH = 2.4
GOAT_MIN_HEIGHT_WIDTH = 0.45
MIN_BOX_PX = 32
MAX_BOX_AREA_FRAC = 0.45

MIN_TRACK_HITS = 1

TRACKER = PROJECT_ROOT / "config" / "bytetrack.yaml"
