"""Paths and default settings for detection and counting."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
RESULTS_DIR = OUTPUT_DIR / "results"

DEFAULT_DATASET_YAML = DATA_DIR / "goat.yaml"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

# Default gate line (vertical, centre). Direction can be changed on upload.
DEFAULT_LINE_MODE = "vertical"
DEFAULT_LINE_X = 0.5
DEFAULT_LINE_Y = 0.5
DEFAULT_CROSSING_DIRECTION = "right_to_left"

# 640 keeps counting fast; filters still drop weak background boxes
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DETECT_IMGSZ = 640

GOAT_CLASS_ID = 0
GOAT_MIN_CONFIDENCE = 0.28
GOAT_MAX_HEIGHT_WIDTH = 2.3
GOAT_MIN_HEIGHT_WIDTH = 0.50
MIN_BOX_PX = 40
MAX_BOX_AREA_FRAC = 0.40

TRACKER = PROJECT_ROOT / "config" / "bytetrack.yaml"
