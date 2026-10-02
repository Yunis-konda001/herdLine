"""Project paths and default counting settings."""
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

# Default: side-on road video (goats cross left–right)
DEFAULT_LINE_MODE = "vertical"
DEFAULT_LINE_X = 0.5
DEFAULT_LINE_Y = 0.5
DEFAULT_CROSSING_DIRECTION = "right_to_left"

CONFIDENCE_THRESHOLD = 0.28
IOU_THRESHOLD = 0.45
DETECT_IMGSZ = 960

GOAT_CLASS_ID = 0
GOAT_MIN_CONFIDENCE = 0.28
GOAT_MAX_HEIGHT_WIDTH = 1.75
MIN_BOX_PX = 28
PERSON_FILTER_ENABLED = True
PERSON_DETECT_CONF = 0.2
PERSON_BOX_PAD = 0.18
PERSON_IOU_REJECT = 0.06
SPATIAL_PSEUDO_ID_START = 10_000
TRACKER = PROJECT_ROOT / "config" / "bytetrack.yaml"
