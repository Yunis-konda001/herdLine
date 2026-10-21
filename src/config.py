"""Project paths and default settings for detection and counting.

Change these values if you move folders or want a different line or score cut-off.
"""
from pathlib import Path

# Main folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
RESULTS_DIR = OUTPUT_DIR / "results"

# Model files
DEFAULT_DATASET_YAML = DATA_DIR / "goat.yaml"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"  # fine-tuned goat model
FALLBACK_WEIGHTS = "yolov8n.pt"  # used only if best.pt is missing

# Default counting line: vertical, middle of the frame, goats walk right to left
DEFAULT_LINE_MODE = "vertical"
DEFAULT_LINE_X = 0.5
DEFAULT_LINE_Y = 0.5
DEFAULT_CROSSING_DIRECTION = "right_to_left"

# Detection cut-offs used at run time
CONFIDENCE_THRESHOLD = 0.15  # keep weak goat boxes so tracks do not die early
IOU_THRESHOLD = 0.30
DETECT_IMGSZ = 960  # larger than training size; helps on gate video

# Goat box filters
GOAT_CLASS_ID = 0
GOAT_MIN_CONFIDENCE = 0.15
GOAT_MAX_HEIGHT_WIDTH = 2.6  # very tall boxes often look like a person
MIN_BOX_PX = 28  # ignore tiny noise boxes

# Background person filter (COCO person class)
PERSON_FILTER_ENABLED = True
PERSON_DETECT_CONF = 0.2
PERSON_BOX_PAD = 0.18
PERSON_IOU_REJECT = 0.06

# ByteTrack config file
TRACKER = PROJECT_ROOT / "config" / "bytetrack.yaml"
