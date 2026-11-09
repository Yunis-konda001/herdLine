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
# Fixed gate convention (not fitted per video to herd size).
# "either" counts L→R and R→L (walk direction does not zero the count).
# Users can still lock one way in the form advanced settings.
DEFAULT_CROSSING_DIRECTION = "either"


# On Render, set HERDLINE_IMGSZ=640 (matches exported ONNX) to save RAM.
# High conf: background/people false boxes are usually weaker than real goats.
CONFIDENCE_THRESHOLD = float(os.getenv("HERDLINE_CONF", "0.40"))
IOU_THRESHOLD = float(os.getenv("HERDLINE_IOU", "0.70"))
DETECT_IMGSZ = int(os.getenv("HERDLINE_IMGSZ", "640"))

GOAT_CLASS_ID = 0
GOAT_MIN_CONFIDENCE = float(os.getenv("HERDLINE_CONF", "0.40"))
# Ignore upright people-shaped boxes
GOAT_MAX_HEIGHT_WIDTH = float(os.getenv("HERDLINE_MAX_HW", "1.45"))
GOAT_MIN_HEIGHT_WIDTH = 0.55
MAX_BOX_AREA_FRAC = 0.45

# Only trust a non-zero count when we see many strong goat-shaped hits
GOAT_STRONG_CONFIDENCE = float(os.getenv("HERDLINE_STRONG_CONF", "0.55"))
MIN_STRONG_GOAT_HITS = int(os.getenv("HERDLINE_MIN_STRONG_HITS", "20"))
# Aspect band treated as goat-like; taller → person-like (never counted)
GOAT_LIKE_MAX_HW = float(os.getenv("HERDLINE_GOAT_LIKE_HW", "1.35"))
PERSON_LIKE_MIN_HW = float(os.getenv("HERDLINE_PERSON_LIKE_HW", "1.40"))

# Need a few sightings before a track can cross-count (drops flicker FPs).
# 2 keeps short gate clips from dropping valid crossings.
MIN_TRACK_HITS = int(os.getenv("HERDLINE_MIN_TRACK_HITS", "2"))

# Base detect stride. Sparse clips prefer 2 (avoids ID over-merge).
# Dense herds auto-use 1 (see CountingPipeline).
FRAME_STRIDE = max(1, int(os.getenv("HERDLINE_FRAME_STRIDE", "2")))
# When declared herd size is at least this, force stride=1 for recall.
DENSE_HERD_STRIDE_THRESHOLD = int(os.getenv("HERDLINE_DENSE_HERD", "8"))

# Line search is OFF by default: count with a fixed gate line, then compare to herd size.
# (Do not fit line/direction to the declared herd size — that inflates accuracy.)
# Set HERDLINE_LINE_SEARCH=fast|full only for experiments.
LINE_SEARCH_MODE = os.getenv("HERDLINE_LINE_SEARCH", "off").strip().lower()

# Smaller goats in wide shots — was 32 and dropped many distant boxes
MIN_BOX_PX = int(os.getenv("HERDLINE_MIN_BOX_PX", "20"))

# Re-encode annotated mp4 for browsers (ffmpeg). Set 0 to skip and save time.
REENCODE_RESULT = os.getenv("HERDLINE_REENCODE", "1").strip() not in {
    "0",
    "false",
    "False",
    "no",
}

TRACKER = PROJECT_ROOT / "config" / "bytetrack.yaml"
