"""Detect non-goat livestock (cattle) so cow/calf clips are not counted as goats.

Uses a COCO pretrained YOLOv8n for class ``cow`` only. Goats are sometimes
mis-labeled as cow by COCO, so we reject only when cattle-like density is
persistently high across sampled frames — not on a single false box.

No-ops when Ultralytics/torch is unavailable (e.g. slim Render image).
"""
from __future__ import annotations

import os

import numpy as np

# COCO class id for cow (not sheep — sheep often fires on real goats)
_COCO_COW = 19
_MODEL = None
_UNAVAILABLE = False


def _coco_conf() -> float:
    return float(os.getenv("HERDLINE_LIVESTOCK_CONF", "0.45"))


def count_cattle_rough(frame: np.ndarray) -> int:
    """Return number of COCO ``cow`` boxes in one frame (0 if gate unavailable)."""
    global _MODEL, _UNAVAILABLE
    if _UNAVAILABLE or frame is None or getattr(frame, "size", 0) == 0:
        return 0
    try:
        if _MODEL is None:
            from ultralytics import YOLO

            _MODEL = YOLO("yolov8n.pt")
    except Exception:
        _UNAVAILABLE = True
        return 0

    try:
        result = _MODEL.predict(
            frame,
            classes=[_COCO_COW],
            conf=_coco_conf(),
            imgsz=320,
            verbose=False,
        )[0]
    except Exception:
        return 0
    if result.boxes is None:
        return 0
    return int(len(result.boxes))


def looks_like_cattle_clip(
    cattle_hits: int,
    samples: int,
    positive_frames: int,
    max_in_frame: int,
) -> bool:
    """True when sampled frames show persistent cattle density (not goat noise)."""
    if samples < 4 or cattle_hits <= 0:
        return False
    mean = cattle_hits / float(samples)
    frac = positive_frames / float(samples)
    # Clear cattle herd in frame
    if mean >= 3.0 or max_in_frame >= 6:
        return True
    # Persistent cattle-like detections across most of the clip.
    # Hit floor stays modest so short calf clips still reject under sparse sampling.
    if mean >= 1.8 and frac >= 0.85 and cattle_hits >= 8:
        return True
    return False
