"""Detect non-goat livestock (cattle) so cow/calf clips are not counted as goats.

Uses a COCO pretrained YOLOv8n for class ``cow`` only. Goats are sometimes
mis-labeled as cow by COCO, so we reject only when cattle-like density is
persistently high across sampled frames — not on a single false box.

- The goat model might mistake a cow for a goat if the shape is similar.
- To avoid false counts, this file runs a second, lightweight YOLO model 
  that specifically looks for cows (COCO class 19).
- It only rejects the video if cows appear consistently across many frames, 
  preventing a single false box from ruining a valid goat count.
"""
from __future__ import annotations

import os

import numpy as np

# COCO class id for cow 
_COCO_COW = 19
_MODEL = None
_UNAVAILABLE = False


def _coco_conf() -> float:
    # Get the confidence threshold for the cattle detector from environment variables.
    return float(os.getenv("HERDLINE_LIVESTOCK_CONF", "0.45"))


def count_cattle_rough(frame: np.ndarray) -> int:
    """Return number of COCO ``cow`` boxes in one frame (0 if gate unavailable)."""
    global _MODEL, _UNAVAILABLE
    
    # If the model is unavailable or the frame is empty, return 0.
    if _UNAVAILABLE or frame is None or getattr(frame, "size", 0) == 0:
        return 0
    
    try:
        # Load the default YOLOv8n model (trained on COCO) only once.
        if _MODEL is None:
            from ultralytics import YOLO

            _MODEL = YOLO("yolov8n.pt")
    except Exception:
        # If Ultralytics/PyTorch is not installed (e.g., on Render), skip this gate.
        _UNAVAILABLE = True
        return 0

    try:
        # Run the model but ONLY look for cows (class 19).
        # Use a smaller image size (320) to save memory and speed up processing.
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
    
    # Return the number of cows detected in this frame.
    return int(len(result.boxes))


def looks_like_cattle_clip(
    cattle_hits: int,
    samples: int,
    positive_frames: int,
    max_in_frame: int,
) -> bool:
    """True when sampled frames show persistent cattle density (not goat noise)."""
    # If we have too few samples or no cattle hits, it's not a cattle clip.
    if samples < 4 or cattle_hits <= 0:
        return False
    
    # Calculate the average number of cows per frame.
    mean = cattle_hits / float(samples)
    # Calculate the fraction of frames where at least one cow was seen.
    frac = positive_frames / float(samples)
    
    # Rule 1: Clear cattle herd in frame
    if mean >= 3.0 or max_in_frame >= 6:
        return True
    
    # Rule 2: Persistent cattle-like detections across most of the clip.
    # Hit floor stays modest so short calf clips still reject under sparse sampling.
    if mean >= 1.8 and frac >= 0.85 and cattle_hits >= 8:
        return True
    
    return False