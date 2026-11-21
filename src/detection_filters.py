"""Keep clear goat boxes for counting.

Drop weak / tiny / huge / flat boxes, and keep one box when two overlap on one goat.

- YOLO sometimes outputs messy boxes: boxes that are too small, too large, 
  too flat (like a shadow), or duplicate boxes on the same goat.
- If we pass all these boxes to the tracker, it will get confused and the count 
  will be wrong.
- This file acts as a "quality control" gate, removing bad boxes before they 
  reach the tracking and counting logic.
"""
from __future__ import annotations

from src.config import (
    GOAT_CLASS_ID,
    GOAT_MAX_HEIGHT_WIDTH,
    GOAT_MIN_CONFIDENCE,
    GOAT_MIN_HEIGHT_WIDTH,
    MAX_BOX_AREA_FRAC,
    MIN_BOX_PX,
)


def _box_iou(a: list[float], b: list[float]) -> float:
    # Calculate the Intersection over Union (IoU) between two boxes.
    # IoU tells us how much two boxes overlap.
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _centroid_in(inner: list[float], outer: list[float]) -> bool:
    # Check if the center point of one box is inside another box.
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def select_goat_indices(
    xyxy: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
    frame_w: float | None = None,
    frame_h: float | None = None,
) -> list[int]:
    """Return indexes of boxes that should be used for tracking and counting."""
    keep: list[int] = []
    frame_area = None
    if frame_w and frame_h and frame_w > 0 and frame_h > 0:
        frame_area = float(frame_w) * float(frame_h)

    # Loop through every box YOLO produced and apply the quality filters.
    for i, box in enumerate(xyxy):
        # Rule 1: Must be the goat class.
        if int(class_ids[i]) != GOAT_CLASS_ID:
            continue
        
        # Rule 2: Must have high enough confidence.
        conf = float(confidences[i])
        if conf < GOAT_MIN_CONFIDENCE:
            continue
        
        # Rule 3: Must not be too small.
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        
        # Rule 4: Must have a goat-like height-to-width ratio.
        # This filters out people (too tall) and shadows (too flat).
        ratio = bh / max(bw, 1e-6)
        if ratio > GOAT_MAX_HEIGHT_WIDTH or ratio < GOAT_MIN_HEIGHT_WIDTH:
            continue
        
        # Rule 5: Must not be too large (e.g., the whole frame).
        if frame_area is not None and (bw * bh) / frame_area > MAX_BOX_AREA_FRAC:
            continue
        
        keep.append(i)
    
    # Finally, remove duplicate boxes that overlap on the same goat.
    return _drop_overlapping(keep, xyxy, confidences)


def _same_animal(a: list[float], b: list[float]) -> bool:
    # Head box inside body box is still one goat and should be merged
    if _box_iou(a, b) >= 0.25:
        return True
    return _centroid_in(a, b) or _centroid_in(b, a)


def _drop_overlapping(indices, xyxy, confidences):
    # Keep the stronger box when two boxes cover the same goat.
    ordered = sorted(indices, key=lambda i: float(confidences[i]), reverse=True)
    kept: list[int] = []
    for i in ordered:
        if any(_same_animal(xyxy[i], xyxy[j]) for j in kept):
            continue
        kept.append(i)
    return kept