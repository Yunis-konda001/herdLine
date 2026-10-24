"""Keep clear goat boxes for counting.

Drop weak / tiny / huge / flat boxes, and keep one box when two overlap on one goat.
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

    for i, box in enumerate(xyxy):
        if int(class_ids[i]) != GOAT_CLASS_ID:
            continue
        conf = float(confidences[i])
        if conf < GOAT_MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        ratio = bh / max(bw, 1e-6)
        if ratio > GOAT_MAX_HEIGHT_WIDTH or ratio < GOAT_MIN_HEIGHT_WIDTH:
            continue
        if frame_area is not None and (bw * bh) / frame_area > MAX_BOX_AREA_FRAC:
            continue
        keep.append(i)
    return _drop_overlapping(keep, xyxy, confidences)


def _same_animal(a: list[float], b: list[float]) -> bool:
    # Head box inside body box is still one goat
    if _box_iou(a, b) >= 0.25:
        return True
    return _centroid_in(a, b) or _centroid_in(b, a)


def _drop_overlapping(indices, xyxy, confidences):
    # Keep the stronger box when two boxes cover the same goat
    ordered = sorted(indices, key=lambda i: float(confidences[i]), reverse=True)
    kept: list[int] = []
    for i in ordered:
        if any(_same_animal(xyxy[i], xyxy[j]) for j in kept):
            continue
        kept.append(i)
    return kept
