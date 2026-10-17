"""Keep only plausible goat detections for counting and overlay."""
from __future__ import annotations

from src.config import (
    GOAT_CLASS_ID,
    GOAT_MAX_HEIGHT_WIDTH,
    GOAT_MIN_CONFIDENCE,
    MIN_BOX_PX,
)
from src.person_filter import PersonFilter, _box_iou, _centroid_in


def _looks_like_standing_person(box: list[float], conf: float, frame_shape) -> bool:
    """Heuristic when COCO person miss (small/distant human misclassified as goat)."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    aspect = bh / bw
    # Only very tall, low-score boxes. A goat facing the camera is not this thin.
    if aspect >= 2.0 and conf < 0.35:
        return True
    return False


def select_goat_indices(
    frame_bgr,
    xyxy: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
    person_filter: PersonFilter,
) -> list[int]:
    person_boxes = person_filter.person_boxes(frame_bgr)
    keep: list[int] = []
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
        if bh / max(bw, 1e-6) > GOAT_MAX_HEIGHT_WIDTH:
            continue
        if _looks_like_standing_person(box, conf, frame_bgr.shape):
            continue
        if person_filter.overlaps_person(box, person_boxes):
            continue
        keep.append(i)
    return _drop_overlapping(keep, xyxy, confidences)


def _same_animal(a: list[float], b: list[float]) -> bool:
    """A head box inside a body box is still one goat."""
    if _box_iou(a, b) >= 0.30:
        return True
    return _centroid_in(a, b) or _centroid_in(b, a)


def _drop_overlapping(indices, xyxy, confidences):
    """One goat was getting two boxes. Keep the stronger box."""
    ordered = sorted(indices, key=lambda i: float(confidences[i]), reverse=True)
    kept: list[int] = []
    for i in ordered:
        if any(_same_animal(xyxy[i], xyxy[j]) for j in kept):
            continue
        kept.append(i)
    return kept
