"""Keep only plausible goat detections for counting and overlay."""
from __future__ import annotations

from src.config import (
    GOAT_CLASS_ID,
    GOAT_MAX_HEIGHT_WIDTH,
    GOAT_MIN_CONFIDENCE,
    MIN_BOX_PX,
)
from src.person_filter import PersonFilter


def _looks_like_standing_person(box: list[float], conf: float, frame_shape) -> bool:
    """Heuristic when COCO person miss (small/distant human misclassified as goat)."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    aspect = bh / bw
    fh = frame_shape[0]
    cy = (y1 + y2) / 2
    if aspect >= 1.45 and conf < 0.42:
        return True
    if aspect >= 1.35 and conf < 0.32 and cy < fh * 0.72:
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
    return keep
