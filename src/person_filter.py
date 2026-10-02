"""Suppress goat detections that overlap COCO 'person' boxes (background false positives)."""
from __future__ import annotations

from ultralytics import YOLO

from src.config import (
    DETECT_IMGSZ,
    FALLBACK_WEIGHTS,
    PERSON_BOX_PAD,
    PERSON_DETECT_CONF,
    PERSON_IOU_REJECT,
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


def _expand_box(box: list[float], pad_frac: float) -> list[float]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def _centroid_in(inner: list[float], outer: list[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


class PersonFilter:
    """COCO person detector to reject human false 'goats'."""

    def __init__(self) -> None:
        self._model: YOLO | None = None

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(str(FALLBACK_WEIGHTS))
        return self._model

    def person_boxes(self, frame_bgr) -> list[list[float]]:
        model = self._ensure_model()
        out = model.predict(
            frame_bgr,
            classes=[0],
            conf=PERSON_DETECT_CONF,
            imgsz=DETECT_IMGSZ,
            verbose=False,
        )[0]
        if out.boxes is None or len(out.boxes) == 0:
            return []
        return out.boxes.xyxy.cpu().tolist()

    def overlaps_person(self, goat_xyxy: list[float], person_boxes: list[list[float]]) -> bool:
        for raw in person_boxes:
            pbox = _expand_box(raw, PERSON_BOX_PAD)
            if _box_iou(goat_xyxy, pbox) >= PERSON_IOU_REJECT:
                return True
            if _centroid_in(goat_xyxy, pbox):
                return True
            if _centroid_in(raw, goat_xyxy):
                return True
        return False
