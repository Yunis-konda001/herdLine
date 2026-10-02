"""YOLO-based goat detection and tracking wrapper."""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.config import (
    CONFIDENCE_THRESHOLD,
    DETECT_IMGSZ,
    FALLBACK_WEIGHTS,
    IOU_THRESHOLD,
    TRACKER,
)


class GoatDetector:
    def __init__(self, weights: Path | str | None = None) -> None:
        path = weights if weights else FALLBACK_WEIGHTS
        if isinstance(path, Path) and not path.is_file():
            path = FALLBACK_WEIGHTS
        self.model = YOLO(str(path))

    def track_frame(self, frame, persist: bool = True):
        """Run detection + tracking on one BGR frame."""
        return self.model.track(
            frame,
            persist=persist,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            tracker=str(TRACKER),
            verbose=False,
        )[0]
