"""Load YOLO and run detection plus tracking on one video frame."""
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
    """Thin wrapper around Ultralytics YOLO.track."""

    def __init__(self, weights: Path | str | None = None) -> None:
        # Prefer the fine-tuned goat weights. Fall back to stock YOLO if missing.
        path = weights if weights else FALLBACK_WEIGHTS
        if isinstance(path, Path) and not path.is_file():
            path = FALLBACK_WEIGHTS
        self.model = YOLO(str(path))

    def track_frame(self, frame, persist: bool = True):
        """Detect goats and keep temporary track IDs across frames."""
        return self.model.track(
            frame,
            persist=persist,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            tracker=str(TRACKER),
            verbose=False,
        )[0]
