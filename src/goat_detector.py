"""Load YOLO and run detect + track on one frame."""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.config import (
    CONFIDENCE_THRESHOLD,
    DETECT_IMGSZ,
    FALLBACK_WEIGHTS,
    GOAT_CLASS_ID,
    IOU_THRESHOLD,
    TRACKER,
)


class GoatDetector:
    def __init__(self, weights: Path | str | None = None) -> None:
        path = weights if weights else FALLBACK_WEIGHTS
        if isinstance(path, Path) and not path.is_file():
            path = FALLBACK_WEIGHTS
        self.model = YOLO(str(path))

    def reset_tracks(self) -> None:
        self.model.predictor = None

    def predict_frame(self, frame):
        """Per-frame detect only — denser boxes than ByteTrack on crowded herds."""
        return self.model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            classes=[GOAT_CLASS_ID],
            verbose=False,
        )[0]

    def track_frame(self, frame, persist: bool = True):
        # Only the goat class; high conf cuts empty background boxes
        return self.model.track(
            frame,
            persist=persist,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            classes=[GOAT_CLASS_ID],
            tracker=str(TRACKER),
            verbose=False,
        )[0]
