"""Load goat detector — prefer ONNX Runtime, fall back to Ultralytics/PyTorch."""
from __future__ import annotations

from pathlib import Path

from src.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_ONNX_WEIGHTS,
    DEFAULT_WEIGHTS,
    DETECT_IMGSZ,
    FALLBACK_WEIGHTS,
    GOAT_CLASS_ID,
    IOU_THRESHOLD,
    TRACKER,
)


def _resolve_weights(weights: Path | str | None) -> Path | str:
    if weights is not None:
        path = Path(weights) if not isinstance(weights, str) or "://" not in weights else weights
        if isinstance(path, Path):
            if path.suffix.lower() == ".onnx" and path.is_file():
                return path
            if path.suffix.lower() == ".pt" and path.is_file():
                onnx = path.with_suffix(".onnx")
                if onnx.is_file():
                    return onnx
                return path
            if path.is_file():
                return path
        return weights
    if DEFAULT_ONNX_WEIGHTS.is_file():
        return DEFAULT_ONNX_WEIGHTS
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    return FALLBACK_WEIGHTS


class GoatDetector:
    """Unified detector used by CountingPipeline."""

    def __init__(self, weights: Path | str | None = None) -> None:
        resolved = _resolve_weights(weights)
        self.weights = resolved
        self._onnx = None
        self._yolo = None
        path = Path(str(resolved))
        if path.suffix.lower() == ".onnx":
            from src.onnx_goat_detector import OnnxGoatDetector

            self._onnx = OnnxGoatDetector(path)
        else:
            from ultralytics import YOLO

            self._yolo = YOLO(str(resolved))

    @property
    def backend(self) -> str:
        return "onnx" if self._onnx is not None else "torch"

    def reset_tracks(self) -> None:
        if self._onnx is not None:
            self._onnx.reset_tracks()
        elif self._yolo is not None:
            self._yolo.predictor = None

    def detect_boxes(self, frame) -> list[tuple[float, float, float, float, float, int]]:
        """Return (x1, y1, x2, y2, conf, track_id) with track_id=-1 for detect-only."""
        if self._onnx is not None:
            return self._onnx.detect_boxes(frame)

        result = self._yolo.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            classes=[GOAT_CLASS_ID],
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().tolist()
        confs = (
            result.boxes.conf.cpu().tolist()
            if result.boxes.conf is not None
            else [1.0] * len(xyxy)
        )
        clss = (
            result.boxes.cls.int().cpu().tolist()
            if result.boxes.cls is not None
            else [0] * len(xyxy)
        )
        out: list[tuple[float, float, float, float, float, int]] = []
        for box, conf, cls in zip(xyxy, confs, clss):
            if int(cls) != GOAT_CLASS_ID:
                continue
            x1, y1, x2, y2 = box
            out.append((float(x1), float(y1), float(x2), float(y2), float(conf), -1))
        return out

    def predict_frame(self, frame):
        """Legacy Ultralytics result API — only for torch backend."""
        if self._onnx is not None:
            raise RuntimeError("predict_frame is not available on the ONNX backend")
        return self._yolo.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            classes=[GOAT_CLASS_ID],
            verbose=False,
        )[0]

    def track_frame(self, frame, persist: bool = True):
        if self._onnx is not None:
            raise RuntimeError("track_frame is not available on the ONNX backend")
        return self._yolo.track(
            frame,
            persist=persist,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=DETECT_IMGSZ,
            classes=[GOAT_CLASS_ID],
            tracker=str(TRACKER),
            verbose=False,
        )[0]
