"""YOLOv8 goat detection with ONNX Runtime (no PyTorch at inference)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.config import (
    CONFIDENCE_THRESHOLD,
    DETECT_IMGSZ,
    GOAT_CLASS_ID,
    IOU_THRESHOLD,
)


def _letterbox(
    image: np.ndarray, new_size: int
) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (float(left), float(top))


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


class OnnxGoatDetector:
    """Load best.onnx and return goat boxes for one BGR frame."""

    def __init__(
        self,
        weights: Path | str,
        imgsz: int | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX weights not found: {path}")
        self.imgsz = int(imgsz or DETECT_IMGSZ)
        self.conf = float(conf if conf is not None else CONFIDENCE_THRESHOLD)
        self.iou = float(iou if iou is not None else IOU_THRESHOLD)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        # Exported model may be fixed at 640 even if DETECT_IMGSZ differs
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            self.imgsz = int(shape[2])

    def reset_tracks(self) -> None:
        return

    def detect_boxes(
        self, frame: np.ndarray
    ) -> list[tuple[float, float, float, float, float, int]]:
        """Return (x1, y1, x2, y2, conf, class_id) in original frame pixels."""
        if frame is None or frame.size == 0:
            return []
        orig_h, orig_w = frame.shape[:2]
        letter, scale, (pad_x, pad_y) = _letterbox(frame, self.imgsz)
        blob = letter[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None])

        outputs = self.session.run(None, {self.input_name: blob})
        pred = np.asarray(outputs[0])
        if pred.ndim == 3:
            pred = pred[0]
        # Ultralytics export: (4+nc, N) — here (5, 8400) for one class
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_xywh = pred[:, :4]
        if pred.shape[1] == 5:
            scores = pred[:, 4]
            classes = np.zeros(len(scores), dtype=np.int32)
        else:
            class_scores = pred[:, 4:]
            classes = class_scores.argmax(axis=1).astype(np.int32)
            scores = class_scores.max(axis=1)

        mask = scores >= self.conf
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        classes = classes[mask]
        if len(scores) == 0:
            return []

        # xywh centre format → xyxy on letterboxed image
        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms_xyxy(boxes, scores, self.iou)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        # Undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)

        out: list[tuple[float, float, float, float, float, int]] = []
        for (bx1, by1, bx2, by2), score, cls in zip(boxes, scores, classes):
            if int(cls) != GOAT_CLASS_ID:
                continue
            out.append(
                (float(bx1), float(by1), float(bx2), float(by2), float(score), -1)
            )
        return out
