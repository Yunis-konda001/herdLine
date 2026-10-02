"""OpenCV video read/write helpers."""
from __future__ import annotations

from pathlib import Path

import cv2


class VideoProcessor:
    def __init__(self, source: Path | str) -> None:
        self.source = Path(source)
        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.source}")

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()

    @staticmethod
    def create_writer(output: Path, width: int, height: int, fps: float):
        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output), fourcc, float(fps), (int(width), int(height))
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {output}")
        return writer
