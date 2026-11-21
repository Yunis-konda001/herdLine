"""Read frames from a video file and create writers for annotated output."""
from __future__ import annotations

from pathlib import Path

import cv2


class VideoProcessor:
    """Open a video with OpenCV and yield frames one by one."""

    def __init__(self, source: Path | str) -> None:
        self.source = Path(source)
        # Initialize OpenCV video capture using the provided file path.
        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.source}")

    @property
    def width(self) -> int:
        # Retrieve the video frame width in pixels.
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        # Retrieve the video frame height in pixels.
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        # Retrieve the video frame rate.
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Some clips report 0 fps; use 25 so writing still works.
        return fps if fps and fps > 0 else 25.0

    def frames(self):
        """Yield each BGR frame until the video ends."""
        while True:
            # Read the next frame from the video stream.
            ok, frame = self.cap.read()
            if not ok:
                break
            # Provide the frame to the caller (e.g., the YOLO pipeline).
            yield frame

    def release(self) -> None:
        # Close the video capture and release system resources.
        self.cap.release()

    @staticmethod
    def create_writer(output: Path, width: int, height: int, fps: float):
        """Create an MP4 writer for the annotated video."""
        # Ensure the output directory exists before saving.
        output.parent.mkdir(parents=True, exist_ok=True)
        # Define the codec for the output MP4 video.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Initialize the writer with the correct dimensions and frame rate.
        writer = cv2.VideoWriter(
            str(output), fourcc, float(fps), (int(width), int(height))
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {output}")
        return writer