"""End-to-end video counting: detect, track, line count, optional annotated output."""
from __future__ import annotations

from pathlib import Path

import cv2
import torch

from src.config import (
    DEFAULT_CROSSING_DIRECTION,
    DEFAULT_LINE_MODE,
    DEFAULT_LINE_X,
    DEFAULT_LINE_Y,
    GOAT_CLASS_ID,
    PERSON_FILTER_ENABLED,
)
from src.detection_filters import select_goat_indices
from src.person_filter import PersonFilter
from src.counting_result import CountingResult
from src.goat_detector import GoatDetector
from src.line_counter import LineCounter
from src.video_encode import make_browser_playable
from src.spatial_track import SpatialTrackMatcher
from src.video_processor import VideoProcessor


class CountingPipeline:
    def __init__(
        self,
        weights: Path | str | None = None,
        line_mode: str = DEFAULT_LINE_MODE,
        line_y: float = DEFAULT_LINE_Y,
        line_x: float = DEFAULT_LINE_X,
        direction: str = DEFAULT_CROSSING_DIRECTION,
    ) -> None:
        self.detector = GoatDetector(weights)
        orientation = "vertical" if line_mode == "vertical" else "horizontal"
        self.orientation = orientation
        self.line_y = line_y
        self.line_x = line_x
        self.direction = direction

    def run(
        self,
        video_path: Path | str,
        output_video: Path | str | None = None,
        expected_count: int | None = None,
        herd_size: int | None = None,
        draw: bool = True,
    ) -> CountingResult:
        video_path = Path(video_path)
        processor = VideoProcessor(video_path)
        counter = LineCounter(
            orientation=self.orientation,
            line_y=self.line_y,
            line_x=self.line_x,
            direction=self.direction,
        )
        writer = None
        out_path = Path(output_video) if output_video else None
        spatial = SpatialTrackMatcher()
        person_filter = PersonFilter() if PERSON_FILTER_ENABLED else None

        if draw and out_path:
            writer = VideoProcessor.create_writer(
                out_path, processor.width, processor.height, processor.fps
            )

        frames_processed = 0
        line_y_px = int(self.line_y * processor.height)
        line_x_px = int(self.line_x * processor.width)

        try:
            for frame in processor.frames():
                frames_processed += 1
                counter.begin_frame(frames_processed)
                result = self.detector.track_frame(frame)

                if result.boxes is not None and len(result.boxes):
                    xyxy_all = result.boxes.xyxy.cpu().tolist()
                    conf_all = (
                        result.boxes.conf.cpu().tolist()
                        if result.boxes.conf is not None
                        else [1.0] * len(xyxy_all)
                    )
                    clss_all = (
                        result.boxes.cls.int().cpu().tolist()
                        if result.boxes.cls is not None
                        else [0] * len(xyxy_all)
                    )
                    if person_filter is not None:
                        keep = select_goat_indices(
                            frame, xyxy_all, conf_all, clss_all, person_filter
                        )
                    else:
                        keep = list(range(len(xyxy_all)))

                    if len(keep) != len(xyxy_all):
                        device = result.boxes.data.device
                        if keep:
                            idx = torch.tensor(keep, device=device, dtype=torch.long)
                            result.boxes = result.boxes[idx]
                        else:
                            result.boxes = result.boxes[0:0]

                    if result.boxes is not None and len(result.boxes):
                        n = len(result.boxes)
                        if result.boxes.id is not None:
                            ids = result.boxes.id.int().cpu().tolist()
                        else:
                            ids = [-1] * n
                        clss = (
                            result.boxes.cls.int().cpu().tolist()
                            if result.boxes.cls is not None
                            else [0] * n
                        )
                        xyxy = result.boxes.xyxy.cpu().tolist()
                        for track_id, box, cls_id in zip(ids, xyxy, clss):
                            if int(cls_id) != GOAT_CLASS_ID:
                                continue
                            x1, y1, x2, y2 = box
                            cx = (x1 + x2) / 2 / processor.width
                            cy = (y1 + y2) / 2 / processor.height
                            tid = int(track_id)
                            if tid < 0:
                                tid = spatial.assign(cx, cy, frames_processed)
                            counter.update(tid, cx, cy)
                counter.end_frame()

                if draw:
                    annotated = result.plot()
                    if self.orientation == "vertical":
                        cv2.line(
                            annotated,
                            (line_x_px, 0),
                            (line_x_px, processor.height),
                            (0, 255, 255),
                            2,
                        )
                    else:
                        cv2.line(
                            annotated,
                            (0, line_y_px),
                            (processor.width, line_y_px),
                            (0, 255, 255),
                            2,
                        )
                    cv2.putText(
                        annotated,
                        f"Count: {counter.total}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2,
                    )
                    if writer:
                        h, w = annotated.shape[:2]
                        if w != processor.width or h != processor.height:
                            annotated = cv2.resize(
                                annotated, (processor.width, processor.height)
                            )
                        writer.write(annotated)

        finally:
            processor.release()
            if writer:
                writer.release()
            if out_path and out_path.is_file() and frames_processed > 0:
                make_browser_playable(out_path)

        return CountingResult(
            total_count=counter.total,
            frames_processed=frames_processed,
            output_video=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            track_ids_counted=counter.counted_ids,
        )
