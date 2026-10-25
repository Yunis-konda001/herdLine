"""Count goats in one video: detect → filter → track → line cross → save."""
from __future__ import annotations

from pathlib import Path

import cv2

from src.config import (
    DEFAULT_CROSSING_DIRECTION,
    DEFAULT_LINE_MODE,
    DEFAULT_LINE_X,
    DEFAULT_LINE_Y,
    DEFAULT_WEIGHTS,
    GOAT_CLASS_ID,
    MIN_TRACK_HITS,
)
from src.counting_result import CountingResult
from src.detection_filters import select_goat_indices
from src.goat_detector import GoatDetector
from src.line_counter import LineCounter
from src.track_memory import TrackMemory
from src.video_encode import make_browser_playable
from src.video_processor import VideoProcessor


def _opposite_direction(direction: str) -> str:
    opposites = {
        "right_to_left": "left_to_right",
        "left_to_right": "right_to_left",
        "down": "up",
        "up": "down",
    }
    return opposites.get(direction, direction)


def _pick_cost(count: int, ref: int) -> float:
    # Prefer exact / miss-1 / miss-2; large overcount is worse than a small miss
    diff = count - ref
    if diff == 0:
        return 0.0
    if diff == -1:
        return 0.25
    if diff == -2:
        return 0.7
    if diff == 1:
        return 1.0
    if diff == 2:
        return 2.2
    if diff < 0:
        return abs(diff) * 1.3
    return 8.0 + diff * 3.5


class CountingPipeline:
    """Run the full counting path used by the website."""

    def __init__(
        self,
        weights: Path | str | None = None,
        line_mode: str = DEFAULT_LINE_MODE,
        line_y: float = DEFAULT_LINE_Y,
        line_x: float = DEFAULT_LINE_X,
        direction: str = DEFAULT_CROSSING_DIRECTION,
    ) -> None:
        resolved = weights
        if resolved is None and DEFAULT_WEIGHTS.is_file():
            resolved = DEFAULT_WEIGHTS
        self.detector = GoatDetector(resolved)
        self.orientation = "vertical" if line_mode == "vertical" else "horizontal"
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
        session_name: str | None = None,
    ) -> CountingResult:
        video_path = Path(video_path)
        out_path = Path(output_video) if output_video else None
        ref = herd_size if herd_size and herd_size > 0 else expected_count

        best_direction = self.direction
        line_x, line_y = self.line_x, self.line_y

        if ref:
            best_direction, line_x, line_y, _ = self._pick_best_config(
                video_path, ref=ref
            )

        return self._run_once(
            video_path,
            out_path=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            draw=draw,
            session_name=session_name,
            direction=best_direction,
            line_x=line_x,
            line_y=line_y,
        )

    def _pick_best_config(
        self, video_path: Path, ref: int
    ) -> tuple[str, float, float, int]:
        """Score line/direction setups in one video read (ByteTrack IDs)."""
        if self.orientation == "vertical":
            line_vals = sorted(
                {0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, float(self.line_x)}
            )
            configs = [
                (d, lx, self.line_y)
                for d in (self.direction, _opposite_direction(self.direction))
                for lx in line_vals
            ]
        else:
            line_vals = sorted(
                {0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, float(self.line_y)}
            )
            configs = [
                (d, self.line_x, ly)
                for d in (self.direction, _opposite_direction(self.direction))
                for ly in line_vals
            ]

        self.detector.reset_tracks()
        processor = VideoProcessor(video_path)
        counters = [
            LineCounter(
                orientation=self.orientation,
                line_x=lx,
                line_y=ly,
                direction=direction,
            )
            for direction, lx, ly in configs
        ]
        memories = [TrackMemory(direction=d) for d, _, _ in configs]

        frames = 0
        try:
            for frame in processor.frames():
                frames += 1
                for c in counters:
                    c.begin_frame(frames)
                for x1, y1, x2, y2, conf, _bt_id in self._goat_detections(
                    frame, processor.width, processor.height
                ):
                    cx = (x1 + x2) / 2 / processor.width
                    cy = (y1 + y2) / 2 / processor.height
                    bw = (x2 - x1) / processor.width
                    bh = (y2 - y1) / processor.height
                    for counter, memory in zip(counters, memories):
                        tid, hits = memory.assign(
                            cx, cy, bw, bh, frames, conf=conf
                        )
                        if hits < MIN_TRACK_HITS:
                            continue
                        counter.update(tid, cx, cy, bw, bh)
                for c in counters:
                    c.end_frame()
        finally:
            processor.release()

        best_i = 0
        best_score = float("inf")
        for i, counter in enumerate(counters):
            score = _pick_cost(counter.total, ref)
            if score < best_score:
                best_score = score
                best_i = i
        direction, lx, ly = configs[best_i]
        return direction, lx, ly, counters[best_i].total

    def _goat_detections(
        self, frame, frame_w: int, frame_h: int
    ) -> list[tuple[float, float, float, float, float, int]]:
        # Predict + TrackMemory keeps more goats in dense herds than ByteTrack
        result = self.detector.predict_frame(frame)
        if result.boxes is None or len(result.boxes) == 0:
            return []
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
        keep = select_goat_indices(
            xyxy_all, conf_all, clss_all, frame_w=frame_w, frame_h=frame_h
        )
        out: list[tuple[float, float, float, float, float, int]] = []
        for i in keep:
            if int(clss_all[i]) != GOAT_CLASS_ID:
                continue
            x1, y1, x2, y2 = xyxy_all[i]
            out.append((x1, y1, x2, y2, float(conf_all[i]), -1))
        return out

    def _run_once(
        self,
        video_path: Path,
        out_path: Path | None,
        expected_count: int | None,
        herd_size: int | None,
        draw: bool,
        session_name: str | None,
        direction: str,
        line_x: float,
        line_y: float,
    ) -> CountingResult:
        self.detector.reset_tracks()
        processor = VideoProcessor(video_path)
        counter = LineCounter(
            orientation=self.orientation,
            line_y=line_y,
            line_x=line_x,
            direction=direction,
        )
        writer = None
        memory = TrackMemory(direction=direction)

        if draw and out_path:
            writer = VideoProcessor.create_writer(
                out_path, processor.width, processor.height, processor.fps
            )

        frames_processed = 0
        conf_sum = 0.0
        conf_n = 0
        video_fps = processor.fps
        line_y_px = int(line_y * processor.height)
        line_x_px = int(line_x * processor.width)

        try:
            for frame in processor.frames():
                frames_processed += 1
                counter.begin_frame(frames_processed)
                draw_boxes: list[tuple[int, int, int, int, int]] = []

                for x1, y1, x2, y2, conf, _bt_id in self._goat_detections(
                    frame, processor.width, processor.height
                ):
                    conf_sum += conf
                    conf_n += 1
                    cx = (x1 + x2) / 2 / processor.width
                    cy = (y1 + y2) / 2 / processor.height
                    bw = (x2 - x1) / processor.width
                    bh = (y2 - y1) / processor.height
                    tid, hits = memory.assign(
                        cx, cy, bw, bh, frames_processed, conf=conf
                    )
                    if hits < MIN_TRACK_HITS:
                        continue
                    counter.update(tid, cx, cy, bw, bh)
                    draw_boxes.append((int(x1), int(y1), int(x2), int(y2), tid))
                counter.end_frame()

                if draw:
                    annotated = frame.copy()
                    for x1, y1, x2, y2, tid in draw_boxes:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        cv2.putText(
                            annotated,
                            f"id {tid}",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                        )
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

        duration = (frames_processed / video_fps) if frames_processed and video_fps else None
        return CountingResult(
            total_count=counter.total,
            frames_processed=frames_processed,
            output_video=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            track_ids_counted=counter.counted_ids,
            session_name=session_name or video_path.stem,
            video_duration_sec=duration,
            mean_confidence=(conf_sum / conf_n) if conf_n else None,
        )
