"""Count goats in one video: detect → filter → track → line count → save."""
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
    # Prefer exact, then ±1; never prefer a large overcount over a miss
    diff = count - ref
    if diff == 0:
        return 0.0
    if diff == -1:
        return 0.35
    if diff == 1:
        return 0.55
    if diff == -2:
        return 1.0
    if diff == 2:
        return 2.5
    if diff < 0:
        return abs(diff) * 1.5
    return 10.0 + diff * 4.0


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
            # One video pass, two directions — much faster than re-reading the clip
            first_count, second_count = self._count_both_directions(
                video_path,
                line_x=line_x,
                line_y=line_y,
            )
            other = _opposite_direction(self.direction)
            if _pick_cost(second_count, ref) < _pick_cost(first_count, ref):
                best_direction = other
        elif not (draw and out_path is not None):
            return self._run_once(
                video_path,
                out_path=out_path,
                expected_count=expected_count,
                herd_size=herd_size,
                draw=draw,
                session_name=session_name,
                direction=self.direction,
                line_x=line_x,
                line_y=line_y,
            )

        # Final pass draws the annotated video with the chosen direction
        if draw and out_path is not None:
            return self._run_once(
                video_path,
                out_path=out_path,
                expected_count=expected_count,
                herd_size=herd_size,
                draw=True,
                session_name=session_name,
                direction=best_direction,
                line_x=line_x,
                line_y=line_y,
            )

        return self._run_once(
            video_path,
            out_path=None,
            expected_count=expected_count,
            herd_size=herd_size,
            draw=False,
            session_name=session_name,
            direction=best_direction,
            line_x=line_x,
            line_y=line_y,
        )

    def _count_both_directions(
        self,
        video_path: Path,
        line_x: float,
        line_y: float,
    ) -> tuple[int, int]:
        """Detect each frame once; update two counters (both walk directions)."""
        self.detector.reset_tracks()
        processor = VideoProcessor(video_path)
        other = _opposite_direction(self.direction)
        c_a = LineCounter(
            orientation=self.orientation,
            line_y=line_y,
            line_x=line_x,
            direction=self.direction,
        )
        c_b = LineCounter(
            orientation=self.orientation,
            line_y=line_y,
            line_x=line_x,
            direction=other,
        )
        mem_a = TrackMemory(direction=self.direction)
        mem_b = TrackMemory(direction=other)
        frames = 0

        try:
            for frame in processor.frames():
                frames += 1
                c_a.begin_frame(frames)
                c_b.begin_frame(frames)
                boxes = self._goat_boxes(frame, processor.width, processor.height)
                for x1, y1, x2, y2, _conf in boxes:
                    cx = (x1 + x2) / 2 / processor.width
                    cy = (y1 + y2) / 2 / processor.height
                    bw = (x2 - x1) / processor.width
                    bh = (y2 - y1) / processor.height
                    tid_a, hits_a = mem_a.assign(cx, cy, bw, bh, frames)
                    if hits_a >= 2:
                        tid_a = c_a.same_goat_id(tid_a, cx, cy, bw, bh)
                        c_a.update(tid_a, cx, cy, bw, bh)
                    tid_b, hits_b = mem_b.assign(cx, cy, bw, bh, frames)
                    if hits_b >= 2:
                        tid_b = c_b.same_goat_id(tid_b, cx, cy, bw, bh)
                        c_b.update(tid_b, cx, cy, bw, bh)
                c_a.end_frame()
                c_b.end_frame()
        finally:
            processor.release()

        return c_a.total, c_b.total

    def _goat_boxes(
        self, frame, frame_w: int, frame_h: int
    ) -> list[tuple[float, float, float, float, float]]:
        result = self.detector.track_frame(frame)
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
        out: list[tuple[float, float, float, float, float]] = []
        for i in keep:
            if int(clss_all[i]) != GOAT_CLASS_ID:
                continue
            x1, y1, x2, y2 = xyxy_all[i]
            out.append((x1, y1, x2, y2, float(conf_all[i])))
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
        line_x: float | None = None,
        line_y: float | None = None,
    ) -> CountingResult:
        use_x = self.line_x if line_x is None else line_x
        use_y = self.line_y if line_y is None else line_y
        self.detector.reset_tracks()
        processor = VideoProcessor(video_path)
        counter = LineCounter(
            orientation=self.orientation,
            line_y=use_y,
            line_x=use_x,
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
        line_y_px = int(use_y * processor.height)
        line_x_px = int(use_x * processor.width)

        try:
            for frame in processor.frames():
                frames_processed += 1
                counter.begin_frame(frames_processed)
                draw_boxes: list[tuple[int, int, int, int, int]] = []
                boxes = self._goat_boxes(frame, processor.width, processor.height)

                for x1, y1, x2, y2, conf in boxes:
                    conf_sum += conf
                    conf_n += 1
                    cx = (x1 + x2) / 2 / processor.width
                    cy = (y1 + y2) / 2 / processor.height
                    bw = (x2 - x1) / processor.width
                    bh = (y2 - y1) / processor.height
                    tid, hits = memory.assign(cx, cy, bw, bh, frames_processed)
                    if hits < 2:
                        continue
                    tid = counter.same_goat_id(tid, cx, cy, bw, bh)
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
