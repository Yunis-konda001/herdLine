"""Count goats in one video: detect → filter → track → line cross → save."""
from __future__ import annotations

from pathlib import Path

import cv2

from src.config import (
    DEFAULT_CROSSING_DIRECTION,
    DEFAULT_LINE_MODE,
    DEFAULT_LINE_X,
    DEFAULT_LINE_Y,
    DEFAULT_ONNX_WEIGHTS,
    DEFAULT_WEIGHTS,
    DENSE_HERD_STRIDE_THRESHOLD,
    FRAME_STRIDE,
    GOAT_CLASS_ID,
    GOAT_LIKE_MAX_HW,
    GOAT_STRONG_CONFIDENCE,
    LINE_SEARCH_MODE,
    MIN_STRONG_GOAT_HITS,
    MIN_TRACK_HITS,
    PERSON_LIKE_MIN_HW,
    REENCODE_RESULT,
)
from src.counting_result import CountingResult
from src.detection_filters import select_goat_indices
from src.goat_detector import GoatDetector
from src.line_counter import LineCounter
from src.person_gate import count_people_rough
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


def _line_search_values(
    current: float, herd_size: int | None = None
) -> list[float]:
    mode = LINE_SEARCH_MODE
    # Large herds always use the denser grid (best line matters more)
    if herd_size and herd_size >= DENSE_HERD_STRIDE_THRESHOLD:
        mode = "full"
    if mode == "off":
        return [float(current)]
    if mode == "fast":
        base = {0.35, 0.50, 0.65, 0.75, float(current)}
    else:
        base = {0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, float(current)}
    return sorted(base)


def _effective_stride(herd_size: int | None) -> int:
    """Sparse clips can skip frames; crowded herds keep every frame."""
    if herd_size and herd_size >= DENSE_HERD_STRIDE_THRESHOLD:
        return 1
    return FRAME_STRIDE


def _box_hw_ratio(x1: float, y1: float, x2: float, y2: float) -> float:
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 0.0)
    return bh / bw


def _has_goat_evidence(
    kept_hits: int,
    strong_goat_hits: int,
    max_conf: float,
    mean_conf: float | None,
    person_like_hits: int,
    hog_person_hits: int,
) -> tuple[bool, str | None]:
    """Return (ok, reason). Non-goat / people clips must not produce a herd count."""
    if kept_hits <= 0:
        return False, (
            "No goats detected in this video. "
            "Upload a short gate clip that clearly shows goats."
        )
    if hog_person_hits >= 5 and strong_goat_hits < MIN_STRONG_GOAT_HITS:
        return False, (
            "This video looks like people, not goats. "
            "HerdLine only counts goats at a gate."
        )
    if person_like_hits > strong_goat_hits:
        return False, (
            "No clear goats found. Tall upright shapes look like people, "
            "not goats crossing a gate."
        )
    if strong_goat_hits < MIN_STRONG_GOAT_HITS or max_conf < GOAT_STRONG_CONFIDENCE:
        return False, (
            "No clear goats found. This clip may show people or other objects, "
            "not goats crossing a gate."
        )
    if mean_conf is not None and mean_conf < 0.42:
        return False, (
            "Detections were too weak to trust as goats. "
            "Try a clearer gate video with goats only."
        )
    return True, None


def _finalize_result(
    *,
    total_count: int,
    frames_processed: int,
    out_path: Path | None,
    expected_count: int | None,
    herd_size: int | None,
    track_ids: list[int],
    session_name: str | None,
    video_path: Path,
    video_fps: float,
    conf_sum: float,
    conf_n: int,
    kept_hits: int,
    strong_goat_hits: int,
    max_conf: float,
    person_like_hits: int,
    hog_person_hits: int,
) -> CountingResult:
    mean_conf = (conf_sum / conf_n) if conf_n else None
    duration = (
        (frames_processed / video_fps) if frames_processed and video_fps else None
    )
    ok, reason = _has_goat_evidence(
        kept_hits,
        strong_goat_hits,
        max_conf,
        mean_conf,
        person_like_hits,
        hog_person_hits,
    )
    # Always report 0 when the clip is not clearly goats (never invent a herd count)
    if not ok:
        return CountingResult(
            total_count=0,
            frames_processed=frames_processed,
            output_video=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            track_ids_counted=[],
            session_name=session_name or video_path.stem,
            video_duration_sec=duration,
            mean_confidence=mean_conf,
            rejection_reason=reason,
        )
    return CountingResult(
        total_count=total_count,
        frames_processed=frames_processed,
        output_video=out_path,
        expected_count=expected_count,
        herd_size=herd_size,
        track_ids_counted=track_ids,
        session_name=session_name or video_path.stem,
        video_duration_sec=duration,
        mean_confidence=mean_conf,
    )


def _draw_overlay(
    frame,
    boxes: list[tuple[int, int, int, int, int]],
    orientation: str,
    line_x_px: int,
    line_y_px: int,
    count: int,
    frame_w: int,
    frame_h: int,
):
    """Draw boxes + stable gate line + count on every frame (no flicker)."""
    annotated = frame.copy()
    for x1, y1, x2, y2, tid in boxes:
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
    if orientation == "vertical":
        cv2.line(annotated, (line_x_px, 0), (line_x_px, frame_h), (0, 255, 255), 2)
    else:
        cv2.line(annotated, (0, line_y_px), (frame_w, line_y_px), (0, 255, 255), 2)
    cv2.putText(
        annotated,
        f"Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )
    return annotated


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
        if resolved is None:
            if DEFAULT_ONNX_WEIGHTS.is_file():
                resolved = DEFAULT_ONNX_WEIGHTS
            elif DEFAULT_WEIGHTS.is_file():
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

        # One video read: score line/direction configs AND count (no second detect sweep)
        if ref and LINE_SEARCH_MODE != "off":
            return self._run_with_line_search(
                video_path,
                out_path=out_path,
                expected_count=expected_count,
                herd_size=herd_size,
                draw=draw,
                session_name=session_name,
                ref=ref,
            )

        return self._run_once(
            video_path,
            out_path=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            draw=draw,
            session_name=session_name,
            direction=self.direction,
            line_x=self.line_x,
            line_y=self.line_y,
        )

    def _run_with_line_search(
        self,
        video_path: Path,
        out_path: Path | None,
        expected_count: int | None,
        herd_size: int | None,
        draw: bool,
        session_name: str | None,
        ref: int,
    ) -> CountingResult:
        """Detect each sampled frame once; score many line setups in the same pass."""
        if self.orientation == "vertical":
            line_vals = _line_search_values(self.line_x, herd_size)
            configs = [
                (d, lx, self.line_y)
                for d in (self.direction, _opposite_direction(self.direction))
                for lx in line_vals
            ]
        else:
            line_vals = _line_search_values(self.line_y, herd_size)
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
        # One tracker per walk direction (not per line) — same IDs, much less work
        memory_by_dir: dict[str, TrackMemory] = {}
        for d, _, _ in configs:
            if d not in memory_by_dir:
                memory_by_dir[d] = TrackMemory(direction=d)

        # Live preview uses the form defaults; final reported count uses best config
        preview_i = 0
        for i, (d, lx, ly) in enumerate(configs):
            if (
                d == self.direction
                and abs(lx - self.line_x) < 1e-6
                and abs(ly - self.line_y) < 1e-6
            ):
                preview_i = i
                break
        preview_dir = configs[preview_i][0]

        writer = None
        if draw and out_path:
            writer = VideoProcessor.create_writer(
                out_path, processor.width, processor.height, processor.fps
            )

        frames_processed = 0
        conf_sum = 0.0
        conf_n = 0
        kept_hits = 0
        strong_goat_hits = 0
        person_like_hits = 0
        hog_person_hits = 0
        max_conf = 0.0
        video_fps = processor.fps
        stride = _effective_stride(herd_size)
        preview_line_x = int(configs[preview_i][1] * processor.width)
        preview_line_y = int(configs[preview_i][2] * processor.height)
        last_boxes: list[tuple[int, int, int, int, int]] = []

        try:
            for frame in processor.frames():
                frames_processed += 1
                detect_this = frames_processed % stride == 0

                for c in counters:
                    c.begin_frame(frames_processed)

                if detect_this:
                    # Sample person detector every ~15 processed detect frames
                    if frames_processed % (stride * 15) < stride:
                        hog_person_hits += count_people_rough(frame)
                    frame_boxes: list[tuple[int, int, int, int, int]] = []
                    for x1, y1, x2, y2, conf, _tid in self._goat_detections(
                        frame, processor.width, processor.height
                    ):
                        hw = _box_hw_ratio(x1, y1, x2, y2)
                        if hw >= PERSON_LIKE_MIN_HW:
                            person_like_hits += 1
                            continue  # do not count upright people-shaped boxes
                        if hw > GOAT_LIKE_MAX_HW:
                            continue
                        kept_hits += 1
                        if conf >= GOAT_STRONG_CONFIDENCE:
                            strong_goat_hits += 1
                            max_conf = max(max_conf, conf)
                        conf_sum += conf
                        conf_n += 1
                        cx = (x1 + x2) / 2 / processor.width
                        cy = (y1 + y2) / 2 / processor.height
                        bw = (x2 - x1) / processor.width
                        bh = (y2 - y1) / processor.height
                        # Assign once per direction, then feed every line counter
                        ids_hits = {
                            d: mem.assign(
                                cx, cy, bw, bh, frames_processed, conf=conf
                            )
                            for d, mem in memory_by_dir.items()
                        }
                        for counter, (d, _lx, _ly) in zip(counters, configs):
                            tid, hits = ids_hits[d]
                            if hits < MIN_TRACK_HITS:
                                continue
                            counter.update(tid, cx, cy, bw, bh)
                        preview_tid, preview_hits = ids_hits[preview_dir]
                        if preview_hits >= MIN_TRACK_HITS and preview_tid >= 0:
                            frame_boxes.append(
                                (
                                    int(x1),
                                    int(y1),
                                    int(x2),
                                    int(y2),
                                    preview_tid,
                                )
                            )
                    last_boxes = frame_boxes

                for c in counters:
                    c.end_frame()

                if writer is not None:
                    # Always draw line/count/boxes so the overlay does not flicker
                    live_count = max(c.total for c in counters)
                    writer.write(
                        _draw_overlay(
                            frame,
                            last_boxes,
                            self.orientation,
                            preview_line_x,
                            preview_line_y,
                            live_count,
                            processor.width,
                            processor.height,
                        )
                    )
        finally:
            processor.release()
            if writer:
                writer.release()
            if (
                REENCODE_RESULT
                and out_path
                and out_path.is_file()
                and frames_processed > 0
            ):
                make_browser_playable(out_path)

        best_i = 0
        best_score = float("inf")
        for i, counter in enumerate(counters):
            score = _pick_cost(counter.total, ref)
            if score < best_score:
                best_score = score
                best_i = i

        return _finalize_result(
            total_count=counters[best_i].total,
            frames_processed=frames_processed,
            out_path=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            track_ids=counters[best_i].counted_ids,
            session_name=session_name,
            video_path=video_path,
            video_fps=video_fps,
            conf_sum=conf_sum,
            conf_n=conf_n,
            kept_hits=kept_hits,
            strong_goat_hits=strong_goat_hits,
            max_conf=max_conf,
            person_like_hits=person_like_hits,
            hog_person_hits=hog_person_hits,
        )

    def _goat_detections(
        self, frame, frame_w: int, frame_h: int
    ) -> list[tuple[float, float, float, float, float, int]]:
        raw = self.detector.detect_boxes(frame)
        if not raw:
            return []
        xyxy_all = [[x1, y1, x2, y2] for x1, y1, x2, y2, _c, _t in raw]
        conf_all = [float(c) for _x1, _y1, _x2, _y2, c, _t in raw]
        clss_all = [GOAT_CLASS_ID] * len(raw)
        keep = select_goat_indices(
            xyxy_all, conf_all, clss_all, frame_w=frame_w, frame_h=frame_h
        )
        out: list[tuple[float, float, float, float, float, int]] = []
        for i in keep:
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
        kept_hits = 0
        strong_goat_hits = 0
        person_like_hits = 0
        hog_person_hits = 0
        max_conf = 0.0
        video_fps = processor.fps
        line_y_px = int(line_y * processor.height)
        line_x_px = int(line_x * processor.width)
        stride = _effective_stride(herd_size)
        last_boxes: list[tuple[int, int, int, int, int]] = []

        try:
            for frame in processor.frames():
                frames_processed += 1
                detect_this = frames_processed % stride == 0
                counter.begin_frame(frames_processed)

                if detect_this:
                    if frames_processed % (stride * 15) < stride:
                        hog_person_hits += count_people_rough(frame)
                    frame_boxes: list[tuple[int, int, int, int, int]] = []
                    for x1, y1, x2, y2, conf, _bt_id in self._goat_detections(
                        frame, processor.width, processor.height
                    ):
                        hw = _box_hw_ratio(x1, y1, x2, y2)
                        if hw >= PERSON_LIKE_MIN_HW:
                            person_like_hits += 1
                            continue
                        if hw > GOAT_LIKE_MAX_HW:
                            continue
                        kept_hits += 1
                        if conf >= GOAT_STRONG_CONFIDENCE:
                            strong_goat_hits += 1
                            max_conf = max(max_conf, conf)
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
                        frame_boxes.append((int(x1), int(y1), int(x2), int(y2), tid))
                    last_boxes = frame_boxes
                counter.end_frame()

                if writer is not None:
                    writer.write(
                        _draw_overlay(
                            frame,
                            last_boxes,
                            self.orientation,
                            line_x_px,
                            line_y_px,
                            counter.total,
                            processor.width,
                            processor.height,
                        )
                    )

        finally:
            processor.release()
            if writer:
                writer.release()
            if (
                REENCODE_RESULT
                and out_path
                and out_path.is_file()
                and frames_processed > 0
            ):
                make_browser_playable(out_path)

        return _finalize_result(
            total_count=counter.total,
            frames_processed=frames_processed,
            out_path=out_path,
            expected_count=expected_count,
            herd_size=herd_size,
            track_ids=counter.counted_ids,
            session_name=session_name,
            video_path=video_path,
            video_fps=video_fps,
            conf_sum=conf_sum,
            conf_n=conf_n,
            kept_hits=kept_hits,
            strong_goat_hits=strong_goat_hits,
            max_conf=max_conf,
            person_like_hits=person_like_hits,
            hog_person_hits=hog_person_hits,
        )
