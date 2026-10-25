"""Count each goat once when its centre crosses the virtual gate line.

Background boxes that never cross the line are ignored.
The same track ID is never counted twice.
"""
from __future__ import annotations


class LineCounter:
    """One count per goat for a vertical or horizontal gate line."""

    LINE_EPS = 0.01
    # Tight: only block a *new* ID that is almost on top of a just-counted goat
    SAME_GOAT_DIST = 0.055
    SAME_GOAT_DY = 0.08
    MIN_AREA = 0.008
    MIN_SIDE = 0.08
    CROSS_MARGIN = 0.005
    NEAR_FRAMES = 12

    def __init__(
        self,
        orientation: str = "horizontal",
        line_y: float = 0.5,
        line_x: float = 0.5,
        direction: str = "down",
    ) -> None:
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        self.orientation = orientation
        self.line_y = line_y
        self.line_x = line_x
        self.direction = direction

        self._counted: set[int] = set()
        self._already_past: set[int] = set()
        self.total = 0

        self._prev_x: dict[int, float] = {}
        self._prev_y: dict[int, float] = {}
        self._last_cx: dict[int, float] = {}
        self._last_cy: dict[int, float] = {}
        self._last_frame: dict[int, int] = {}
        self._seen_entry: set[int] = set()
        self._frame_idx = 0

        if orientation == "horizontal":
            if direction not in ("down", "up"):
                raise ValueError("horizontal direction must be 'down' or 'up'")
        else:
            if direction not in ("left_to_right", "right_to_left"):
                raise ValueError("vertical direction must be 'left_to_right' or 'right_to_left'")

    def begin_frame(self, frame_index: int) -> None:
        self._frame_idx = frame_index

    def end_frame(self) -> None:
        return

    def update(
        self,
        track_id: int,
        cx_norm: float,
        cy_norm: float,
        bw: float = 0.0,
        bh: float = 0.0,
        confirmed: bool = True,
    ) -> bool:
        if track_id < 0 or not confirmed:
            return False

        zone = self._current_zone(cx_norm, cy_norm)

        if track_id in self._counted or track_id in self._already_past:
            self._remember(track_id, cx_norm, cy_norm)
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        if bw > 0 and bh > 0 and bw * bh < self.MIN_AREA and max(bw, bh) < self.MIN_SIDE:
            self._already_past.add(track_id)
            self._remember(track_id, cx_norm, cy_norm)
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        # New ID almost on top of a goat that just counted — same animal, skip
        if self._match_counted(cx_norm, cy_norm) is not None:
            self._already_past.add(track_id)
            self._remember(track_id, cx_norm, cy_norm)
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        # First sight: watch only. Count only when the line is crossed.
        if track_id not in self._prev_x:
            if zone in ("entry", "mid"):
                self._seen_entry.add(track_id)
            elif zone == "exit" and self._line_distance(cx_norm, cy_norm) > 0.22:
                self._already_past.add(track_id)
            self._remember(track_id, cx_norm, cy_norm)
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        if zone in ("entry", "mid"):
            self._seen_entry.add(track_id)

        crossed = self._centroid_crossed(track_id, cx_norm, cy_norm)
        on_entry_before = track_id in self._seen_entry or self._was_on_entry_side(track_id)

        self._remember(track_id, cx_norm, cy_norm)

        if on_entry_before and crossed:
            self._counted.add(track_id)
            self.total += 1
            self._store_prev(track_id, cx_norm, cy_norm)
            return True

        self._store_prev(track_id, cx_norm, cy_norm)
        return False

    def same_goat_id(
        self, track_id: int, cx: float, cy: float, bw: float = 0.0, bh: float = 0.0
    ) -> int:
        # Do not remap uncounted tracks — that was merging separate goats
        if track_id in self._counted:
            return track_id
        match = self._match_counted(cx, cy)
        if match is None:
            return track_id
        # Only remap onto an already-counted ID (ID swap after a count)
        return match

    def _remember(self, track_id: int, cx: float, cy: float) -> None:
        self._last_cx[track_id] = cx
        self._last_cy[track_id] = cy
        self._last_frame[track_id] = self._frame_idx

    def _store_prev(self, track_id: int, cx: float, cy: float) -> None:
        self._prev_x[track_id] = cx
        self._prev_y[track_id] = cy

    def _was_on_entry_side(self, track_id: int) -> bool:
        px = self._prev_x.get(track_id)
        py = self._prev_y.get(track_id)
        if px is None or py is None:
            return False
        return self._current_zone(px, py) in ("entry", "mid")

    def _match_counted(self, cx: float, cy: float) -> int | None:
        best_id = None
        best_dist = 999.0
        for track_id in self._counted:
            seen = self._last_frame.get(track_id)
            if seen is None or self._frame_idx - seen > self.NEAR_FRAMES:
                continue
            px = self._last_cx.get(track_id)
            py = self._last_cy.get(track_id)
            if px is None or py is None:
                continue
            if abs(cy - py) > self.SAME_GOAT_DY:
                continue
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist <= self.SAME_GOAT_DIST and dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _line_distance(self, cx: float, cy: float) -> float:
        if self.orientation == "vertical":
            return abs(cx - self.line_x)
        return abs(cy - self.line_y)

    def _current_zone(self, cx: float, cy: float) -> str:
        if self.orientation == "vertical":
            line = self.line_x
            if self.direction == "left_to_right":
                if cx < line - self.LINE_EPS:
                    return "entry"
                if cx >= line + self.LINE_EPS:
                    return "exit"
            else:
                if cx > line + self.LINE_EPS:
                    return "entry"
                if cx <= line - self.LINE_EPS:
                    return "exit"
            return "mid"
        line = self.line_y
        if self.direction == "down":
            if cy < line - self.LINE_EPS:
                return "entry"
            if cy >= line + self.LINE_EPS:
                return "exit"
        else:
            if cy > line + self.LINE_EPS:
                return "entry"
            if cy <= line - self.LINE_EPS:
                return "exit"
        return "mid"

    def _centroid_crossed(self, track_id: int, cx: float, cy: float) -> bool:
        m = self.CROSS_MARGIN
        if self.orientation == "horizontal":
            prev = self._prev_y.get(track_id)
            if prev is None:
                return False
            line = self.line_y
            if self.direction == "down":
                return prev < line - m and cy >= line
            return prev > line + m and cy <= line
        prev = self._prev_x.get(track_id)
        if prev is None:
            return False
        line = self.line_x
        if self.direction == "left_to_right":
            return prev < line - m and cx >= line
        return prev > line + m and cx <= line

    @property
    def counted_ids(self) -> list[int]:
        return sorted(self._counted)
