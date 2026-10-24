"""Count each goat once when its centre crosses the gate line.

Background boxes that never cross the line are ignored.
The same track ID is never counted twice.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _PendingHandoff:
    track_id: int
    cx: float
    cy: float
    frame: int


class LineCounter:
    """One count per goat for a vertical or horizontal gate line."""

    # Off by default: ID handoff was double-counting goats on several clips
    HANDOFF_CY_TOLERANCE = 0.10
    HANDOFF_MAX_FRAMES = 0
    LINE_BAND = 0.25
    LINE_EPS = 0.01
    SAME_GOAT_DIST = 0.08
    SAME_GOAT_DY = 0.10
    MIN_AREA = 0.010
    MIN_SIDE = 0.09
    CROSS_MARGIN = 0.01

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
        self._zone: dict[int, str] = {}
        self._last_cx: dict[int, float] = {}
        self._last_cy: dict[int, float] = {}
        self._last_bw: dict[int, float] = {}
        self._last_bh: dict[int, float] = {}
        self._last_frame: dict[int, int] = {}
        self._seen_entry: set[int] = set()

        self._frame_idx = 0
        self._ids_this_frame: set[int] = set()
        self._ids_prev_frame: set[int] = set()
        self._pending: list[_PendingHandoff] = []

        if orientation == "horizontal":
            if direction not in ("down", "up"):
                raise ValueError("horizontal direction must be 'down' or 'up'")
        else:
            if direction not in ("left_to_right", "right_to_left"):
                raise ValueError("vertical direction must be 'left_to_right' or 'right_to_left'")

    def begin_frame(self, frame_index: int) -> None:
        self._frame_idx = frame_index
        self._ids_this_frame = set()

    def end_frame(self) -> None:
        if self.HANDOFF_MAX_FRAMES <= 0:
            self._ids_prev_frame = set(self._ids_this_frame)
            self._pending.clear()
            return

        # Tracks lost near the line may hand off to a new ID on the exit side
        lost = self._ids_prev_frame - self._ids_this_frame
        for track_id in lost:
            if track_id in self._counted or track_id in self._already_past:
                continue
            if track_id not in self._seen_entry:
                continue
            cx = self._last_cx.get(track_id)
            cy = self._last_cy.get(track_id)
            if cx is None or cy is None:
                continue
            if self._line_distance(cx, cy) > self.LINE_BAND:
                continue
            self._pending.append(
                _PendingHandoff(track_id=track_id, cx=cx, cy=cy, frame=self._frame_idx)
            )

        new_ids = self._ids_this_frame - self._ids_prev_frame
        for track_id in new_ids:
            if track_id in self._counted or track_id in self._already_past:
                continue
            cx = self._last_cx.get(track_id)
            cy = self._last_cy.get(track_id)
            if cx is None or cy is None:
                continue
            if self._current_zone(cx, cy) != "exit":
                continue
            if self._line_distance(cx, cy) > self.LINE_BAND:
                self._already_past.add(track_id)
                continue
            if not self._try_handoff_count(cy, track_id):
                self._already_past.add(track_id)

        self._ids_prev_frame = set(self._ids_this_frame)
        self._expire_pending()

    def update(
        self,
        track_id: int,
        cx_norm: float,
        cy_norm: float,
        bw: float = 0.0,
        bh: float = 0.0,
        confirmed: bool = True,
    ) -> bool:
        if track_id < 0:
            return False

        self._ids_this_frame.add(track_id)
        zone = self._current_zone(cx_norm, cy_norm)

        if track_id in self._counted or track_id in self._already_past:
            self._remember(track_id, cx_norm, cy_norm, bw, bh)
            self._store_prev(track_id, cx_norm, cy_norm)
            self._zone[track_id] = zone
            return False

        if not confirmed:
            self._remember(track_id, cx_norm, cy_norm, bw, bh)
            self._zone[track_id] = zone
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        # Tiny noise never counts
        if bw > 0 and bh > 0 and bw * bh < self.MIN_AREA and max(bw, bh) < self.MIN_SIDE:
            self._already_past.add(track_id)
            self._remember(track_id, cx_norm, cy_norm, bw, bh)
            self._store_prev(track_id, cx_norm, cy_norm)
            self._zone[track_id] = zone
            return False

        # Same goat already counted nearby — do not count again
        if self._near_counted(cx_norm, cy_norm):
            self._already_past.add(track_id)
            self._remember(track_id, cx_norm, cy_norm, bw, bh)
            self._store_prev(track_id, cx_norm, cy_norm)
            self._zone[track_id] = zone
            return False

        # First sight: only watch. Count happens on a real line cross later.
        if track_id not in self._prev_x:
            if zone in ("entry", "mid"):
                self._seen_entry.add(track_id)
            elif zone == "exit" and self._line_distance(cx_norm, cy_norm) > self.LINE_BAND:
                self._already_past.add(track_id)
            self._remember(track_id, cx_norm, cy_norm, bw, bh)
            self._zone[track_id] = zone
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        if zone in ("entry", "mid"):
            self._seen_entry.add(track_id)

        crossed = self._centroid_crossed(track_id, cx_norm, cy_norm)
        on_entry_before = track_id in self._seen_entry or self._was_on_entry_side(track_id)

        self._remember(track_id, cx_norm, cy_norm, bw, bh)

        if on_entry_before and crossed:
            self._register_count(track_id)
            self._zone[track_id] = zone
            self._store_prev(track_id, cx_norm, cy_norm)
            return True

        self._zone[track_id] = zone if zone != "mid" else self._zone.get(track_id, "mid")
        if zone == "mid" and self._zone.get(track_id) == "entry":
            self._zone[track_id] = "mid"
        self._store_prev(track_id, cx_norm, cy_norm)
        return False

    def same_goat_id(self, track_id: int, cx: float, cy: float, bw: float = 0.0, bh: float = 0.0) -> int:
        match = self._match_counted(cx, cy)
        if match is None or match == track_id:
            return track_id
        return match

    def _register_count(self, track_id: int, cy: float | None = None) -> None:
        if track_id in self._counted:
            return
        self._counted.add(track_id)
        self.total += 1

    def _remember(self, track_id: int, cx: float, cy: float, bw: float, bh: float) -> None:
        self._last_cx[track_id] = cx
        self._last_cy[track_id] = cy
        self._last_bw[track_id] = bw
        self._last_bh[track_id] = bh
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

    def _near_counted(self, cx: float, cy: float) -> bool:
        return self._match_counted(cx, cy) is not None

    def _match_counted(self, cx: float, cy: float) -> int | None:
        best_id = None
        best_dist = 999.0
        for track_id in self._counted:
            seen = self._last_frame.get(track_id)
            if seen is None or self._frame_idx - seen > 8:
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

    def _try_handoff_count(self, cy: float, new_track_id: int) -> bool:
        self._expire_pending()
        best_idx = None
        best_dy = self.HANDOFF_CY_TOLERANCE + 1.0
        for i, pending in enumerate(self._pending):
            if pending.track_id in self._counted:
                continue
            dy = abs(pending.cy - cy)
            if dy <= self.HANDOFF_CY_TOLERANCE and dy < best_dy:
                best_dy = dy
                best_idx = i
        if best_idx is None:
            return False
        pending = self._pending.pop(best_idx)
        self._seen_entry.add(new_track_id)
        self._register_count(new_track_id)
        self._counted.add(pending.track_id)
        self._zone[new_track_id] = "exit"
        return True

    def _expire_pending(self) -> None:
        cutoff = self._frame_idx - self.HANDOFF_MAX_FRAMES
        self._pending = [
            p for p in self._pending if p.frame >= cutoff and p.track_id not in self._counted
        ]

    @property
    def counted_ids(self) -> list[int]:
        return sorted(self._counted)
