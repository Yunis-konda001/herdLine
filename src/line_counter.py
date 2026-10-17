"""Count a goat only when it crosses the line. Ignore goats already past it."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _PendingHandoff:
    track_id: int
    cx: float
    cy: float
    frame: int


class LineCounter:
    """
    Count a goat once, when it is first seen on the entry side of the line.

    The same track keeps that count after it crosses. A track that is already
    on the exit side the first time it is seen is not counted.
    """

    HANDOFF_CY_TOLERANCE = 0.15
    HANDOFF_MAX_FRAMES = 15
    # How close to the line a late ID can still be the same goat.
    LINE_BAND = 0.22
    RECENT_FRAMES = 18
    RECENT_DY = 0.12
    LINE_EPS = 0.012
    # A second box on the same goat should not add another count.
    SAME_GOAT_DIST = 0.12

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
        self._past_at: dict[int, int] = {}
        self.total = 0

        self._prev_x: dict[int, float] = {}
        self._prev_y: dict[int, float] = {}
        self._zone: dict[int, str] = {}
        self._last_cx: dict[int, float] = {}
        self._last_cy: dict[int, float] = {}
        self._last_bw: dict[int, float] = {}
        self._last_bh: dict[int, float] = {}
        self._last_frame: dict[int, int] = {}

        self._frame_idx = 0
        self._ids_this_frame: set[int] = set()
        self._ids_prev_frame: set[int] = set()
        self._pending: list[_PendingHandoff] = []
        self._recent_counts: list[tuple[float, int]] = []
        self._count_anchor: list[tuple[float, float, int]] = []

        if orientation == "horizontal":
            if direction not in ("down", "up"):
                raise ValueError("horizontal direction must be 'down' or 'up'")
            if not 0.0 <= line_y <= 1.0:
                raise ValueError("line_y must be between 0 and 1")
        else:
            if direction not in ("left_to_right", "right_to_left"):
                raise ValueError("vertical direction must be 'left_to_right' or 'right_to_left'")
            if not 0.0 <= line_x <= 1.0:
                raise ValueError("line_x must be between 0 and 1")

    def begin_frame(self, frame_index: int) -> None:
        self._frame_idx = frame_index
        self._ids_this_frame = set()

    def end_frame(self) -> None:
        lost = self._ids_prev_frame - self._ids_this_frame
        for track_id in lost:
            if track_id in self._counted or track_id in self._already_past:
                continue
            if self._zone.get(track_id) not in ("entry", "mid"):
                continue
            cx = self._last_cx[track_id]
            cy = self._last_cy[track_id]
            if self._line_distance(cx, cy) > self.LINE_BAND + 0.08:
                continue
            self._pending.append(
                _PendingHandoff(track_id=track_id, cx=cx, cy=cy, frame=self._frame_idx)
            )

        new_ids = self._ids_this_frame - self._ids_prev_frame
        for track_id in new_ids:
            if track_id in self._counted or track_id in self._already_past:
                continue
            cy = self._last_cy.get(track_id)
            cx = self._last_cx.get(track_id)
            if cy is None or cx is None:
                continue
            if self._current_zone(cx, cy) != "exit":
                continue
            distance = self._line_distance(cx, cy)
            if distance > self.LINE_BAND:
                self._mark_past(track_id)
                continue
            # Only an ID swap from a goat that was still on the entry side.
            # A goat that first appears already past the line is not a new crossing.
            if not self._try_handoff_count(cy, track_id):
                self._mark_past(track_id)

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

        if track_id in self._counted or track_id in self._already_past:
            # First box was already past the line, but a wider box still covers the gate.
            marked = self._past_at.get(track_id)
            if (
                track_id not in self._counted
                and marked is not None
                and self._frame_idx - marked <= 6
                and bw >= 0.16
                and self._reaches_entry(cx_norm, cy_norm, bw, bh)
                and not self._near_counted(cx_norm, cy_norm, bw, bh)
            ):
                self._already_past.discard(track_id)
                self._remember(track_id, cx_norm, cy_norm, bw, bh)
                self._register_count(track_id)
                self._zone[track_id] = self._current_zone(cx_norm, cy_norm)
                self._store_prev(track_id, cx_norm, cy_norm)
                return True
            # A loose match must not drag this goat onto a different animal.
            if self._jump_ok(track_id, cx_norm, cy_norm, bw, bh):
                self._remember(track_id, cx_norm, cy_norm, bw, bh)
            self._store_prev(track_id, cx_norm, cy_norm)
            self._zone[track_id] = self._current_zone(cx_norm, cy_norm)
            return False

        self._remember(track_id, cx_norm, cy_norm, bw, bh)

        zone = self._current_zone(cx_norm, cy_norm)
        prev_zone = self._zone.get(track_id)

        if prev_zone is None or (
            confirmed and track_id not in self._counted and track_id not in self._already_past
        ):
            if not confirmed:
                self._zone[track_id] = zone
                self._store_prev(track_id, cx_norm, cy_norm)
                return False
            if bw > 0 and bh > 0 and bw * bh < 0.008 and max(bw, bh) < 0.08:
                self._mark_past(track_id)
                self._zone[track_id] = zone
                self._store_prev(track_id, cx_norm, cy_norm)
                return False
            if self._near_counted(cx_norm, cy_norm, bw, bh) or self._small_follow(
                cx_norm, cy_norm, bw, bh
            ):
                self._mark_past(track_id)
                self._zone[track_id] = zone
                self._store_prev(track_id, cx_norm, cy_norm)
                return False
            if zone in ("entry", "mid") or self._reaches_entry(cx_norm, cy_norm, bw, bh):
                self._register_count(track_id)
                self._zone[track_id] = zone
                self._store_prev(track_id, cx_norm, cy_norm)
                return True
            if zone == "exit" and self._line_distance(cx_norm, cy_norm) <= self.LINE_BAND:
                self._zone[track_id] = zone
                self._store_prev(track_id, cx_norm, cy_norm)
                return False
            self._mark_past(track_id)
            self._zone[track_id] = zone
            self._store_prev(track_id, cx_norm, cy_norm)
            return False

        if zone != "mid":
            self._zone[track_id] = zone
        elif prev_zone == "entry":
            self._zone[track_id] = "mid"

        self._store_prev(track_id, cx_norm, cy_norm)
        return False

    def _mark_past(self, track_id: int) -> None:
        self._already_past.add(track_id)
        self._past_at.setdefault(track_id, self._frame_idx)

    def _small_follow(self, cx: float, cy: float, bw: float, bh: float) -> bool:
        """A narrow box just behind a goat that was counted a moment ago."""
        if bw <= 0 or bw >= 0.12:
            return False
        for px, py, frame in self._count_anchor:
            gap = self._frame_idx - frame
            # One frame later can be the next goat standing beside this one.
            if gap < 4 or gap > 12:
                continue
            if abs(cy - py) > 0.12:
                continue
            if self._is_upstream(cx, cy, px, py):
                return True
        return False

    def _register_count(self, track_id: int, cy: float | None = None) -> None:
        if track_id in self._counted:
            return
        self._counted.add(track_id)
        self.total += 1
        if cy is None:
            cy = self._last_cy.get(track_id, 0.5)
        self._recent_counts.append((cy, self._frame_idx))
        self._count_anchor.append((self._last_cx.get(track_id, 0.5), cy, self._frame_idx))

    def _recent_count_at(self, cy: float) -> bool:
        cutoff = self._frame_idx - self.RECENT_FRAMES
        self._recent_counts = [item for item in self._recent_counts if item[1] >= cutoff]
        return any(abs(prev_cy - cy) <= self.RECENT_DY for prev_cy, _ in self._recent_counts)

    def same_goat_id(
        self,
        track_id: int,
        cx: float,
        cy: float,
        bw: float = 0.0,
        bh: float = 0.0,
    ) -> int:
        """Keep the earlier ID when this box is the same goat a few frames later."""
        match = self._match_counted(cx, cy, bw, bh)
        if match is None or match == track_id:
            return track_id
        return match

    def _remember(self, track_id: int, cx: float, cy: float, bw: float, bh: float) -> None:
        self._last_cx[track_id] = cx
        self._last_cy[track_id] = cy
        self._last_bw[track_id] = bw
        self._last_bh[track_id] = bh
        self._last_frame[track_id] = self._frame_idx

    def _near_counted(self, cx: float, cy: float, bw: float, bh: float) -> bool:
        return self._match_counted(cx, cy, bw, bh) is not None

    def _match_counted(self, cx: float, cy: float, bw: float, bh: float) -> int | None:
        best_id: int | None = None
        best_dist = 999.0
        for track_id in self._counted:
            seen = self._last_frame.get(track_id)
            if seen is None:
                continue
            gap = self._frame_idx - seen
            if gap > 10:
                continue
            px = self._last_cx.get(track_id)
            py = self._last_cy.get(track_id)
            if px is None or py is None:
                continue
            # A box still upstream is the next goat, not this one walking on.
            if self._is_upstream(cx, cy, px, py):
                continue
            allowed = min(0.22, self.SAME_GOAT_DIST + 0.025 * gap)
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist <= allowed and dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _is_upstream(self, cx: float, cy: float, px: float, py: float) -> bool:
        """True when the new box is back toward the entry, behind a goat already seen."""
        if self.orientation == "vertical":
            if self.direction == "right_to_left":
                return cx > px + 0.08
            return cx < px - 0.08
        if self.direction == "down":
            return cy < py - 0.08
        return cy > py + 0.08

    def _jump_ok(self, track_id: int, cx: float, cy: float, bw: float, bh: float) -> bool:
        px = self._last_cx.get(track_id)
        py = self._last_cy.get(track_id)
        if px is None or py is None:
            return True
        if self._is_upstream(cx, cy, px, py):
            return False
        scale = max(bw, bh, self._last_bw.get(track_id, 0.0), self._last_bh.get(track_id, 0.0), 0.08)
        allowed = min(0.20, 0.7 * scale + 0.04)
        return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 <= allowed

    def _reaches_entry(self, cx: float, cy: float, bw: float, bh: float) -> bool:
        """True when a real box still covers the entry side, not a tiny sliver."""
        if max(bw, bh) < 0.12:
            return False
        if self.orientation == "vertical":
            half = bw / 2
            if self.direction == "right_to_left":
                overlap = (cx + half) - self.line_x
            else:
                overlap = self.line_x - (cx - half)
        else:
            half = bh / 2
            if self.direction == "down":
                overlap = self.line_y - (cy - half)
            else:
                overlap = (cy + half) - self.line_y
        return overlap >= 0.04

    def _line_distance(self, cx: float, cy: float) -> float:
        if self.orientation == "vertical":
            return abs(cx - self.line_x)
        return abs(cy - self.line_y)

    def _store_prev(self, track_id: int, cx: float, cy: float) -> None:
        self._prev_x[track_id] = cx
        self._prev_y[track_id] = cy

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
        if self.orientation == "horizontal":
            prev = self._prev_y.get(track_id)
            if prev is None:
                return False
            return self._crossed_horizontal(prev, cy)
        prev = self._prev_x.get(track_id)
        if prev is None:
            return False
        return self._crossed_vertical(prev, cx)

    def _crossed_horizontal(self, prev_y: float, curr_y: float) -> bool:
        line = self.line_y
        if self.direction == "down":
            return prev_y < line <= curr_y
        return prev_y > line >= curr_y

    def _crossed_vertical(self, prev_x: float, curr_x: float) -> bool:
        line = self.line_x
        if self.direction == "left_to_right":
            return prev_x < line <= curr_x
        return prev_x > line >= curr_x

    def _try_handoff_count(self, cy: float, new_track_id: int) -> bool:
        self._expire_pending()
        best_idx: int | None = None
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
        self._pending.pop(best_idx)
        self._register_count(new_track_id, cy)
        self._zone[new_track_id] = "exit"
        return True

    def _expire_pending(self) -> None:
        cutoff = self._frame_idx - self.HANDOFF_MAX_FRAMES
        self._pending = [
            pending
            for pending in self._pending
            if pending.frame >= cutoff and pending.track_id not in self._counted
        ]

    @property
    def counted_ids(self) -> list[int]:
        return sorted(self._counted)
