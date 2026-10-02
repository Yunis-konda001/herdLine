"""Virtual line crossing with zones, handoff, and late-detection exit debut."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _PendingHandoff:
    track_id: int
    cx: float
    cy: float
    frame: int


@dataclass
class _CountSite:
    cx: float
    cy: float
    frame: int


class LineCounter:
    """
    Counts unique passages across a virtual line.

    - entry/mid → exit zone transition (same track ID)
    - classic centroid line cross
    - ID handoff when tracker swaps at the line
    - exit debut when a track is first seen already past the line (late/low conf detect)
    """

    HANDOFF_CY_TOLERANCE = 0.14
    HANDOFF_MAX_FRAMES = 25
    EXIT_DEBUT_CY_MIN = 0.11
    EXIT_DEBUT_FRAME_GAP = 45
    EXIT_DEBUT_MIN_FRAMES = 5
    PSEUDO_ID_START = 10_000
    LINE_EPS = 0.012

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
        self.total = 0

        self._prev_x: dict[int, float] = {}
        self._prev_y: dict[int, float] = {}
        self._zone: dict[int, str] = {}
        self._last_cx: dict[int, float] = {}
        self._last_cy: dict[int, float] = {}

        self._frame_idx = 0
        self._ids_this_frame: set[int] = set()
        self._ids_prev_frame: set[int] = set()
        self._pending: list[_PendingHandoff] = []
        self._count_sites: list[_CountSite] = []
        self._exit_streak: dict[int, int] = {}

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
            if track_id in self._counted:
                continue
            if self._zone.get(track_id) in ("entry", "mid"):
                self._pending.append(
                    _PendingHandoff(
                        track_id=track_id,
                        cx=self._last_cx[track_id],
                        cy=self._last_cy[track_id],
                        frame=self._frame_idx,
                    )
                )

        new_ids = self._ids_this_frame - self._ids_prev_frame
        for track_id in new_ids:
            if track_id in self._counted:
                continue
            cy = self._last_cy.get(track_id)
            cx = self._last_cx.get(track_id)
            if cy is None or cx is None:
                continue
            if self._current_zone(cx, cy) == "exit":
                self._try_handoff_count(cy, track_id)

        self._ids_prev_frame = set(self._ids_this_frame)
        self._expire_pending()
        self._expire_count_sites()

    def update(self, track_id: int, cx_norm: float, cy_norm: float) -> bool:
        if track_id < 0:
            return False

        self._ids_this_frame.add(track_id)
        self._last_cx[track_id] = cx_norm
        self._last_cy[track_id] = cy_norm

        if track_id in self._counted:
            self._prev_x[track_id] = cx_norm
            self._prev_y[track_id] = cy_norm
            self._zone[track_id] = self._current_zone(cx_norm, cy_norm)
            return False

        zone = self._current_zone(cx_norm, cy_norm)
        prev_zone = self._zone.get(track_id)

        if zone == "exit":
            self._exit_streak[track_id] = self._exit_streak.get(track_id, 0) + 1
        else:
            self._exit_streak[track_id] = 0

        if prev_zone is None and zone == "exit":
            if self._try_handoff_count(cy_norm, track_id):
                return True

        if zone == "exit" and prev_zone in ("entry", "mid"):
            self._register_count(track_id, cx_norm, cy_norm)
            self._zone[track_id] = zone
            self._store_prev(track_id, cx_norm, cy_norm)
            return True

        if self._centroid_crossed(track_id, cx_norm, cy_norm):
            self._register_count(track_id, cx_norm, cy_norm)
            self._zone[track_id] = zone
            self._store_prev(track_id, cx_norm, cy_norm)
            return True

        if (
            zone == "exit"
            and track_id not in self._counted
            and track_id < self.PSEUDO_ID_START
            and self._exit_streak.get(track_id, 0) >= self.EXIT_DEBUT_MIN_FRAMES
            and prev_zone == "exit"
        ):
            if self._try_exit_debut_count(track_id, cx_norm, cy_norm):
                self._zone[track_id] = zone
                self._store_prev(track_id, cx_norm, cy_norm)
                return True

        if prev_zone is None:
            self._zone[track_id] = zone
        elif zone != "mid":
            self._zone[track_id] = zone
        elif prev_zone == "entry":
            self._zone[track_id] = "mid"

        self._store_prev(track_id, cx_norm, cy_norm)
        return False

    def _register_count(self, track_id: int, cx: float, cy: float) -> None:
        self._counted.add(track_id)
        self.total += 1
        self._count_sites.append(_CountSite(cx=cx, cy=cy, frame=self._frame_idx))

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
        for i, p in enumerate(self._pending):
            if p.track_id in self._counted:
                continue
            dy = abs(p.cy - cy)
            if dy <= self.HANDOFF_CY_TOLERANCE and dy < best_dy:
                best_dy = dy
                best_idx = i
        if best_idx is None:
            return False
        pending = self._pending.pop(best_idx)
        cx = self._last_cx.get(new_track_id, pending.cx)
        self._register_count(new_track_id, cx, cy)
        self._zone[new_track_id] = "exit"
        return True

    def _try_exit_debut_count(self, track_id: int, cx: float, cy: float) -> bool:
        """Late detection: stable real track on exit side, not yet counted."""
        if track_id >= self.PSEUDO_ID_START:
            return False
        if not self._is_distinct_passage(cy):
            return False
        self._register_count(track_id, cx, cy)
        self._zone[track_id] = "exit"
        return True

    def _is_distinct_passage(self, cy: float) -> bool:
        for site in self._count_sites:
            if abs(site.cy - cy) < self.EXIT_DEBUT_CY_MIN:
                if self._frame_idx - site.frame <= self.EXIT_DEBUT_FRAME_GAP:
                    return False
        return True

    def _expire_pending(self) -> None:
        cutoff = self._frame_idx - self.HANDOFF_MAX_FRAMES
        self._pending = [p for p in self._pending if p.frame >= cutoff and p.track_id not in self._counted]

    def _expire_count_sites(self) -> None:
        cutoff = self._frame_idx - self.EXIT_DEBUT_FRAME_GAP * 2
        self._count_sites = [s for s in self._count_sites if s.frame >= cutoff]

    @property
    def counted_ids(self) -> list[int]:
        return sorted(self._counted)
