"""Keep one temporary ID when a goat box flickers off for a few frames.

ByteTrack often starts a new ID after a miss. That new ID was counted again
and made totals too high. TrackMemory links the returning box to the old ID.
"""
from __future__ import annotations


class TrackMemory:
    """Match a new detection to a recent goat by position and travel direction."""

    def __init__(self, max_gap: int = 10, direction: str = "right_to_left") -> None:
        self.max_gap = max_gap
        self.direction = direction
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    def assign(self, cx: float, cy: float, bw: float, bh: float, frame: int) -> tuple[int, int]:
        """Return (stable id, times this goat has been seen)."""
        self._expire(frame)
        match = self._nearest(cx, cy, bw, bh, frame)
        if match is None:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "cx": cx,
                "cy": cy,
                "vx": 0.0,
                "vy": 0.0,
                "bw": bw,
                "bh": bh,
                "frame": frame,
                "hits": 1,
            }
            return tid, 1

        track = self._tracks[match]
        gap = max(1, frame - track["frame"])
        track["vx"], track["vy"] = self._clamp_velocity(
            (cx - track["cx"]) / gap,
            (cy - track["cy"]) / gap,
        )
        track["cx"] = cx
        track["cy"] = cy
        track["bw"] = bw
        track["bh"] = bh
        track["frame"] = frame
        track["hits"] += 1
        return match, track["hits"]

    def ghosts(self, frame: int) -> list[tuple[float, float, float, float, int]]:
        """Boxes to keep drawing while the detector misses for a few frames."""
        held: list[tuple[float, float, float, float, int]] = []
        for tid, track in self._tracks.items():
            gap = frame - track["frame"]
            if gap <= 0 or gap > 8 or track["hits"] < 2:
                continue
            cx = track["cx"] + track["vx"] * gap
            cy = track["cy"] + track["vy"] * gap
            held.append((cx, cy, track["bw"], track["bh"], tid))
        return held

    def _nearest(self, cx: float, cy: float, bw: float, bh: float, frame: int) -> int | None:
        best_id: int | None = None
        best_d = 999.0
        for tid, track in self._tracks.items():
            gap = frame - track["frame"]
            if gap <= 0 or gap > self.max_gap:
                continue
            pred_x = track["cx"] + track["vx"] * gap
            pred_y = track["cy"] + track["vy"] * gap
            # A box back toward the gate is the next goat, not this one.
            if self._upstream(cx, cy, track["cx"], track["cy"]):
                continue
            scale_x = max(bw, track["bw"], 0.1)
            scale_y = max(bh, track["bh"], 0.1)
            allowed_x = min(0.28, 0.60 * scale_x + 0.025 * gap)
            allowed_y = min(0.11, 0.40 * scale_y + 0.012 * gap)
            dx = abs(cx - pred_x)
            dy = abs(cy - pred_y)
            if dx <= allowed_x and dy <= allowed_y:
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_d:
                    best_d = dist
                    best_id = tid
        return best_id

    def _upstream(self, cx: float, cy: float, px: float, py: float) -> bool:
        """True when the new box is back toward the entrance."""
        slack = 0.08
        if self.direction == "right_to_left":
            return cx > px + slack
        if self.direction == "left_to_right":
            return cx < px - slack
        if self.direction == "down":
            return cy < py - slack
        if self.direction == "up":
            return cy > py + slack
        return False

    def _clamp_velocity(self, vx: float, vy: float) -> tuple[float, float]:
        """Do not predict a goat walking back toward the entrance."""
        if self.direction == "right_to_left":
            return min(0.0, vx), vy
        if self.direction == "left_to_right":
            return max(0.0, vx), vy
        if self.direction == "down":
            return vx, max(0.0, vy)
        if self.direction == "up":
            return vx, min(0.0, vy)
        return vx, vy

    def _expire(self, frame: int) -> None:
        stale = [tid for tid, track in self._tracks.items() if frame - track["frame"] > self.max_gap]
        for tid in stale:
            del self._tracks[tid]
