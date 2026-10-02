"""Match detections without tracker IDs across frames (centroid proximity)."""
from __future__ import annotations


class SpatialTrackMatcher:
    """Assigns stable pseudo-IDs to detections when ByteTrack returns id=-1."""

    def __init__(self, max_dist: float = 0.12, max_gap_frames: int = 8) -> None:
        self.max_dist = max_dist
        self.max_gap_frames = max_gap_frames
        self._next_id = 10_000
        self._tracks: dict[int, tuple[float, float, int]] = {}

    def assign(self, cx: float, cy: float, frame: int) -> int:
        self._expire(frame)
        best_id: int | None = None
        best_d = self.max_dist + 1.0
        for tid, (tx, ty, _) in self._tracks.items():
            d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if d <= self.max_dist and d < best_d:
                best_d = d
                best_id = tid
        if best_id is not None:
            self._tracks[best_id] = (cx, cy, frame)
            return best_id
        pid = self._next_id
        self._next_id += 1
        self._tracks[pid] = (cx, cy, frame)
        return pid

    def _expire(self, frame: int) -> None:
        cutoff = frame - self.max_gap_frames
        stale = [tid for tid, (_, _, f) in self._tracks.items() if f < cutoff]
        for tid in stale:
            del self._tracks[tid]
