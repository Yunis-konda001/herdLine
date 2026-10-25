"""Reuse one ID when a goat box flickers off for a few frames."""
from __future__ import annotations


class TrackMemory:
    """Match a new box to a recent goat by place so one animal keeps one ID."""

    def __init__(self, max_gap: int = 60, direction: str = "right_to_left") -> None:
        self.max_gap = max_gap
        self.direction = direction
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    def assign(
        self, cx: float, cy: float, bw: float, bh: float, frame: int, conf: float = 1.0
    ) -> tuple[int, int]:
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
                "conf_sum": float(conf),
            }
            return tid, 1

        track = self._tracks[match]
        gap = max(1, frame - track["frame"])
        track["vx"] = (cx - track["cx"]) / gap
        track["vy"] = (cy - track["cy"]) / gap
        track["cx"] = cx
        track["cy"] = cy
        track["bw"] = bw
        track["bh"] = bh
        track["frame"] = frame
        track["hits"] += 1
        track["conf_sum"] = float(track.get("conf_sum", 0.0)) + float(conf)
        return match, track["hits"]

    def mean_conf(self, track_id: int) -> float:
        track = self._tracks.get(track_id)
        if not track or track["hits"] <= 0:
            return 0.0
        return float(track.get("conf_sum", 0.0)) / float(track["hits"])

    def _nearest(self, cx: float, cy: float, bw: float, bh: float, frame: int) -> int | None:
        best_id: int | None = None
        best_d = 999.0
        for tid, track in self._tracks.items():
            gap = frame - track["frame"]
            if gap <= 0 or gap > self.max_gap:
                continue
            pred_x = track["cx"] + track["vx"] * gap
            pred_y = track["cy"] + track["vy"] * gap
            scale = max(bw, track["bw"], bh, track["bh"], 0.08)
            allowed = min(0.30, 1.0 * scale + 0.012 * gap)
            dx = abs(cx - pred_x)
            dy = abs(cy - pred_y)
            if dx <= allowed and dy <= allowed * 0.85:
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_d:
                    best_d = dist
                    best_id = tid
        return best_id

    def _expire(self, frame: int) -> None:
        stale = [tid for tid, track in self._tracks.items() if frame - track["frame"] > self.max_gap]
        for tid in stale:
            del self._tracks[tid]
