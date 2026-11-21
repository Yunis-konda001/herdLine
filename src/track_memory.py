"""Reuse one ID when a goat box flickers off for a few frames."""
from __future__ import annotations


class TrackMemory:
    """Match a new box to a recent goat by place so one animal keeps one ID."""

    def __init__(self, max_gap: int = 60, direction: str = "right_to_left") -> None:
        # max_gap: how many frames a goat can disappear before we forget its ID.
        self.max_gap = max_gap
        self.direction = direction
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    def assign(
        self, cx: float, cy: float, bw: float, bh: float, frame: int, conf: float = 1.0
    ) -> tuple[int, int]:
        # Remove goats that have been missing for too long.
        self._expire(frame)
        # Check if this new box matches a goat we've already seen recently.
        match = self._nearest(cx, cy, bw, bh, frame)
        
        if match is None:
            # No match found: this is a new goat. Create a new ID for it.
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "cx": cx, "cy": cy,   # Current position
                "vx": 0.0, "vy": 0.0, # Velocity (for prediction)
                "bw": bw, "bh": bh,   # Box size
                "frame": frame,
                "hits": 1,
                "conf_sum": float(conf),
            }
            return tid, 1

        # A match was found: the goat just flickered. Update its existing track.
        track = self._tracks[match]
        gap = max(1, frame - track["frame"])
        # Calculate the goat's velocity (how far it moved per frame).
        track["vx"] = (cx - track["cx"]) / gap
        track["vy"] = (cy - track["cy"]) / gap
        # Update the position and frame.
        track["cx"] = cx
        track["cy"] = cy
        track["bw"] = bw
        track["bh"] = bh
        track["frame"] = frame
        track["hits"] += 1
        track["conf_sum"] = float(track.get("conf_sum", 0.0)) + float(conf)
        return match, track["hits"]

    def mean_conf(self, track_id: int) -> float:
        # Calculate the average confidence for this goat's detections.
        track = self._tracks.get(track_id)
        if not track or track["hits"] <= 0:
            return 0.0
        return float(track.get("conf_sum", 0.0)) / float(track["hits"])

    def _nearest(self, cx: float, cy: float, bw: float, bh: float, frame: int) -> int | None:
        # Find the closest existing track to this new box.
        best_id: int | None = None
        best_d = 999.0
        for tid, track in self._tracks.items():
            gap = frame - track["frame"]
            # Ignore tracks that are too old to be the same goat.
            if gap <= 0 or gap > self.max_gap:
                continue
            
            # Predict where the goat should be now, based on its velocity.
            pred_x = track["cx"] + track["vx"] * gap
            pred_y = track["cy"] + track["vy"] * gap
            
            # Calculate how close the new box is to the predicted position.
            # Tight match so neighbouring goats in a crowd keep separate IDs
            scale = max(bw, track["bw"], bh, track["bh"], 0.06)
            allowed = min(0.10, 0.55 * scale + 0.006 * gap)
            dx = abs(cx - pred_x)
            dy = abs(cy - pred_y)
            
            # If the box is within the allowed distance, consider it a match.
            if dx <= allowed and dy <= allowed * 0.75:
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_d:
                    best_d = dist
                    best_id = tid
        return best_id

    def _expire(self, frame: int) -> None:
        # Delete tracks for goats that have been missing for longer than max_gap.
        stale = [tid for tid, track in self._tracks.items() if frame - track["frame"] > self.max_gap]
        for tid in stale:
            del self._tracks[tid]