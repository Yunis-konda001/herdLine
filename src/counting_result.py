"""Result of a counting run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CountingResult:
    total_count: int
    frames_processed: int
    output_video: Path | None = None
    expected_count: int | None = None
    herd_size: int | None = None
    track_ids_counted: list[int] = field(default_factory=list)

    @property
    def reference_size(self) -> int | None:
        """Ground truth size used for accuracy (herd profile or expected count)."""
        if self.herd_size is not None:
            return self.herd_size
        return self.expected_count

    @property
    def counting_accuracy(self) -> float | None:
        ref = self.reference_size
        if ref is None or ref == 0:
            return None
        return max(0.0, 1.0 - abs(self.total_count - ref) / ref)

    @property
    def missing_count(self) -> int | None:
        if self.herd_size is None:
            return None
        return max(0, self.herd_size - self.total_count)

    @property
    def extra_count(self) -> int | None:
        if self.herd_size is None:
            return None
        return max(0, self.total_count - self.herd_size)

    @property
    def herd_match_status(self) -> str:
        if self.herd_size is None:
            return "unknown"
        if self.total_count < self.herd_size:
            return "missing"
        if self.total_count > self.herd_size:
            return "extra"
        return "complete"

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "frames_processed": self.frames_processed,
            "output_video": str(self.output_video) if self.output_video else None,
            "expected_count": self.expected_count,
            "herd_size": self.herd_size,
            "missing_count": self.missing_count,
            "extra_count": self.extra_count,
            "herd_match_status": self.herd_match_status,
            "counting_accuracy": self.counting_accuracy,
            "track_ids_counted": self.track_ids_counted,
        }
