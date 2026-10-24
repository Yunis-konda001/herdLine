"""One counting run: total, herd size, and report text."""
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
    session_name: str | None = None
    video_duration_sec: float | None = None
    mean_confidence: float | None = None

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

    @property
    def actual_count(self) -> int | None:
        """Number the user says should have crossed. Not estimated by the model."""
        return self.reference_size

    @property
    def correctly_counted(self) -> int | None:
        """How many of the actual goats the totals can agree on.

        A single total cannot tell a miss that was cancelled by an extra count.
        This is min(system, actual), not a per-goat check.
        """
        actual = self.actual_count
        if actual is None:
            return None
        return min(self.total_count, actual)

    @property
    def counting_accuracy_pct(self) -> float | None:
        """100 only when the system total matches the actual total.

        A higher count and a lower count both reduce this number.
        """
        actual = self.actual_count
        if actual is None or actual == 0:
            return None
        return max(0.0, 1.0 - abs(self.total_count - actual) / actual) * 100.0

    @property
    def missed_goats(self) -> int | None:
        actual = self.actual_count
        if actual is None:
            return None
        return max(0, actual - self.total_count)

    @property
    def extra_counts(self) -> int | None:
        """System total above the actual number. Not split into duplicate vs false."""
        actual = self.actual_count
        if actual is None:
            return None
        return max(0, self.total_count - actual)

    def report_lines(self) -> list[str]:
        lines = [
            f"Session: {self.session_name or 'unnamed'}",
            f"Actual goat count: {self._num(self.actual_count)}",
            f"System goat count: {self.total_count}",
            f"Missed goats: {self._num(self.missed_goats)}",
            f"Extra counts: {self._num(self.extra_counts)}",
            "Duplicate counts: not separated from extra counts",
            "False counts: not separated from extra counts",
        ]
        if self.counting_accuracy_pct is None:
            lines.append("Counting accuracy: not calculated (no actual count)")
        else:
            lines.append(f"Counting accuracy: {self.counting_accuracy_pct:.1f}%")
        if self.mean_confidence is None:
            lines.append("Average detection confidence: not calculated")
        else:
            lines.append(f"Average detection confidence: {self.mean_confidence:.3f}")
        if self.video_duration_sec is None:
            lines.append("Video duration: not calculated")
        else:
            lines.append(f"Video duration: {self.video_duration_sec:.1f} s")
        lines.append(
            "Extra counts are system minus actual. One total cannot say which of those were the same goat twice or a non-goat."
        )
        return lines

    @staticmethod
    def _num(value: int | None) -> str:
        return "not given" if value is None else str(value)

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
            "counting_accuracy_pct": self.counting_accuracy_pct,
            "missed_goats": self.missed_goats,
            "extra_counts": self.extra_counts,
            "mean_confidence": self.mean_confidence,
            "video_duration_sec": self.video_duration_sec,
            "session_name": self.session_name,
            "track_ids_counted": self.track_ids_counted,
        }
