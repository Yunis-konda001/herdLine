"""Fixed-line evaluation: count then compare (no herd-size line fitting)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

from src.config import (
    DEFAULT_CROSSING_DIRECTION,
    DEFAULT_LINE_X,
    LINE_SEARCH_MODE,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from src.counting_pipeline import CountingPipeline

VIDEOS = [
    ("test_video_4_goats.mp4", 4),
    ("test_video_5_goats.mp4", 5),
    ("test_video_6_goats.mp4", 6),
    ("test_video_8_goats.mp4", 8),
    ("test_video_12_goats.mp4", 12),
    ("test_video_17_goats.mp4", 17),
]


def main() -> None:
    print("LINE_SEARCH_MODE", LINE_SEARCH_MODE)
    print("fixed line_x", DEFAULT_LINE_X, "direction", DEFAULT_CROSSING_DIRECTION)
    pipe = CountingPipeline()
    rows = []
    eval_dir = PROJECT_ROOT / "data" / "videos" / "evaluation"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, expected in VIDEOS:
        path = eval_dir / name
        t0 = time.perf_counter()
        result = pipe.run(
            path, herd_size=expected, draw=False, session_name=path.stem
        )
        elapsed = time.perf_counter() - t0
        system = result.total_count
        err = system - expected
        acc = max(0.0, 1.0 - abs(err) / expected) if expected else 0.0
        row = {
            "video": name,
            "expected": expected,
            "system": system,
            "error": err,
            "misses": max(0, expected - system),
            "extras": max(0, system - expected),
            "accuracy": round(acc, 3),
            "time_sec": round(elapsed, 1),
            "line_x": DEFAULT_LINE_X,
            "direction": DEFAULT_CROSSING_DIRECTION,
            "rejected": bool(result.rejection_reason),
            "rejection_reason": result.rejection_reason or "",
        }
        rows.append(row)
        print(
            f"{name}: sys={system} exp={expected} err={err:+d} "
            f"miss={row['misses']} acc={acc:.3f} time={elapsed:.1f}s"
        )

    out = RESULTS_DIR / "fixed_line_eval.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)
    print()
    print("| Video | Expected | System | Error | Misses | Extras | Accuracy | Time (s) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['video']} | {row['expected']} | {row['system']} | "
            f"{row['error']:+d} | {row['misses']} | {row['extras']} | "
            f"{row['accuracy']:.3f} | {row['time_sec']} |"
        )


if __name__ == "__main__":
    main()
