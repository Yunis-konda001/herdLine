"""Counting report uses the entered actual count. It does not invent one."""
from src.counting_result import CountingResult


def test_accuracy_is_correct_over_actual():
    result = CountingResult(
        total_count=8,
        frames_processed=100,
        herd_size=10,
        mean_confidence=0.42,
        video_duration_sec=4.0,
        session_name="gate-a",
    )
    assert result.missed_goats == 2
    assert result.extra_counts == 0
    assert result.correctly_counted == 8
    assert result.counting_accuracy_pct == 80.0


def test_extra_counts_are_not_called_correct():
    result = CountingResult(total_count=12, frames_processed=50, herd_size=10)
    assert result.missed_goats == 0
    assert result.extra_counts == 2
    assert result.correctly_counted == 10
    assert result.counting_accuracy_pct == 80.0
    text = "\n".join(result.report_lines())
    assert "Duplicate counts: not separated" in text
    assert "False counts: not separated" in text


def test_no_actual_count_leaves_accuracy_blank():
    result = CountingResult(total_count=4, frames_processed=10)
    assert result.counting_accuracy_pct is None
    assert result.missed_goats is None
    text = "\n".join(result.report_lines())
    assert "not calculated" in text
    assert "not given" in text
