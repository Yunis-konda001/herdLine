"""Tests for line-crossing counting (once per goat)."""
from src.line_counter import LineCounter


def _rtl_counter() -> LineCounter:
    return LineCounter(
        orientation="vertical",
        line_x=0.5,
        direction="right_to_left",
    )


def test_counts_only_when_line_is_crossed():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.75, 0.5, bw=0.20, bh=0.25) is False
    c.end_frame()
    assert c.total == 0

    c.begin_frame(2)
    assert c.update(1, 0.45, 0.5, bw=0.20, bh=0.25) is True
    c.end_frame()
    assert c.total == 1


def test_no_second_count_after_crossing():
    c = _rtl_counter()
    for frame, x in enumerate((0.7, 0.55, 0.4, 0.25, 0.15), start=1):
        c.begin_frame(frame)
        c.update(4, x, 0.5, bw=0.15, bh=0.20)
        c.end_frame()
    assert c.total == 1


def test_two_goats_must_each_cross():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.8, 0.2, bw=0.12, bh=0.16) is False
    assert c.update(2, 0.8, 0.8, bw=0.12, bh=0.16) is False
    c.end_frame()
    c.begin_frame(2)
    assert c.update(1, 0.4, 0.2, bw=0.12, bh=0.16) is True
    assert c.update(2, 0.4, 0.8, bw=0.12, bh=0.16) is True
    c.end_frame()
    assert c.total == 2


def test_already_past_the_line_is_not_counted():
    c = _rtl_counter()
    for frame, x in enumerate((0.2, 0.18, 0.15, 0.12, 0.1), start=1):
        c.begin_frame(frame)
        assert c.update(9, x, 0.4, bw=0.15, bh=0.20) is False
        c.end_frame()
    assert c.total == 0


def test_nearby_new_id_right_after_count_is_not_added():
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(1, 0.7, 0.5, bw=0.20, bh=0.25)
    c.end_frame()
    c.begin_frame(2)
    c.update(1, 0.4, 0.5, bw=0.20, bh=0.25)
    c.end_frame()
    c.begin_frame(3)
    c.update(2, 0.38, 0.51, bw=0.20, bh=0.25)
    c.end_frame()
    assert c.total == 1


def test_next_goat_on_different_path_still_counts():
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(1, 0.73, 0.57, bw=0.10, bh=0.16)
    c.end_frame()
    c.begin_frame(2)
    c.update(1, 0.40, 0.57, bw=0.10, bh=0.16)
    c.end_frame()
    c.begin_frame(3)
    c.update(2, 0.88, 0.20, bw=0.10, bh=0.16)
    c.end_frame()
    c.begin_frame(4)
    c.update(2, 0.40, 0.20, bw=0.10, bh=0.16)
    c.end_frame()
    assert c.total == 2
