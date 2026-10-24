"""Tests for line crossing and ID handoff counting."""
from src.line_counter import LineCounter


def _rtl_counter() -> LineCounter:
    return LineCounter(
        orientation="vertical",
        line_x=0.5,
        direction="right_to_left",
    )


def test_single_track_zone_cross():
    c = _rtl_counter()
    c.begin_frame(1)
    # First sight on entry: watch only, do not count yet
    assert c.update(1, 0.75, 0.5) is False
    c.end_frame()
    assert c.total == 0

    c.begin_frame(2)
    assert c.update(1, 0.45, 0.5) is True
    c.end_frame()
    assert c.total == 1


def test_id_swap_without_crossing_does_not_count():
    """New ID past the line is not a crossing (handoff disabled to avoid overcount)."""
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(3, 0.62, 0.4)
    c.end_frame()

    c.begin_frame(2)
    c.update(3, 0.56, 0.41)
    c.end_frame()

    c.begin_frame(3)
    assert c.update(7, 0.42, 0.42) is False
    c.end_frame()
    assert c.total == 0


def test_new_id_on_exit_is_not_counted():
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(3, 0.58, 0.5)
    c.end_frame()

    c.begin_frame(2)
    assert c.update(7, 0.42, 0.5) is False
    c.end_frame()
    assert c.total == 0


def test_late_lock_just_past_the_line_counts_once():
    # Already past the line, with no earlier sighting: not a crossing.
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(5, 0.36, 0.3) is False
    c.end_frame()
    assert c.total == 0


def test_new_id_after_a_count_is_not_added():
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(1, 0.7, 0.5)
    c.end_frame()
    c.begin_frame(2)
    c.update(1, 0.4, 0.5)
    c.end_frame()
    c.begin_frame(3)
    c.update(2, 0.36, 0.5)
    c.end_frame()
    assert c.total == 1


def test_already_past_the_line_is_not_counted():
    c = _rtl_counter()
    for frame, x in enumerate((0.2, 0.18, 0.15, 0.12, 0.1, 0.08), start=1):
        c.begin_frame(frame)
        assert c.update(9, x, 0.4) is False
        c.end_frame()
    assert c.total == 0


def test_no_second_count_after_crossing():
    c = _rtl_counter()
    for frame, x in enumerate((0.7, 0.55, 0.4, 0.25, 0.15), start=1):
        c.begin_frame(frame)
        c.update(4, x, 0.5)
        c.end_frame()
    assert c.total == 1


def test_id_change_a_few_frames_later_is_not_a_new_goat():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.75, 0.54, bw=0.30, bh=0.35) is False
    c.end_frame()
    c.begin_frame(2)
    assert c.update(1, 0.45, 0.54, bw=0.30, bh=0.35) is True
    c.end_frame()
    c.begin_frame(5)
    assert c.update(9, 0.40, 0.60, bw=0.30, bh=0.35) is False
    c.end_frame()
    assert c.total == 1


def test_entry_sight_alone_does_not_count():
    # Background / waiting goats on the entry side must not inflate the total.
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(4, 0.72, 0.55, bw=0.20, bh=0.25) is False
    c.end_frame()
    assert c.total == 0


def test_next_goat_crossing_still_counts():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.73, 0.57, bw=0.10, bh=0.16) is False
    c.end_frame()
    c.begin_frame(2)
    assert c.update(1, 0.40, 0.57, bw=0.10, bh=0.16) is True
    c.end_frame()
    # Second goat on a different path (different y) so it is not treated as the same box
    c.begin_frame(3)
    assert c.update(2, 0.88, 0.20, bw=0.10, bh=0.16) is False
    c.end_frame()
    c.begin_frame(4)
    assert c.update(2, 0.40, 0.20, bw=0.10, bh=0.16) is True
    c.end_frame()
    assert c.total == 2


def test_two_goats_must_each_cross():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.8, 0.2) is False
    assert c.update(2, 0.8, 0.8) is False
    c.end_frame()
    assert c.total == 0
    c.begin_frame(2)
    assert c.update(1, 0.4, 0.2) is True
    assert c.update(2, 0.4, 0.8) is True
    c.end_frame()
    assert c.total == 2


def test_second_box_on_the_same_goat_does_not_count():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.75, 0.50) is False
    c.end_frame()
    c.begin_frame(2)
    assert c.update(1, 0.45, 0.50) is True
    assert c.update(2, 0.48, 0.52) is False
    c.end_frame()
    assert c.total == 1


def test_tiny_box_is_not_a_goat():
    c = _rtl_counter()
    c.begin_frame(1)
    assert c.update(1, 0.7, 0.8, bw=0.03, bh=0.04) is False
    c.end_frame()
    c.begin_frame(2)
    assert c.update(1, 0.4, 0.8, bw=0.03, bh=0.04) is False
    c.end_frame()
    assert c.total == 0


def test_no_double_count_stable_track():
    c = _rtl_counter()
    for i, x in enumerate([0.8, 0.6, 0.4, 0.3], start=1):
        c.begin_frame(i)
        c.update(1, x, 0.5)
        c.end_frame()
    assert c.total == 1
