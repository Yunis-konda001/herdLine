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
    assert c.update(1, 0.75, 0.5) is False
    c.end_frame()

    c.begin_frame(2)
    assert c.update(1, 0.45, 0.5) is True
    c.end_frame()
    assert c.total == 1


def test_id_handoff_after_swap():
    """Same goat: ID on entry side lost, new ID appears on exit side."""
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(3, 0.72, 0.4)
    c.end_frame()

    c.begin_frame(2)
    c.update(3, 0.68, 0.41)
    c.end_frame()

    c.begin_frame(3)
    assert c.update(7, 0.42, 0.42) is False
    c.end_frame()
    assert c.total == 1
    assert 7 in c.counted_ids


def test_same_frame_id_swap():
    c = _rtl_counter()
    c.begin_frame(1)
    c.update(3, 0.7, 0.5)
    c.end_frame()

    c.begin_frame(2)
    assert c.update(7, 0.4, 0.5) is False
    c.end_frame()
    assert c.total == 1


def test_no_double_count_stable_track():
    c = _rtl_counter()
    for i, x in enumerate([0.8, 0.6, 0.4, 0.3], start=1):
        c.begin_frame(i)
        c.update(1, x, 0.5)
        c.end_frame()
    assert c.total == 1
