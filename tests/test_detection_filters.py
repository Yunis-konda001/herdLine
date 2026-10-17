"""Two boxes on one goat should collapse to the stronger box."""
from src.detection_filters import _drop_overlapping


def test_head_box_inside_body_box_keeps_one():
    body = [100.0, 80.0, 420.0, 360.0]
    head = [150.0, 90.0, 250.0, 160.0]
    kept = _drop_overlapping([0, 1], [body, head], [0.17, 0.16])
    assert kept == [0]


def test_two_goats_side_by_side_both_stay():
    left = [20.0, 40.0, 90.0, 140.0]
    right = [220.0, 40.0, 300.0, 140.0]
    kept = _drop_overlapping([0, 1], [left, right], [0.4, 0.5])
    assert sorted(kept) == [0, 1]
