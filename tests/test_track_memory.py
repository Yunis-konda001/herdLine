"""A box that vanishes and comes back nearby keeps the same ID."""
from src.track_memory import TrackMemory


def test_gap_reuses_the_same_id():
    memory = TrackMemory()
    first, hits = memory.assign(0.70, 0.55, 0.20, 0.30, frame=1)
    assert hits == 1
    again, hits = memory.assign(0.62, 0.56, 0.20, 0.30, frame=6)
    assert again == first
    assert hits == 2


def test_next_goat_waiting_upstream_is_new():
    memory = TrackMemory(direction="right_to_left")
    first, _ = memory.assign(0.40, 0.55, 0.20, 0.30, frame=1)
    memory.assign(0.36, 0.55, 0.20, 0.30, frame=2)
    other, hits = memory.assign(0.70, 0.56, 0.20, 0.30, frame=3)
    assert other != first
    assert hits == 1


def test_a_goat_in_another_lane_is_new():
    memory = TrackMemory()
    first, _ = memory.assign(0.70, 0.30, 0.18, 0.24, frame=1)
    other, hits = memory.assign(0.70, 0.75, 0.18, 0.24, frame=2)
    assert other != first
    assert hits == 1
