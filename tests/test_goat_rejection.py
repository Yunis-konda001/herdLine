"""Ignore anything that's not a goat."""
from src.counting_pipeline import _has_goat_evidence
from src.counting_result import CountingResult
from src.detection_filters import select_goat_indices
from src.livestock_gate import looks_like_cattle_clip


def test_tall_person_box_is_filtered():
    person = [100.0, 40.0, 160.0, 220.0]  # h/w ≈ 2.5
    kept = select_goat_indices([person], [0.70], [0], frame_w=640, frame_h=360)
    assert kept == []


def test_side_goat_box_is_kept():
    goat = [100.0, 120.0, 220.0, 210.0]  # h/w ≈ 0.75
    kept = select_goat_indices([goat], [0.70], [0], frame_w=640, frame_h=360)
    assert kept == [0]


def test_no_detections_rejected():
    ok, reason = _has_goat_evidence(0, 0, 0.0, None, 0, 0)
    assert ok is False
    assert reason and "No goats detected" in reason


def test_people_video_evidence_rejected():
    ok, reason = _has_goat_evidence(
        kept_hits=30,
        strong_goat_hits=5,
        max_conf=0.70,
        mean_conf=0.50,
        person_like_hits=40,
        hog_person_hits=15,
    )
    assert ok is False
    assert reason is not None


def test_weak_false_positives_rejected():
    ok, reason = _has_goat_evidence(
        kept_hits=25,
        strong_goat_hits=8,
        max_conf=0.48,
        mean_conf=0.40,
        person_like_hits=0,
        hog_person_hits=0,
    )
    assert ok is False


def test_strong_goat_evidence_accepted():
    ok, reason = _has_goat_evidence(
        kept_hits=80,
        strong_goat_hits=40,
        max_conf=0.80,
        mean_conf=0.55,
        person_like_hits=2,
        hog_person_hits=0,
    )
    assert ok is True
    assert reason is None


def test_rejected_result_forces_zero_count():
    result = CountingResult(
        total_count=0,
        frames_processed=50,
        herd_size=14,
        rejection_reason="No clear goats found.",
    )
    assert result.total_count == 0
    assert result.herd_match_status == "rejected"


def test_cattle_density_rule_rejects_herd_like_clips():
    assert looks_like_cattle_clip(
        cattle_hits=40, samples=8, positive_frames=8, max_in_frame=7
    )
    assert looks_like_cattle_clip(
        cattle_hits=9, samples=4, positive_frames=4, max_in_frame=3
    )


def test_sparse_cow_false_labels_do_not_reject():
    # Goat clips can get a few COCO "cow" misfires — must not zero the count
    assert not looks_like_cattle_clip(
        cattle_hits=21, samples=13, positive_frames=10, max_in_frame=3
    )
    assert not looks_like_cattle_clip(
        cattle_hits=0, samples=10, positive_frames=0, max_in_frame=0
    )


def test_cattle_evidence_rejects_goat_count():
    ok, reason = _has_goat_evidence(
        kept_hits=80,
        strong_goat_hits=40,
        max_conf=0.80,
        mean_conf=0.55,
        person_like_hits=0,
        hog_person_hits=0,
        cattle_hits=48,
        cattle_samples=8,
        cattle_positive_frames=8,
        cattle_max_in_frame=7,
    )
    assert ok is False
    assert reason and "livestock" in reason.lower()
