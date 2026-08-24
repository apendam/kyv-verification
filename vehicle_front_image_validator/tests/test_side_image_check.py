import pytest

from vfiv.validators.side_image_check import _worst_decision, classify_axle_count, decide_axle_count


def test_worst_decision_picks_reject_over_anything():
    assert _worst_decision("PASS", "MANUAL_REVIEW", "REJECT") == "REJECT"


def test_worst_decision_picks_manual_review_over_pass():
    assert _worst_decision("PASS", "MANUAL_REVIEW") == "MANUAL_REVIEW"


def test_worst_decision_all_pass_is_pass():
    assert _worst_decision("PASS", "PASS") == "PASS"


def _axle_read(count, confidence=90.0, lift=False):
    return {"checked": True, "axle_count": count, "axle_confidence": confidence,
            "lift_axle_suspected": lift, "reason": "test"}


def test_axle_count_matches_claimed_passes():
    result = decide_axle_count(_axle_read(3), claimed_axle_count=3)
    assert result["decision"] == "PASS"
    assert result["status"] == "MATCH"


def test_axle_count_mismatch_rejects():
    result = decide_axle_count(_axle_read(2), claimed_axle_count=3)
    assert result["decision"] == "REJECT"
    assert result["status"] == "MISMATCH"


def test_axle_count_low_confidence_is_manual_review_regardless_of_match():
    """Even a numerically-correct count isn't trusted below the confidence floor."""
    result = decide_axle_count(_axle_read(3, confidence=40.0), claimed_axle_count=3)
    assert result["decision"] == "MANUAL_REVIEW"
    assert result["status"] == "UNREADABLE"


def test_lift_axle_suspected_still_passes_but_notes_it():
    result = decide_axle_count(_axle_read(3, lift=True), claimed_axle_count=3)
    assert result["decision"] == "PASS"
    assert "lift axle" in result["reason"]


def test_unknown_axle_backend_raises_before_any_image_io():
    """The if/elif/else dispatch means an unknown backend never reaches
    call_vlm_json/call_gemini_json -- raises immediately, even for a path that
    doesn't exist."""
    with pytest.raises(ValueError, match="unknown axle-count backend"):
        classify_axle_count("does-not-exist.jpg", backend="not-a-real-backend")
