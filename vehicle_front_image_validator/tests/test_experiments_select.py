"""Tests for the experiments/ backend-selection layer: invalid-backend errors,
Q3's generalized multi-vote decision logic, and (live) the reconstructed legacy
all-Claude Q1/Q2/Q3 prompts against real samples."""
import os

import pytest

from vfiv.experiments import q1_select, q2_select, q3_select

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
requires_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="requires ANTHROPIC_API_KEY")


def test_q1_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown Q1 backend"):
        q1_select.classify_q1("does-not-matter.jpg", backend="not-a-real-backend")


def test_q2_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown Q2 backend"):
        q2_select.classify_q2("does-not-matter.jpg", backend="not-a-real-backend")


def test_q3_unknown_make_backend_raises():
    with pytest.raises(ValueError, match="unknown Q3 make backend"):
        q3_select.collect_make_votes("does-not-matter.jpg", backend="not-a-real-backend")


def test_q3_unknown_model_backend_raises():
    with pytest.raises(ValueError, match="unknown Q3 model backend"):
        q3_select.classify_model("does-not-matter.jpg", backend="not-a-real-backend")


def test_decide_make_multi_no_votes_is_unreadable():
    result = q3_select.decide_make_multi([], claimed_make="TATA MOTORS LTD")
    assert result["make_status"] == "UNREADABLE"
    assert result["matched_via"] is None
    assert result["errors"] == []


def test_decide_make_multi_surfaces_backend_errors():
    """A genuine backend failure (bad/unenabled credentials, network error) must be
    distinguishable from "checked fine, found nothing" — not silently identical."""
    errors = [{"source": "gcv_logo", "error": "403 Cloud Vision API has not been used..."}]
    result = q3_select.decide_make_multi([], claimed_make="TATA MOTORS LTD", errors=errors)
    assert result["make_status"] == "UNREADABLE"
    assert result["errors"] == errors


def test_collect_make_votes_gcv_backend_failure_returns_error_not_silent_empty(monkeypatch):
    """Mirrors the real Cloud-Vision-API-disabled case: the backend call fails
    (checked=False) — must come back as an error, not indistinguishable from a
    successful-but-empty read."""
    def fake_classify_logo(image):
        return {"checked": False, "error": "403 Cloud Vision API has not been used..."}

    monkeypatch.setattr(q3_select.google_vision, "classify_logo", fake_classify_logo)
    votes, errors = q3_select.collect_make_votes("does-not-matter.jpg", backend="gcv_logo")
    assert votes == []
    assert errors == [{"source": "gcv_logo", "error": "403 Cloud Vision API has not been used..."}]


def test_decide_make_multi_single_vote_matches():
    votes = [("claude", "TATA", 97.0)]
    result = q3_select.decide_make_multi(votes, claimed_make="TATA MOTORS LTD")
    assert result["make_status"] == "MATCH"
    assert result["matched_via"] == ["claude"]


def test_decide_make_multi_two_votes_one_matches():
    """Mirrors the real truck2.jpg case: one source misreads, the other is correct —
    matched via whichever source got it right."""
    votes = [("siglip", "Force Motors", 20.0), ("rekognition", "TATA", 100.0)]
    result = q3_select.decide_make_multi(votes, claimed_make="TATA MOTORS LTD")
    assert result["make_status"] == "MATCH"
    assert result["matched_via"] == ["rekognition"]
    assert len(result["votes"]) == 2


def test_decide_make_multi_all_votes_mismatch():
    votes = [("claude", "TATA", 97.0), ("gemini", "TATA", 90.0)]
    result = q3_select.decide_make_multi(votes, claimed_make="ASHOK LEYLAND LTD")
    assert result["make_status"] == "MISMATCH"
    assert result["matched_via"] is None


@requires_anthropic
def test_legacy_claude_q1_matches_real_sample():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    r = q1_select.classify_q1(path, backend="claude")
    assert r["checked"] is True
    assert r["vehicle_type"] == "truck"
    assert r["is_front"] is True


@requires_anthropic
def test_legacy_claude_q2_reads_known_vrn():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    r = q2_select.classify_q2(path, backend="claude")
    assert r["checked"] is True
    assert r["plate"] == "UP42T4069"


@requires_anthropic
def test_legacy_claude_q3_make_votes_match():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    votes, errors = q3_select.collect_make_votes(path, backend="claude")
    assert errors == []
    assert len(votes) == 1
    assert votes[0][0] == "claude"
    assert votes[0][1] == "TATA"
