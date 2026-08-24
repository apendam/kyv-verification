import os

import pytest

from vfiv.backends.vector_store import DuplicateMatch
from vfiv.validators.duplicate_check import check_duplicate, decide_duplicate

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_PGVECTOR = bool(os.environ.get("VFIV_PGVECTOR_DSN"))
requires_pgvector = pytest.mark.skipif(not HAS_PGVECTOR, reason="requires a live pgvector DSN")


def test_no_prior_uploads_passes():
    result = decide_duplicate([], claimed_vrn="UP42T4069")
    assert result.decision == "PASS"
    assert result.is_duplicate_suspect is False
    assert result.best_match_id is None


def test_near_duplicate_under_same_vrn_is_not_flagged():
    """An honest re-upload of the same claim is expected, not fraud."""
    matches = [DuplicateMatch(upload_id="img_1", claimed_vrn="UP42T4069", similarity=0.999)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069")
    assert result.decision == "PASS"
    assert result.is_duplicate_suspect is False
    assert result.best_match_id == "img_1"


def test_vrn_comparison_ignores_case_and_spacing():
    """Same underlying VRN, formatted differently -- normalize_vrn should treat it
    as the SAME vrn, not a mismatch worth flagging."""
    matches = [DuplicateMatch(upload_id="img_1", claimed_vrn="up 42 t 4069", similarity=0.99)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069")
    assert result.is_duplicate_suspect is False


def test_near_duplicate_under_different_vrn_flags_manual_review():
    matches = [DuplicateMatch(upload_id="img_1", claimed_vrn="MH12AB1234", similarity=0.985)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069", similarity_min=0.97)
    assert result.decision == "MANUAL_REVIEW"
    assert result.is_duplicate_suspect is True
    assert result.best_match_id == "img_1"
    assert result.best_match_vrn == "MH12AB1234"
    assert result.best_match_similarity == 0.985


def test_dissimilar_match_below_threshold_is_not_flagged():
    matches = [DuplicateMatch(upload_id="img_1", claimed_vrn="MH12AB1234", similarity=0.6)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069", similarity_min=0.97)
    assert result.decision == "PASS"
    assert result.is_duplicate_suspect is False


def test_picks_the_highest_similarity_suspect_among_several():
    matches = [
        DuplicateMatch(upload_id="img_low", claimed_vrn="MH12AB1234", similarity=0.98),
        DuplicateMatch(upload_id="img_high", claimed_vrn="DL01ZZ9999", similarity=0.995),
        DuplicateMatch(upload_id="img_same_vrn", claimed_vrn="UP42T4069", similarity=0.999),
    ]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069", similarity_min=0.97)
    assert result.best_match_id == "img_high"
    assert result.best_match_similarity == 0.995


def test_manual_review_when_pgvector_not_configured(monkeypatch):
    """Backend not configured (no VFIV_PGVECTOR_DSN) -> checked=False, surfaced as
    MANUAL_REVIEW rather than crashing -- same posture as the AWS-credential-error
    path in vrn_check.py."""
    import vfiv.validators.duplicate_check as duplicate_check_module
    from vfiv.backends.vector_store import DuplicateStoreError

    class _FakeSiglip:
        def embed_image(self, image):
            return object()  # never reaches find_similar in this test

    monkeypatch.setattr(duplicate_check_module, "get_siglip_model", lambda: _FakeSiglip())

    def _fake_find_similar(embedding, top_k=5):
        raise DuplicateStoreError("pgvector not configured")

    monkeypatch.setattr(duplicate_check_module, "find_similar", _fake_find_similar)

    result = check_duplicate("does-not-matter.jpg", upload_id="img_x", claimed_vrn="UP42T4069")
    assert result.checked is False
    assert result.decision == "MANUAL_REVIEW"
    assert result.is_duplicate_suspect is False


@requires_pgvector
def test_stores_and_finds_itself_as_a_near_duplicate():
    """Live smoke test against a real pgvector DSN: storing an upload's own
    embedding, then querying it back, should surface itself as a near-perfect
    match. Requires VFIV_PGVECTOR_DSN and the sample images."""
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = check_duplicate(path, upload_id="test_truck2", claimed_vrn="UP42T4069")
    assert result.checked is True
