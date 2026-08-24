import os

import pytest

from vfiv.backends.vector_store import DuplicateMatch
from vfiv.validators.duplicate_check import check_duplicate, decide_duplicate

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_PGVECTOR = bool(os.environ.get("VFIV_PGVECTOR_DSN"))
requires_pgvector = pytest.mark.skipif(not HAS_PGVECTOR, reason="requires a live pgvector DSN")


def _match(upload_id, claimed_vrn, similarity, image_type="front"):
    return DuplicateMatch(upload_id=upload_id, claimed_vrn=claimed_vrn,
                          image_type=image_type, similarity=similarity)


def test_no_prior_uploads_passes():
    result = decide_duplicate([], claimed_vrn="UP42T4069")
    assert result.decision == "PASS"
    assert result.is_duplicate_suspect is False
    assert result.best_match_id is None


def test_near_duplicate_under_same_vrn_is_not_flagged():
    """An honest re-upload of the same claim is expected, not fraud."""
    matches = [_match("img_1", "UP42T4069", 0.999)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069")
    assert result.decision == "PASS"
    assert result.is_duplicate_suspect is False
    assert result.best_match_id == "img_1"


def test_vrn_comparison_ignores_case_and_spacing():
    """Same underlying VRN, formatted differently -- normalize_vrn should treat it
    as the SAME vrn, not a mismatch worth flagging."""
    matches = [_match("img_1", "up 42 t 4069", 0.99)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069")
    assert result.is_duplicate_suspect is False


def test_near_duplicate_under_different_vrn_flags_manual_review():
    matches = [_match("img_1", "MH12AB1234", 0.985)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069", similarity_min=0.97)
    assert result.decision == "MANUAL_REVIEW"
    assert result.is_duplicate_suspect is True
    assert result.best_match_id == "img_1"
    assert result.best_match_vrn == "MH12AB1234"
    assert result.best_match_similarity == 0.985


def test_dissimilar_match_below_threshold_is_not_flagged():
    matches = [_match("img_1", "MH12AB1234", 0.6)]
    result = decide_duplicate(matches, claimed_vrn="UP42T4069", similarity_min=0.97)
    assert result.decision == "PASS"
    assert result.is_duplicate_suspect is False


def test_picks_the_highest_similarity_suspect_among_several():
    matches = [
        _match("img_low", "MH12AB1234", 0.98),
        _match("img_high", "DL01ZZ9999", 0.995),
        _match("img_same_vrn", "UP42T4069", 0.999),
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

    def _fake_find_similar(embedding, image_type, top_k=5):
        raise DuplicateStoreError("pgvector not configured")

    monkeypatch.setattr(duplicate_check_module, "find_similar", _fake_find_similar)

    result = check_duplicate("does-not-matter.jpg", upload_id="img_x", claimed_vrn="UP42T4069")
    assert result.checked is False
    assert result.decision == "MANUAL_REVIEW"
    assert result.is_duplicate_suspect is False


def test_image_type_defaults_to_front_and_is_passed_through(monkeypatch):
    """The image_type argument must reach find_similar/store_embedding unchanged
    -- this is what keeps front/side/fastag from ever being compared against
    each other."""
    import vfiv.validators.duplicate_check as duplicate_check_module

    seen = {}

    class _FakeSiglip:
        def embed_image(self, image):
            return "fake-embedding"

    def _fake_find_similar(embedding, image_type, top_k=5):
        seen["find_image_type"] = image_type
        return []

    def _fake_store_embedding(upload_id, claimed_vrn, image_type, embedding):
        seen["store_image_type"] = image_type

    monkeypatch.setattr(duplicate_check_module, "get_siglip_model", lambda: _FakeSiglip())
    monkeypatch.setattr(duplicate_check_module, "find_similar", _fake_find_similar)
    monkeypatch.setattr(duplicate_check_module, "store_embedding", _fake_store_embedding)

    check_duplicate("does-not-matter.jpg", upload_id="img_x", claimed_vrn="UP42T4069",
                    image_type="fastag")
    assert seen["find_image_type"] == "fastag"
    assert seen["store_image_type"] == "fastag"


@requires_pgvector
def test_stores_and_finds_itself_as_a_near_duplicate():
    """Live smoke test against a real pgvector DSN: storing an upload's own
    embedding, then querying it back, should surface itself as a near-perfect
    match. Requires VFIV_PGVECTOR_DSN and the sample images."""
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = check_duplicate(path, upload_id="test_truck2", claimed_vrn="UP42T4069", image_type="front")
    assert result.checked is True
