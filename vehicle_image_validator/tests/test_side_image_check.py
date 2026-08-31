import pytest

from vfiv.side_image.side_image_check import (
    _worst_decision,
    check_axle_count,
    check_side_identity,
    classify_axle_count,
    decide_axle_count,
)


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


def test_check_axle_count_wraps_classify_and_decide(monkeypatch):
    """check_axle_count is the standalone (webapp bucket) entry point -- must
    return the same decision classify+decide would, packaged as AxleCountResult."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "classify_axle_count",
                        lambda image, backend, model=None: _axle_read(3))

    result = check_axle_count("does-not-matter.jpg", claimed_axle_count=3)
    assert result.decision == "PASS"
    assert result.status == "MATCH"
    assert result.checked is True
    assert result.axle_count == 3


def test_check_axle_count_degrades_on_classify_exception(monkeypatch):
    """A raised exception (e.g. an unknown/misconfigured backend) must degrade to
    MANUAL_REVIEW, not crash the caller -- same posture as the rest of the module."""
    import vfiv.side_image.side_image_check as side_module

    def _boom(image, backend, model=None):
        raise ValueError("unknown axle-count backend: 'bogus'")

    monkeypatch.setattr(side_module, "classify_axle_count", _boom)

    result = check_axle_count("does-not-matter.jpg", claimed_axle_count=3, backend="bogus")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert "unknown axle-count backend" in result.reason


def test_check_side_identity_routes_to_vrn_visible_bucket(monkeypatch):
    """When the type-classifier says vrn_visible, check_side_identity must dispatch
    to the VRN-based identity check, not the corner/pure-side-profile arms."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "load_rgb_array", lambda image: "fake-array")
    monkeypatch.setattr(side_module, "get_side_image_type_classifier",
                        lambda: type("_C", (), {"predict": lambda self, arr: {"bucket": "vrn_visible"}})())
    monkeypatch.setattr(side_module, "_identity_via_vrn",
                        lambda image, claimed_vrn: ("PASS", "[vrn_visible] matched", {"bucket": "vrn_visible"}))

    result = check_side_identity("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.decision == "PASS"
    assert result.identity_bucket == "vrn_visible"
    assert result.checked is True


def _stub_corner_view_deps(monkeypatch, similarity: float):
    """Wires just enough of corner_view's dependency chain (detector + embeddings)
    to exercise _identity_via_corner_view for real, while asserting the make
    classifier is never touched -- it's reserved for pure_side_profile only now."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "load_rgb_array", lambda image: "fake-array")

    class _FakeDetector:
        def best_truck(self, arr):
            return None  # falls back to using the full "array" as the crop

    monkeypatch.setattr(side_module, "get_vehicle_detector", lambda: _FakeDetector())

    class _FakeSiglip:
        def embed_image(self, image):
            return image  # identity mapping -- _cosine is stubbed below anyway

    monkeypatch.setattr(side_module, "get_siglip_model", lambda: _FakeSiglip())
    monkeypatch.setattr(side_module, "_cosine", lambda a, b: similarity)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("corner_view must never call the make classifier")

    monkeypatch.setattr(side_module, "get_make_classifier", _fail_if_called)


def test_corner_view_without_front_reference_is_unverifiable(monkeypatch):
    """No front_reference_image -- corner_view has nothing to compare against, so
    it must be MANUAL_REVIEW (not a silent PASS, and not a fall back to the make
    classifier, which is reserved for pure_side_profile only). Must bail out
    before touching the image at all -- no load_rgb_array/detector stubbing
    needed here, unlike the other corner_view tests."""
    import vfiv.side_image.side_image_check as side_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("corner_view must not touch the image or the make "
                             "classifier when there's no front reference to compare against")

    monkeypatch.setattr(side_module, "load_rgb_array", _fail_if_called)
    monkeypatch.setattr(side_module, "get_vehicle_detector", _fail_if_called)
    monkeypatch.setattr(side_module, "get_make_classifier", _fail_if_called)

    decision, reason, detail = side_module._identity_via_corner_view(
        "does-not-matter.jpg", front_reference_image=None)
    assert decision == "MANUAL_REVIEW"
    assert "no front reference photo" in reason
    assert detail["front_similarity"] is None


def test_corner_view_passes_on_high_front_similarity(monkeypatch):
    import vfiv.side_image.side_image_check as side_module

    _stub_corner_view_deps(monkeypatch, similarity=0.99)

    decision, reason, detail = side_module._identity_via_corner_view(
        "does-not-matter.jpg", front_reference_image="fake-front.jpg", similarity_min=0.9)
    assert decision == "PASS"
    assert detail["front_similarity"] == 0.99


def test_corner_view_manual_review_on_low_front_similarity(monkeypatch):
    import vfiv.side_image.side_image_check as side_module

    _stub_corner_view_deps(monkeypatch, similarity=0.5)

    decision, reason, detail = side_module._identity_via_corner_view(
        "does-not-matter.jpg", front_reference_image="fake-front.jpg", similarity_min=0.9)
    assert decision == "MANUAL_REVIEW"
    assert "uncalibrated signal" in reason


def test_check_side_identity_routes_corner_view_without_claimed_make_dependency(monkeypatch):
    """End-to-end routing: a corner_view bucket must reach
    _identity_via_corner_view without ever needing claimed_make, confirming the
    make classifier really is out of this bucket's path entirely."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "load_rgb_array", lambda image: "fake-array")
    monkeypatch.setattr(side_module, "get_side_image_type_classifier",
                        lambda: type("_C", (), {"predict": lambda self, arr: {"bucket": "corner_view"}})())

    seen = {}

    def _fake_corner_view(image, front_reference_image, similarity_min=None):
        seen["called"] = True
        return "PASS", "[corner_view] front-similarity 0.99", {"bucket": "corner_view", "front_similarity": 0.99}

    monkeypatch.setattr(side_module, "_identity_via_corner_view", _fake_corner_view)

    result = check_side_identity("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD",
                                 front_reference_image="fake-front.jpg")
    assert seen["called"] is True
    assert result.decision == "PASS"
    assert result.identity_bucket == "corner_view"
