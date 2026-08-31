import numpy as np
import pytest

from vfiv.side_image.side_image_check import (
    _color_histogram_similarity,
    _worst_decision,
    check_axle_count,
    check_side_identity,
    classify_axle_count,
    decide_axle_count,
    decide_axle_source_consistency,
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

    monkeypatch.setattr(side_module, "classify_side_image_type",
                        lambda image, backend, model=None: {"checked": True, "bucket": "vrn_visible",
                                                            "reason": "plate legible"})
    monkeypatch.setattr(side_module, "_identity_via_vrn",
                        lambda image, claimed_vrn: ("PASS", "[vrn_visible] matched", {"bucket": "vrn_visible"}))

    result = check_side_identity("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.decision == "PASS"
    assert result.identity_bucket == "vrn_visible"
    assert "bucket routing: plate legible" in result.reason


def test_check_side_identity_degrades_when_type_classifier_unavailable(monkeypatch):
    """A missing API key/credentials for the bucket-routing VLM call must degrade
    to MANUAL_REVIEW, not crash -- same posture as every other VLM-backed check
    in this codebase."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "classify_side_image_type",
                        lambda image, backend, model=None: {"checked": False, "error": "no ANTHROPIC_API_KEY"})

    result = check_side_identity("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert "no ANTHROPIC_API_KEY" in result.reason


def test_check_side_identity_degrades_on_unrecognized_bucket(monkeypatch):
    """A malformed/unexpected bucket value from the VLM must degrade gracefully,
    not raise or silently misroute."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "classify_side_image_type",
                        lambda image, backend, model=None: {"checked": True, "bucket": "not-a-real-bucket"})

    result = check_side_identity("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False


def _stub_corner_view_deps(monkeypatch, similarity: float, color_similarity: float = 1.0):
    """Wires just enough of corner_view's dependency chain (detector + embeddings +
    colour histogram) to exercise _identity_via_corner_view for real -- no make
    classifier stubbing needed any more since neither bucket imports it at all."""
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
    monkeypatch.setattr(side_module, "_color_histogram_similarity", lambda a, b: color_similarity)


def test_corner_view_without_front_reference_is_unverifiable(monkeypatch):
    """No front_reference_image -- corner_view has nothing to compare against, so
    it must be MANUAL_REVIEW (not a silent PASS). Must bail out before touching
    the image at all -- no load_rgb_array/detector stubbing needed here, unlike
    the other corner_view tests."""
    import vfiv.side_image.side_image_check as side_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("corner_view must not touch the image when there's "
                             "no front reference to compare against")

    monkeypatch.setattr(side_module, "load_rgb_array", _fail_if_called)
    monkeypatch.setattr(side_module, "get_vehicle_detector", _fail_if_called)

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


def _stub_pure_side_profile_deps(monkeypatch, color_similarity: float):
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "load_rgb_array", lambda image: "fake-array")

    class _FakeDetector:
        def best_truck(self, arr):
            return None

    monkeypatch.setattr(side_module, "get_vehicle_detector", lambda: _FakeDetector())
    monkeypatch.setattr(side_module, "_color_histogram_similarity", lambda a, b: color_similarity)


def test_pure_side_profile_without_front_reference_is_unverifiable(monkeypatch):
    """Same as corner_view's no-reference case -- no plate, no front grille, and
    now no front reference photo either means there's nothing left to check at
    all. Must bail out before touching the image."""
    import vfiv.side_image.side_image_check as side_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("pure_side_profile must not touch the image when "
                             "there's no front reference to compare against")

    monkeypatch.setattr(side_module, "load_rgb_array", _fail_if_called)
    monkeypatch.setattr(side_module, "get_vehicle_detector", _fail_if_called)

    decision, reason, detail = side_module._identity_via_pure_side_profile(
        "does-not-matter.jpg", front_reference_image=None)
    assert decision == "MANUAL_REVIEW"
    assert "no front reference photo" in reason
    assert detail["color_hist_similarity"] is None


def test_pure_side_profile_never_passes_even_on_perfect_color_match(monkeypatch):
    """Individual-vehicle identity is never solved from a bare side profile alone
    -- even a perfect colour match is capped at MANUAL_REVIEW, same principle as
    the old make-classifier version (two different trucks of the same colour
    would pass this too)."""
    import vfiv.side_image.side_image_check as side_module

    _stub_pure_side_profile_deps(monkeypatch, color_similarity=1.0)

    decision, reason, detail = side_module._identity_via_pure_side_profile(
        "does-not-matter.jpg", front_reference_image="fake-front.jpg", color_hist_min=0.8)
    assert decision == "MANUAL_REVIEW"
    assert detail["color_hist_similarity"] == 1.0


def test_pure_side_profile_color_mismatch_is_manual_review_not_reject(monkeypatch):
    """Unlike the make classifier it replaced, a colour mismatch here must NOT
    auto-reject -- same uncalibrated-threshold caveat as corner_view's colour
    check, so it stays a lead for a human rather than a verdict."""
    import vfiv.side_image.side_image_check as side_module

    _stub_pure_side_profile_deps(monkeypatch, color_similarity=0.1)

    decision, reason, detail = side_module._identity_via_pure_side_profile(
        "does-not-matter.jpg", front_reference_image="fake-front.jpg", color_hist_min=0.8)
    assert decision == "MANUAL_REVIEW"
    assert "possible mismatch" in reason


def test_check_side_identity_routes_corner_view_without_claimed_make_dependency(monkeypatch):
    """End-to-end routing: a corner_view bucket must reach
    _identity_via_corner_view without ever needing claimed_make, confirming the
    make classifier really is out of this bucket's path entirely."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "classify_side_image_type",
                        lambda image, backend, model=None: {"checked": True, "bucket": "corner_view"})

    seen = {}

    def _fake_corner_view(image, front_reference_image, similarity_min=None, color_hist_min=None):
        seen["called"] = True
        return "PASS", "[corner_view] front-similarity 0.99", {"bucket": "corner_view", "front_similarity": 0.99}

    monkeypatch.setattr(side_module, "_identity_via_corner_view", _fake_corner_view)

    result = check_side_identity("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD",
                                 front_reference_image="fake-front.jpg")
    assert seen["called"] is True
    assert result.decision == "PASS"
    assert result.identity_bucket == "corner_view"


def test_color_histogram_similarity_identical_crops_is_near_one():
    crop = np.random.default_rng(0).integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
    assert _color_histogram_similarity(crop, crop) == pytest.approx(1.0, abs=1e-6)


def test_color_histogram_similarity_distinguishes_solid_colours():
    """A solid red truck crop vs. a solid blue one -- about as clear-cut a
    different-colour-vehicle case as it gets -- must score far below a same-crop
    comparison, not just marginally lower."""
    red = np.zeros((40, 40, 3), dtype=np.uint8)
    red[..., 0] = 200
    blue = np.zeros((40, 40, 3), dtype=np.uint8)
    blue[..., 2] = 200
    same = _color_histogram_similarity(red, red)
    different = _color_histogram_similarity(red, blue)
    assert different < same


def test_axle_source_consistency_auto_is_always_trusted():
    """auto-filled (straight from RC) is trusted as-is -- no vehicle_mapper needed,
    and no cross-check even if the count would disagree with a mapper class."""
    result = decide_axle_source_consistency(claimed_axle_count=4, axle_source="auto", vehicle_mapper=None)
    assert result["decision"] == "PASS"


def test_axle_source_consistency_manual_matching_mapper_passes():
    result = decide_axle_source_consistency(claimed_axle_count=4, axle_source="manual", vehicle_mapper="VC12")
    assert result["decision"] == "PASS"


def test_axle_source_consistency_manual_mismatching_mapper_rejects():
    """VC12 implies 4 axles -- an agent claiming 6 for a VC12-classified vehicle
    disagrees with the vehicle's own RC-derived class."""
    result = decide_axle_source_consistency(claimed_axle_count=6, axle_source="manual", vehicle_mapper="VC12")
    assert result["decision"] == "REJECT"
    assert "VC12" in result["reason"]


def test_axle_source_consistency_manual_without_mapper_is_manual_review():
    result = decide_axle_source_consistency(claimed_axle_count=4, axle_source="manual", vehicle_mapper=None)
    assert result["decision"] == "MANUAL_REVIEW"


def test_axle_source_consistency_unknown_mapper_code_is_manual_review():
    result = decide_axle_source_consistency(claimed_axle_count=4, axle_source="manual", vehicle_mapper="VC999")
    assert result["decision"] == "MANUAL_REVIEW"
    assert "unknown vehicle mapper" in result["reason"]


def test_axle_source_consistency_mapper_with_no_defined_axle_count():
    """VC4 (Car) has no axle count in the source table -- nothing to compare
    against, so this is unresolved rather than a false mismatch."""
    result = decide_axle_source_consistency(claimed_axle_count=2, axle_source="manual", vehicle_mapper="VC4")
    assert result["decision"] == "MANUAL_REVIEW"


def test_axle_source_consistency_unknown_source_value():
    result = decide_axle_source_consistency(claimed_axle_count=4, axle_source="bogus", vehicle_mapper="VC12")
    assert result["decision"] == "MANUAL_REVIEW"


def test_check_axle_count_skips_source_consistency_without_axle_source(monkeypatch):
    """axle_source is opt-in -- omitting it must not touch
    decide_axle_source_consistency at all, and the resulting fields stay None."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "classify_axle_count",
                        lambda image, backend, model=None: _axle_read(4))

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("decide_axle_source_consistency should not run without axle_source")

    monkeypatch.setattr(side_module, "decide_axle_source_consistency", _fail_if_called)

    result = check_axle_count("does-not-matter.jpg", claimed_axle_count=4)
    assert result.decision == "PASS"
    assert result.axle_source is None
    assert result.mapper_expected_axle_count is None


def test_check_axle_count_folds_in_source_consistency_mismatch(monkeypatch):
    """A photo that matches the claimed count can still be REJECTed overall if the
    manually-entered count disagrees with the vehicle's own RC-derived mapper
    class -- two independent signals, worst-of."""
    import vfiv.side_image.side_image_check as side_module

    monkeypatch.setattr(side_module, "classify_axle_count",
                        lambda image, backend, model=None: _axle_read(6))

    result = check_axle_count("does-not-matter.jpg", claimed_axle_count=6,
                              axle_source="manual", vehicle_mapper="VC12")
    assert result.status == "MATCH"  # the photo-vs-claim check itself passed
    assert result.decision == "REJECT"  # but source-consistency overrides it
    assert result.mapper_expected_axle_count == 4
    assert "source-consistency" in result.reason
