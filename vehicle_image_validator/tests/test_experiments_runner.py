import os

import pytest

from vfiv.experiments.runner import run_q1_only, run_test_case

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_AWS = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
requires_both = pytest.mark.skipif(
    not (HAS_ANTHROPIC and HAS_AWS), reason="requires ANTHROPIC_API_KEY and AWS credentials")


def test_manual_review_when_q1_backend_unavailable(monkeypatch, tmp_path):
    """Simulates Gemini being fully unavailable — must clear all three of its
    credential env vars (API key AND Vertex AI service account), since Vertex AI
    credentials are configured in this environment now."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GEMINI_VERTEX_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = run_test_case(str(img_path), claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD",
                           q1_backend="gemini")
    assert result.overall_decision == "MANUAL_REVIEW"
    assert result.q1.checked is False
    assert result.q2 is None


HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
requires_gemini_and_both = pytest.mark.skipif(
    not (HAS_ANTHROPIC and HAS_AWS and HAS_GEMINI),
    reason="requires ANTHROPIC_API_KEY, AWS credentials, and Gemini (API key or Vertex AI)")


@requires_gemini_and_both
def test_gemini_backend_end_to_end_via_vertex_ai():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = run_test_case(
        path, claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD",
        q1_backend="gemini", q2_backend="gemini", q3_make_backend="gemini", q3_model_backend="gemini",
    )
    assert result.overall_decision == "PASS"
    assert result.q2.extracted_raw == "UP42T4069"
    assert result.q3_make_votes[0]["source"] == "gemini"


@requires_both
def test_default_backends_pass_matches_production():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = run_test_case(path, claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.q1_backend == "real_cv"
    assert result.q2_backend == "rekognition"
    assert result.q3_make_backend == "siglip_rekognition"
    assert result.overall_decision == "PASS"
    assert result.q3_make_status == "MATCH"


def _passing_q1_raw():
    return {"checked": True, "vehicle_type": "truck", "view": "front", "is_front": True,
           "front_complete": True, "confidence": 95.0, "is_screenshot": False,
           "is_photo_of_photo": False, "ai_generated": False, "ai_confidence": 0.0,
           "reason": "clear front view"}


def test_q1_only_skips_duplicate_check_without_vrn_and_upload_id(monkeypatch):
    """Without BOTH claimed_vrn and upload_id, run_q1_only must not touch the
    duplicate-detection corpus at all -- it's opt-in, same as
    check_side_image_upload's upload_id gate."""
    import vfiv.experiments.runner as runner_module

    monkeypatch.setattr(runner_module.q1_select, "classify_q1", lambda image, backend, gemini_model=None: _passing_q1_raw())

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("check_duplicate should not be called without claimed_vrn + upload_id")

    monkeypatch.setattr(runner_module, "check_duplicate", _fail_if_called)

    result = run_q1_only("does-not-matter.jpg")
    assert result.decision == "PASS"
    assert result.duplicate_is_suspect is None


def test_q1_only_folds_in_duplicate_suspect_when_vrn_and_upload_id_given(monkeypatch):
    """With both claimed_vrn and upload_id given, a duplicate-check MANUAL_REVIEW
    verdict must escalate Q1's own PASS decision and surface duplicate_is_suspect."""
    import vfiv.experiments.runner as runner_module
    from vfiv.schemas import DuplicateCheckResult

    monkeypatch.setattr(runner_module.q1_select, "classify_q1", lambda image, backend, gemini_model=None: _passing_q1_raw())

    def _fake_check_duplicate(image, upload_id, claimed_vrn, image_type="front"):
        assert image_type == "front"
        return DuplicateCheckResult(
            decision="MANUAL_REVIEW", reason="near-duplicate of img_1", checked=True,
            claimed_vrn=claimed_vrn, is_duplicate_suspect=True, best_match_id="img_1",
            best_match_similarity=0.99, best_match_vrn="MH12AB1234",
        )

    monkeypatch.setattr(runner_module, "check_duplicate", _fake_check_duplicate)

    result = run_q1_only("does-not-matter.jpg", claimed_vrn="UP42T4069", upload_id="img_x")
    assert result.decision == "MANUAL_REVIEW"
    assert result.duplicate_is_suspect is True
    assert "duplicate: near-duplicate of img_1" in result.reason


def test_q1_only_surfaces_duplicate_matches_list_for_manual_review(monkeypatch):
    """The list of every suspect upload_id (not just the single best match) must
    reach the caller, so a manual reviewer can pull each one up and compare."""
    import vfiv.experiments.runner as runner_module
    from vfiv.schemas import DuplicateCheckResult, DuplicateMatchInfo

    monkeypatch.setattr(runner_module.q1_select, "classify_q1", lambda image, backend, gemini_model=None: _passing_q1_raw())

    def _fake_check_duplicate(image, upload_id, claimed_vrn, image_type="front"):
        return DuplicateCheckResult(
            decision="MANUAL_REVIEW", reason="near-duplicate of img_1", checked=True,
            claimed_vrn=claimed_vrn, is_duplicate_suspect=True, best_match_id="img_1",
            best_match_similarity=0.99, best_match_vrn="MH12AB1234",
            duplicate_matches=[
                DuplicateMatchInfo(upload_id="img_1", claimed_vrn="MH12AB1234", similarity=0.99),
                DuplicateMatchInfo(upload_id="img_2", claimed_vrn="DL01ZZ9999", similarity=0.975),
            ],
        )

    monkeypatch.setattr(runner_module, "check_duplicate", _fake_check_duplicate)

    result = run_q1_only("does-not-matter.jpg", claimed_vrn="UP42T4069", upload_id="img_x")
    assert [m.upload_id for m in result.duplicate_matches] == ["img_1", "img_2"]


def test_run_test_case_skips_duplicate_check_without_upload_id(monkeypatch):
    """run_test_case's claimed_vrn is always present (Q2 needs it regardless), so
    the duplicate check must be gated on upload_id alone -- omitting it must not
    touch the duplicate-detection corpus."""
    import vfiv.experiments.runner as runner_module
    from vfiv.schemas import MakeModelCheckResult, VrnCheckResult

    monkeypatch.setattr(runner_module.q1_select, "classify_q1",
                        lambda image, backend, gemini_model=None: _passing_q1_raw())

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("check_duplicate should not be called without upload_id")

    monkeypatch.setattr(runner_module, "check_duplicate", _fail_if_called)
    monkeypatch.setattr(runner_module, "run_q2_only", lambda *a, **k: VrnCheckResult(
        decision="PASS", status="MATCH", claimed_vrn="UP42T4069", reason="test", checked=True))
    monkeypatch.setattr(runner_module, "run_q3_only", lambda *a, **k: {
        "decision": "PASS", "reason": "test", "make_status": "MATCH", "make_match_via": ["siglip"],
        "make_votes": [], "model_checked": False, "model_status": None, "extracted_model": None})

    result = run_test_case("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.q1.duplicate_is_suspect is None


def test_run_test_case_folds_duplicate_check_into_q1_and_short_circuits(monkeypatch):
    """Passing upload_id must thread claimed_vrn + upload_id into run_test_case's own
    Q1 call -- a duplicate-check MANUAL_REVIEW verdict escalates Q1 and short-circuits
    Q2/Q3, same as any other non-PASS Q1 decision (front.decision != "PASS")."""
    import vfiv.experiments.runner as runner_module
    from vfiv.schemas import DuplicateCheckResult

    monkeypatch.setattr(runner_module.q1_select, "classify_q1",
                        lambda image, backend, gemini_model=None: _passing_q1_raw())

    def _fake_check_duplicate(image, upload_id, claimed_vrn, image_type="front"):
        assert image_type == "front"
        assert upload_id == "img_x"
        assert claimed_vrn == "UP42T4069"
        return DuplicateCheckResult(
            decision="MANUAL_REVIEW", reason="near-duplicate of img_1", checked=True,
            claimed_vrn=claimed_vrn, is_duplicate_suspect=True, best_match_id="img_1",
            best_match_similarity=0.99, best_match_vrn="MH12AB1234",
        )

    monkeypatch.setattr(runner_module, "check_duplicate", _fake_check_duplicate)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Q2/Q3 must not run once Q1 is escalated to MANUAL_REVIEW")

    monkeypatch.setattr(runner_module, "run_q2_only", _fail_if_called)
    monkeypatch.setattr(runner_module, "run_q3_only", _fail_if_called)

    result = run_test_case("does-not-matter.jpg", claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD",
                           upload_id="img_x")
    assert result.overall_decision == "MANUAL_REVIEW"
    assert result.q1.duplicate_is_suspect is True
    assert result.q2 is None


@requires_both
def test_q1_reject_short_circuits_q2_and_q3():
    """truck1.png is a known CV-gate REJECT — Q2/Q3 should never run."""
    path = os.path.join(SAMPLES, "truck1.png")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = run_test_case(path, claimed_vrn="BR28GB7804", claimed_make="ASHOK LEYLAND LTD")
    assert result.q1.decision == "REJECT"
    assert result.overall_decision == "REJECT"
    assert result.q2 is None
    assert result.q3_make_status is None


@requires_both
def test_legacy_claude_backends_end_to_end():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = run_test_case(
        path, claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD",
        q1_backend="claude", q2_backend="claude", q3_make_backend="claude", q3_model_backend="claude",
    )
    assert result.overall_decision == "PASS"
    assert result.q2.extracted_raw == "UP42T4069"
    assert result.q3_make_votes[0]["source"] == "claude"
