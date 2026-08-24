import os

import pytest

from vfiv.experiments.runner import run_test_case

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
