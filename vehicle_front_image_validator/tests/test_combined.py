import os

import pytest

from vfiv.validators.combined import validate_upload

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_AWS = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
requires_both = pytest.mark.skipif(
    not (HAS_ANTHROPIC and HAS_AWS), reason="requires ANTHROPIC_API_KEY and AWS credentials")


def test_manual_review_when_no_api_key(monkeypatch, tmp_path):
    """Q1's CV gate still runs without a key, but its Claude judgment call doesn't —
    Q1 never reaches PASS, so Q2/Q3 never even run (the short-circuit)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = validate_upload(str(img_path), claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert result.front.checked is False
    assert result.vrn is None
    assert result.make_model is None


@requires_both
def test_pass_when_all_three_checks_pass():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_upload(path, claimed_vrn="UP42T4069", claimed_make="TATA MOTORS LTD")
    assert result.front.decision == "PASS"
    assert result.vrn.status == "MATCH"
    assert result.make_model.make_status == "MATCH"
    assert result.make_model.make_match_via == "rekognition"  # known SigLIP miss, backstopped
    assert result.decision == "PASS"


@requires_both
def test_reject_when_front_ok_but_vrn_mismatches():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_upload(path, claimed_vrn="MH12AB1234", claimed_make="TATA MOTORS LTD")
    assert result.front.decision == "PASS"
    assert result.vrn.status == "MISMATCH"
    assert result.decision == "REJECT"


@requires_both
def test_overall_decision_takes_worst_of_q2_and_q3():
    """VRN mismatches (REJECT) while make still matches (PASS) — overall must be
    REJECT (severity: REJECT > MANUAL_REVIEW > PASS), and Q3 still ran (no
    short-circuiting between Q2 and Q3 once Q1 has passed)."""
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_upload(path, claimed_vrn="MH12AB1234", claimed_make="TATA MOTORS LTD")
    assert result.vrn.decision == "REJECT"
    assert result.make_model.decision == "PASS"
    assert result.decision == "REJECT"


@requires_both
def test_q2_and_q3_never_run_when_q1_rejects():
    """truck1.png is a known CV-gate REJECT (see test_front_image.py) — Q2/Q3 should be
    short-circuited entirely (no Rekognition/SigLIP calls spent) rather than run
    pointlessly."""
    path = os.path.join(SAMPLES, "truck1.png")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_upload(path, claimed_vrn="BR28GB7804", claimed_make="ASHOK LEYLAND LTD")
    assert result.front.decision == "REJECT"
    assert result.decision == "REJECT"
    assert result.vrn is None
    assert result.make_model is None
