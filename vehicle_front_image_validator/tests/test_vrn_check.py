import os

import pytest

from vfiv.validators.front_image.vrn_check import decide_vrn, validate_vrn

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_AWS = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
requires_aws = pytest.mark.skipif(not HAS_AWS, reason="requires AWS credentials (Rekognition)")


def _raw(plate: str, confidence: float = 90.0) -> dict:
    return {"checked": True, "plate": plate, "plate_confidence": confidence,
            "plate_colour": "yellow", "colour_confidence": 90.0, "reason": "test"}


@pytest.mark.parametrize("claimed,read", [
    ("UP42T4069", "UP42T4O69"),   # 0 <-> O
    ("UP42T4O69", "UP42T4069"),   # O <-> 0, reversed direction
    ("HR61F5938", "HR61FS938"),   # 5 <-> S
    ("HR61FS938", "HR61F5938"),   # S <-> 5, reversed direction
    ("MH12AB1234", "MHI2ABI234"),  # 1 <-> I
])
def test_offline_confusable_pairs_still_match(claimed, read):
    """No credentials needed — exercises decide_vrn's use of truck_extract_match's
    confusion-aware matcher directly against a synthetic Rekognition read."""
    result = decide_vrn(_raw(read), claimed_vrn=claimed)
    assert result.decision == "PASS"
    assert result.status == "MATCH"
    assert result.inferred is True


def test_manual_review_when_aws_credentials_invalid(monkeypatch):
    """Q2 no longer calls Claude at all — its tech-failure path is an AWS Rekognition
    credential error, not a missing ANTHROPIC_API_KEY. Mocked deterministically: boto3
    caches its default session process-wide, so injecting fake env-var credentials
    after a real client already exists in the same process doesn't reliably force a
    fresh (failing) credential resolution."""
    import vfiv.validators.front_image.vrn_check as vrn_check_module
    from vfiv.backends.rekognition import RekognitionCredentialError

    def _fake_detect_plate(image):
        raise RekognitionCredentialError("AWS Rekognition credentials are invalid or expired.")

    monkeypatch.setattr(vrn_check_module, "detect_plate", _fake_detect_plate)

    result = validate_vrn("does-not-matter.jpg", claimed_vrn="UP42T4069")
    assert result.decision == "MANUAL_REVIEW"
    assert result.status == "UNREADABLE"
    assert result.checked is False


@requires_aws
def test_exact_match_passes():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_vrn(path, claimed_vrn="UP42T4069")
    assert result.decision == "PASS"
    assert result.status == "MATCH"
    assert result.inferred is False
    assert result.plate_colour in ("white", "yellow", "green", "black", "red", "unknown")


@requires_aws
def test_smudge_confusable_char_still_matches():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    # 'O' in place of the actual '0' — confusable, should still MATCH (smudge-inferred).
    result = validate_vrn(path, claimed_vrn="UP42T4O69")
    assert result.decision == "PASS"
    assert result.inferred is True


@requires_aws
def test_wrong_claimed_vrn_rejects():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_vrn(path, claimed_vrn="MH12AB1234")
    assert result.decision == "REJECT"
    assert result.status == "MISMATCH"
