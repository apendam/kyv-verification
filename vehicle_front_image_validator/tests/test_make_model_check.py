import os

import pytest

from vfiv.validators.front_image.make_model_check import decide_make_model, validate_make_model

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")
HAS_AWS = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
requires_aws = pytest.mark.skipif(not HAS_AWS, reason="requires AWS credentials (Rekognition backstop)")


def _raw(make_siglip: str, siglip_conf: float = 95.0, make_rekognition: str | None = None,
        model: str = "", model_conf: float = 0.0) -> dict:
    return {"checked": True, "make_siglip": make_siglip, "make_siglip_confidence": siglip_conf,
            "make_rekognition": make_rekognition, "model": model, "model_confidence": model_conf,
            "reason": "test"}


def test_offline_make_matches_via_legal_entity_alias():
    """No credentials needed — painted 'EICHER' (SigLIP) vs Parivahan
    'VE COMMERCIAL VEHICLES LTD'."""
    result = decide_make_model(_raw("EICHER"), claimed_make="VE COMMERCIAL VEHICLES LTD")
    assert result.decision == "PASS"
    assert result.make_status == "MATCH"
    assert result.make_match_via == "siglip"
    assert result.model_checked is False


def test_offline_make_mismatch_on_both_sources_rejects():
    result = decide_make_model(_raw("TATA", make_rekognition="TATA"), claimed_make="ASHOK LEYLAND LTD")
    assert result.decision == "REJECT"
    assert result.make_status == "MISMATCH"
    assert result.make_match_via is None


def test_offline_rekognition_backstops_wrong_siglip_read():
    """Mirrors the real truck2.jpg case: SigLIP misreads the brand, but Rekognition's
    brand-text read is correct — match via EITHER source is enough."""
    result = decide_make_model(
        _raw("Force Motors", siglip_conf=20.0, make_rekognition="TATA"),
        claimed_make="TATA MOTORS LTD",
    )
    assert result.decision == "PASS"
    assert result.make_status == "MATCH"
    assert result.make_match_via == "rekognition"


def test_offline_both_sources_agree():
    result = decide_make_model(_raw("Ashok Leyland", make_rekognition="ASHOK LEYLAND"),
                               claimed_make="ASHOK LEYLAND LTD")
    assert result.decision == "PASS"
    assert result.make_match_via == "both"


def test_offline_no_rekognition_vote_falls_back_to_siglip_alone():
    """Rekognition found no legible brand text (make_rekognition=None) — SigLIP alone
    still decides."""
    result = decide_make_model(_raw("TATA", make_rekognition=None), claimed_make="TATA MOTORS LTD")
    assert result.decision == "PASS"
    assert result.make_match_via == "siglip"
    assert result.extracted_make_rekognition is None


def test_offline_model_not_checked_below_confidence_threshold():
    """Model read present but low-confidence -> mismatch is NOT enforced, make still governs."""
    result = decide_make_model(
        _raw("TATA", make_rekognition="TATA", model="SIGNA 4225", model_conf=40.0),
        claimed_make="TATA MOTORS LTD", claimed_model="LPT 1613",
    )
    assert result.decision == "PASS"
    assert result.model_checked is False
    assert result.model_status is None


def test_offline_model_checked_and_matches_above_confidence_threshold():
    result = decide_make_model(
        _raw("TATA", make_rekognition="TATA", model="LPT 1613", model_conf=95.0),
        claimed_make="TATA MOTORS LTD", claimed_model="LPT 1613",
    )
    assert result.decision == "PASS"
    assert result.model_checked is True
    assert result.model_status == "MATCH"


def test_offline_model_checked_and_mismatches_above_confidence_threshold():
    """High-confidence model mismatch REJECTs even though make matches."""
    result = decide_make_model(
        _raw("TATA", make_rekognition="TATA", model="LPT 1613", model_conf=95.0),
        claimed_make="TATA MOTORS LTD", claimed_model="SIGNA 4225",
    )
    assert result.decision == "REJECT"
    assert result.model_checked is True
    assert result.model_status == "MISMATCH"


def test_offline_no_claimed_model_skips_model_check_entirely():
    result = decide_make_model(
        _raw("TATA", make_rekognition="TATA", model="LPT 1613", model_conf=95.0),
        claimed_make="TATA MOTORS LTD", claimed_model=None,
    )
    assert result.decision == "PASS"
    assert result.model_checked is False


@requires_aws
def test_make_check_works_without_any_api_key(monkeypatch, tmp_path):
    """Make no longer depends on Claude at all (SigLIP + Rekognition are both real
    models) — a missing ANTHROPIC_API_KEY should only affect the optional model arm,
    not the make decision."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = os.path.join(SAMPLES, "truck1.png")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_make_model(path, claimed_make="ASHOK LEYLAND LTD")
    assert result.checked is True
    assert result.decision == "PASS"
    assert result.make_status == "MATCH"
    assert result.model_checked is False  # Claude unavailable -> model read unusable


@requires_aws
def test_make_matches_on_real_sample():
    path = os.path.join(SAMPLES, "truck1.png")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_make_model(path, claimed_make="ASHOK LEYLAND LTD")
    assert result.decision == "PASS"
    assert result.make_status == "MATCH"


@requires_aws
def test_rekognition_backstops_siglip_on_real_tata_sample():
    """truck2.jpg is a known SigLIP miss (misreads as 'Force Motors') — Rekognition's
    real brand-text read should still make this PASS."""
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_make_model(path, claimed_make="TATA MOTORS LTD")
    assert result.decision == "PASS"
    assert result.make_status == "MATCH"
    assert result.make_match_via == "rekognition"


@requires_aws
def test_wrong_claimed_make_rejects_on_real_sample():
    path = os.path.join(SAMPLES, "truck1.png")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_make_model(path, claimed_make="TATA MOTORS LTD")
    assert result.decision == "REJECT"
    assert result.make_status == "MISMATCH"
