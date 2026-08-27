import pytest

from vfiv.backends.fastag_reader import DecodedCode, FastagRead, parse_qr_payload, read_fastag
from vfiv.fastag_image.fastag_check import decide_fastag


def _read(codes=None, printed_id=None) -> dict:
    return {"checked": True, "read": FastagRead(decoded_codes=codes or [], printed_id_text=printed_id)}


def test_parse_qr_payload_splits_id_and_bank():
    fastag_id, bank_code = parse_qr_payload("34161234567890123@icici")
    assert fastag_id == "34161234567890123"
    assert bank_code == "icici"


def test_parse_qr_payload_rejects_unexpected_shape():
    assert parse_qr_payload("not-a-vpa-string") == (None, None)
    assert parse_qr_payload("a@b@c") == (None, None)


def test_barcode_matches_claimed_id_passes():
    r = _read(codes=[DecodedCode(symbology="CODE128", data="607469-009-0874936")],
              printed_id="607469-009-0874936")
    result = decide_fastag(r, claimed_fastag_id="607469-009-0874936")
    assert result.decision == "PASS"
    assert result.matched_via == "barcode:code128"


def test_qr_matches_claimed_id_passes_and_is_preferred_over_ocr():
    r = _read(codes=[DecodedCode(symbology="QRCODE", data="607469009874936@icici")],
              printed_id="607469009874936")
    result = decide_fastag(r, claimed_fastag_id="607469009874936")
    assert result.decision == "PASS"
    assert result.matched_via == "qr"


def test_qr_bank_code_mismatch_downgrades_to_manual_review_not_reject():
    r = _read(codes=[DecodedCode(symbology="QRCODE", data="607469009874936@icici")])
    result = decide_fastag(r, claimed_fastag_id="607469009874936", claimed_bank_code="hdfc")
    assert result.decision == "MANUAL_REVIEW"
    assert "bank code mismatch" in result.reason


def test_ocr_only_fuzzy_match_still_passes():
    """No barcode/QR decoded at all (poor photo) -- OCR is the last resort, with
    the same confusable-character tolerance as VRN matching (0<->O etc.)."""
    r = _read(printed_id="6074690090874936".replace("0", "O", 1))  # one 0->O smudge
    result = decide_fastag(r, claimed_fastag_id="6074690090874936")
    assert result.decision == "PASS"
    assert result.matched_via == "ocr"


def test_sources_disagreeing_with_each_other_rejects_as_tampering():
    """Barcode decodes to one value, OCR reads a DIFFERENT value -- both legible,
    but inconsistent with each other. This is flagged even if the OCR value
    happens to match the claimed id -- internal inconsistency is the signal."""
    r = _read(
        codes=[DecodedCode(symbology="CODE128", data="607469-009-0874936")],
        printed_id="111111-111-1111111",
    )
    result = decide_fastag(r, claimed_fastag_id="111111-111-1111111")
    assert result.decision == "REJECT"
    assert "disagree" in result.reason


def test_wrong_id_with_no_disagreement_rejects_as_mismatch_not_tampering():
    r = _read(codes=[DecodedCode(symbology="CODE128", data="607469-009-0874936")],
              printed_id="607469-009-0874936")
    result = decide_fastag(r, claimed_fastag_id="999999-999-9999999")
    assert result.decision == "REJECT"
    assert "disagree" not in result.reason


def test_nothing_readable_is_manual_review_not_reject():
    r = _read()
    result = decide_fastag(r, claimed_fastag_id="607469-009-0874936")
    assert result.decision == "MANUAL_REVIEW"


def test_unknown_ocr_backend_raises_before_any_image_io():
    """The backend name is validated first, before decode_codes/pyzbar even opens
    the image -- so this raises ValueError even for a path that doesn't exist."""
    with pytest.raises(ValueError, match="unknown FASTag OCR backend"):
        read_fastag("does-not-exist.jpg", backend="not-a-real-backend")
