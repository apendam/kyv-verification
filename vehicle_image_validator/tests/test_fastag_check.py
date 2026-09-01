import pytest

from vfiv.backends.fastag_reader import DecodedCode, FastagRead, parse_qr_payload, read_fastag
from vfiv.fastag_image.fastag_check import (
    check_fastag_completeness,
    check_fastag_upload,
    check_printed_digits_only,
    check_qr_only,
    classify_fastag_upload,
    decide_fastag,
    decide_printed_digits_only,
    decide_qr_only,
)


def _complete_read(complete=True, confidence=95.0, reason="all parts visible"):
    return {"checked": True, "sticker_complete": complete, "confidence": confidence, "reason": reason}


def _read(codes=None, printed_id=None) -> dict:
    return {"checked": True, "read": FastagRead(decoded_codes=codes or [], printed_id_text=printed_id)}


def test_parse_qr_payload_splits_id_and_bank():
    """The bare '<id>@<bank>' shape -- a fallback for a non-UPI-URI QR encoding,
    not the real format (see the upi:// tests below), but still supported."""
    fastag_id, bank_code = parse_qr_payload("34161234567890123@icici")
    assert fastag_id == "34161234567890123"
    assert bank_code == "icici"


def test_parse_qr_payload_rejects_unexpected_shape():
    assert parse_qr_payload("not-a-vpa-string") == (None, None)
    assert parse_qr_payload("a@b@c") == (None, None)


def test_parse_qr_payload_real_upi_uri_format():
    """The actual format a FASTag recharge QR encodes -- a UPI deep link, with
    the tag id/bank code inside the 'pa' param, prefixed with 'netc.' (confirmed
    against a real decoded FASTag QR)."""
    data = ("upi://pay?ver=01&mode=01&pa=netc.607469009874936@icici&purpose=00"
           "&mc=1234&pn=NETC%20FASTag%20Recharge&orgid=123456&qrMedium=04")
    fastag_id, bank_code = parse_qr_payload(data)
    assert fastag_id == "607469009874936"
    assert bank_code == "icici"


def test_parse_qr_payload_upi_uri_without_pa_param_is_unreadable():
    assert parse_qr_payload("upi://pay?ver=01&mode=01&purpose=00") == (None, None)


def test_parse_qr_payload_upi_uri_missing_netc_prefix_still_splits():
    """Robustness -- if some issuer's QR omits the 'netc.' prefix on 'pa', the id
    itself should still come through untouched rather than failing outright."""
    fastag_id, bank_code = parse_qr_payload("upi://pay?pa=607469009874936@icici")
    assert fastag_id == "607469009874936"
    assert bank_code == "icici"


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


def test_check_fastag_upload_skips_duplicate_check_without_vrn_and_upload_id(monkeypatch):
    """Without BOTH claimed_vrn and upload_id, check_fastag_upload must not touch
    the duplicate-detection corpus -- it's opt-in, same as Q1/side's own checks."""
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_upload", lambda image, backend, vlm_model=None: _read(
        codes=[DecodedCode(symbology="CODE128", data="607469-009-0874936")],
        printed_id="607469-009-0874936"))
    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: _complete_read())

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("check_duplicate should not be called without claimed_vrn + upload_id")

    monkeypatch.setattr(fastag_module, "check_duplicate", _fail_if_called)

    result = check_fastag_upload("does-not-matter.jpg", claimed_fastag_id="607469-009-0874936")
    assert result.decision == "PASS"
    assert result.duplicate_is_suspect is None
    assert result.sticker_complete is True


def test_check_fastag_upload_folds_in_duplicate_suspect_when_vrn_and_upload_id_given(monkeypatch):
    """With both claimed_vrn and upload_id given, a duplicate-check MANUAL_REVIEW
    verdict must escalate the fastag decision and surface duplicate_is_suspect."""
    import vfiv.fastag_image.fastag_check as fastag_module
    from vfiv.schemas import DuplicateCheckResult

    monkeypatch.setattr(fastag_module, "classify_fastag_upload", lambda image, backend, vlm_model=None: _read(
        codes=[DecodedCode(symbology="CODE128", data="607469-009-0874936")],
        printed_id="607469-009-0874936"))
    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: _complete_read())

    def _fake_check_duplicate(image, upload_id, claimed_vrn, image_type="front"):
        assert image_type == "fastag"
        return DuplicateCheckResult(
            decision="MANUAL_REVIEW", reason="near-duplicate of img_1", checked=True,
            claimed_vrn=claimed_vrn, is_duplicate_suspect=True, best_match_id="img_1",
            best_match_similarity=0.99, best_match_vrn="MH12AB1234",
        )

    monkeypatch.setattr(fastag_module, "check_duplicate", _fake_check_duplicate)

    result = check_fastag_upload("does-not-matter.jpg", claimed_fastag_id="607469-009-0874936",
                                 claimed_vrn="UP42T4069", upload_id="img_x")
    assert result.decision == "MANUAL_REVIEW"
    assert result.duplicate_is_suspect is True
    assert "duplicate: near-duplicate of img_1" in result.reason


# --- QR-only check --------------------------------------------------------------

def test_qr_only_matching_tag_id_passes():
    r = _read(codes=[DecodedCode(symbology="QRCODE", data="607469009874936@icici")])
    result = decide_qr_only(r, claimed_fastag_id="607469009874936")
    assert result.decision == "PASS"
    assert result.status == "MATCH"
    assert result.qr_tag_id == "607469009874936"
    assert result.qr_bank_code == "icici"


def test_qr_only_mismatched_tag_id_rejects():
    r = _read(codes=[DecodedCode(symbology="QRCODE", data="607469009874936@icici")])
    result = decide_qr_only(r, claimed_fastag_id="999999999999999")
    assert result.decision == "REJECT"
    assert result.status == "MISMATCH"


def test_qr_only_bank_code_mismatch_is_manual_review_not_reject():
    r = _read(codes=[DecodedCode(symbology="QRCODE", data="607469009874936@icici")])
    result = decide_qr_only(r, claimed_fastag_id="607469009874936", claimed_bank_code="hdfc")
    assert result.decision == "MANUAL_REVIEW"
    assert "bank code mismatch" in result.reason


def test_qr_only_no_qr_decoded_is_manual_review():
    """A barcode-only read (no QR at all) must not be treated as a mismatch --
    there's simply nothing to compare, same "bad photo isn't proof of fraud"
    posture as everywhere else in this module."""
    r = _read(codes=[DecodedCode(symbology="CODE128", data="607469-009-0874936")])
    result = decide_qr_only(r, claimed_fastag_id="607469009874936")
    assert result.decision == "MANUAL_REVIEW"
    assert result.status == "UNREADABLE"


def test_check_qr_only_degrades_when_read_unavailable(monkeypatch):
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_upload",
                        lambda image, backend, vlm_model=None: {"checked": False, "error": "no AWS credentials"})

    result = check_qr_only("does-not-matter.jpg", claimed_fastag_id="607469009874936")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert "no AWS credentials" in result.reason


# --- Printed-digits-only check ---------------------------------------------------

def test_printed_digits_only_exact_match_passes():
    r = _read(printed_id="607469-009-0874936")
    result = decide_printed_digits_only(r, claimed_barcode="607469-009-0874936")
    assert result.decision == "PASS"
    assert result.status == "MATCH"


def test_printed_digits_only_fuzzy_confusable_match_passes():
    """Same confusable-character tolerance (0<->O etc.) as the rest of this
    module's OCR matching."""
    r = _read(printed_id="6074690090874936".replace("0", "O", 1))
    result = decide_printed_digits_only(r, claimed_barcode="6074690090874936")
    assert result.decision == "PASS"


def test_printed_digits_only_mismatch_rejects():
    r = _read(printed_id="607469-009-0874936")
    result = decide_printed_digits_only(r, claimed_barcode="111111-111-1111111")
    assert result.decision == "REJECT"
    assert result.status == "MISMATCH"


def test_printed_digits_only_nothing_readable_is_manual_review():
    r = _read()
    result = decide_printed_digits_only(r, claimed_barcode="607469-009-0874936")
    assert result.decision == "MANUAL_REVIEW"
    assert result.status == "UNREADABLE"


def test_check_printed_digits_only_degrades_when_read_unavailable(monkeypatch):
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_upload",
                        lambda image, backend, vlm_model=None: {"checked": False, "error": "no AWS credentials"})

    result = check_printed_digits_only("does-not-matter.jpg", claimed_barcode="607469-009-0874936")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert "no AWS credentials" in result.reason


# --- classify_fastag_upload degradation ------------------------------------------

def test_classify_fastag_upload_degrades_on_unexpected_exception_type(monkeypatch):
    """A missing OS-level dependency (libzbar0 for pyzbar) or an unexpected SDK
    error raises something OTHER than FastagReadError -- must still degrade
    cleanly to checked=False, not crash the caller (which, before this fix,
    propagated all the way to the webapp as a bare unexplained "Error")."""
    import vfiv.fastag_image.fastag_check as fastag_module

    def _boom(image, backend="rekognition", reader=None, vlm_model=None):
        raise RuntimeError("zbar shared library not found")

    monkeypatch.setattr(fastag_module, "read_fastag", _boom)

    result = classify_fastag_upload("does-not-matter.jpg")
    assert result["checked"] is False
    assert "zbar shared library not found" in result["error"]


# --- Sticker completeness (is the whole FASTag actually captured?) ------------------

def test_check_fastag_completeness_passes_on_confident_complete_read(monkeypatch):
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: _complete_read(complete=True, confidence=95.0))

    result = check_fastag_completeness("does-not-matter.jpg")
    assert result.decision == "PASS"
    assert result.checked is True
    assert result.sticker_complete is True
    assert result.completeness_confidence == 95.0


def test_check_fastag_completeness_manual_review_on_confident_incomplete_read(monkeypatch):
    """A confident 'no' is trusted just like a confident 'yes' -- capped at
    MANUAL_REVIEW rather than REJECT, since this is an uncalibrated VLM judgment
    call, not a proven signal like Q1's own completeness gate."""
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: _complete_read(
                            complete=False, confidence=90.0, reason="barcode cut off at right edge"))

    result = check_fastag_completeness("does-not-matter.jpg")
    assert result.decision == "MANUAL_REVIEW"
    assert result.sticker_complete is False
    assert "barcode cut off" in result.reason


def test_check_fastag_completeness_manual_review_when_confidence_below_floor(monkeypatch):
    """A low-confidence read is too uncertain to act on either way -- MANUAL_REVIEW
    regardless of what sticker_complete itself says."""
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: _complete_read(complete=True, confidence=30.0))

    result = check_fastag_completeness("does-not-matter.jpg", conf_min=70.0)
    assert result.decision == "MANUAL_REVIEW"
    assert "too uncertain" in result.reason


def test_check_fastag_completeness_degrades_when_unavailable(monkeypatch):
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: {"checked": False, "error": "no ANTHROPIC_API_KEY"})

    result = check_fastag_completeness("does-not-matter.jpg")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert "no ANTHROPIC_API_KEY" in result.reason


def test_check_fastag_completeness_degrades_on_unknown_backend(monkeypatch):
    result = check_fastag_completeness("does-not-matter.jpg", backend="rekognition")
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False
    assert "unknown fastag-completeness backend" in result.reason


def test_check_fastag_upload_downgrades_to_manual_review_on_incomplete_sticker(monkeypatch):
    """A matching Tag ID isn't enough on its own -- if the sticker itself looks
    only partially captured, the overall decision must reflect that even though
    the identity match alone would have PASSed."""
    import vfiv.fastag_image.fastag_check as fastag_module

    monkeypatch.setattr(fastag_module, "classify_fastag_upload", lambda image, backend, vlm_model=None: _read(
        codes=[DecodedCode(symbology="QRCODE", data="607469009874936@icici")]))
    monkeypatch.setattr(fastag_module, "classify_fastag_completeness",
                        lambda image, backend=None, model=None: _complete_read(
                            complete=False, confidence=88.0, reason="printed digits cut off"))

    result = check_fastag_upload("does-not-matter.jpg", claimed_fastag_id="607469009874936")
    assert result.decision == "MANUAL_REVIEW"
    assert result.sticker_complete is False
    assert "printed digits cut off" in result.reason
