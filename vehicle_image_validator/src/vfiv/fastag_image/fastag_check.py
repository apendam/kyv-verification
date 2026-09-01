"""FASTag validator — does the uploaded FASTag sticker photo belong to this truck,
and is the WHOLE sticker actually captured in the photo?

Three independent representations of the tag's identity are cross-checked, both
against EACH OTHER and against the claimed value:
  - the QR code — decoded directly, most damage-tolerant of the three (built-in
    error correction). Encodes a UPI deep link, not a bare
    ``<fastag_id>@<bank_code>`` string — see ``backends/fastag_reader.py``'s
    ``parse_qr_payload`` for the real format.
  - the 1D barcode — decoded directly, checksum-backed
  - the printed human-readable digits — read via OCR, the only fuzzy/error-prone
    source of the three

Forging all three consistently (a valid QR + a valid checksummed barcode + matching
printed digits) is a much higher bar than editing the visible digits alone — so a
MISMATCH between sources that were each independently, legibly read is itself a
REJECT-worthy tamper signal, checked BEFORE comparing any of them to the claimed
value. See ``backends/fastag_reader.py`` for the raw reads.

A fourth, independent check — ``check_fastag_completeness`` — asks whether the
whole sticker (QR + barcode + printed digits) is actually in frame, not just
whether whatever WAS captured happens to read/match. A photo cropped tight to just
the QR code can still legitimately PASS the identity check above (the QR alone is
enough to match); this check exists to flag that as a framing/photo-quality issue
in its own right. No dedicated sticker detector exists to check this with a
bounding box (unlike the side-image truck check), so it's a narrow VLM judgment
call instead — same "no CV model does this reliably here" posture as
side_image_check.py's axle-count/bucket-routing. UNCALIBRATED — capped at
MANUAL_REVIEW, never a solo REJECT.
"""
from __future__ import annotations

import re

from truck_extract_match.plate.format import confusable_distance

from vfiv import config
from vfiv.backends.fastag_reader import FastagReadError, parse_qr_payload, read_fastag
from vfiv.base import call_vlm_json
from vfiv.duplicate_check import check_duplicate
from vfiv.schemas import (
    FastagCheckResult,
    FastagCompletenessResult,
    PrintedDigitsOnlyResult,
    QrOnlyResult,
)


def _norm(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _fuzzy_match(a: str, b: str, max_edits: int) -> bool:
    if not a or not b:
        return False
    return confusable_distance(a, b) <= max_edits


def classify_fastag_upload(
    image,
    backend: str = config.FASTAG_OCR_BACKEND,
    vlm_model: str | None = None,
) -> dict:
    """image: file path or PIL.Image. Real barcode/QR decode + OCR — no
    matching/decisioning here, see ``decide_fastag``. ``backend`` selects the
    printed-digit OCR source ("rekognition" | "claude" | "gemini") — see
    ``backends/fastag_reader.py``; the barcode/QR decode is unaffected either way.

    Catches ANY exception, not just ``FastagReadError`` — same "never crash,
    always degrade" posture as every other ``classify_*``/``check_*`` function in
    this codebase (``check_axle_count``, ``check_side_identity``, etc.). A missing
    OS-level dependency (e.g. ``libzbar0`` for ``pyzbar`` — see module docstring)
    or an unexpected SDK error would otherwise propagate uncaught straight through
    to the webapp as a bare "Error" with no message, instead of a clear reason."""
    try:
        read = read_fastag(image, backend=backend, vlm_model=vlm_model)
    except FastagReadError as e:
        return {"checked": False, "error": str(e)}
    except Exception as e:
        return {"checked": False, "error": f"unexpected error reading FASTag ({e})"}
    return {"checked": True, "read": read}


FASTAG_COMPLETENESS_PROMPT = """You are checking whether the FASTag sticker in this
uploaded photo is FULLY captured, for a document-validation platform. A FASTag sticker
has three parts that all need to be visible in the frame:
- a QR code (a square barcode block)
- a 1D barcode (a strip of vertical bars)
- a printed human-readable digit string below/near the barcode (e.g.
  "607469-009-0874936")

Decide whether ALL THREE parts are visible within the photo's frame -- not cut off at
an edge, not entirely covered by a finger/glare/reflection, and not so far away or
blurry that a part is unreadable in principle (rather than just momentarily hard to
read). A photo that only shows the QR code with the barcode/printed digits cut off or
out of frame is NOT complete, even if the QR code itself is perfectly legible.

Reply with STRICT JSON only:
{"reason":"<short -- which parts are visible, which (if any) are missing/cut off/obscured>","sticker_complete":true|false,"confidence":0-100}"""

FASTAG_COMPLETENESS_BACKENDS = ["claude", "gemini"]


def classify_fastag_completeness(
    image,
    backend: str = config.FASTAG_COMPLETENESS_BACKEND,
    model: str | None = None,
) -> dict:
    """VLM judgment call — see module docstring for why no dedicated sticker
    detector is wired. ``backend`` — "claude" (default) | "gemini"; "rekognition"
    isn't an option here, unlike the printed-digit OCR backend, since Rekognition
    has no notion of "FASTag sticker", only generic text/object detection."""
    if backend == "claude":
        r = call_vlm_json(image, FASTAG_COMPLETENESS_PROMPT, model or config.VLM_MODEL, max_tokens=200)
    elif backend == "gemini":
        from vfiv.backends.gemini import call_gemini_json
        r = call_gemini_json(image, FASTAG_COMPLETENESS_PROMPT, model=model)
    else:
        raise ValueError(f"unknown fastag-completeness backend: {backend!r} "
                         f"(expected one of {FASTAG_COMPLETENESS_BACKENDS})")
    return r


def check_fastag_completeness(
    image,
    backend: str = config.FASTAG_COMPLETENESS_BACKEND,
    model: str | None = None,
    conf_min: float = config.FASTAG_COMPLETENESS_CONF_MIN,
) -> FastagCompletenessResult:
    """Is the whole sticker (QR + barcode + printed digits) actually captured in
    this photo? Exposed standalone so it's independently testable from the
    identity/match checks below — see module docstring for why this exists as its
    own check rather than folding into ``decide_fastag``'s "nothing readable"
    path.

    ``conf_min`` gates on the VLM's own self-reported confidence (same idea as
    ``AXLE_COUNT_CONF_MIN``): below it, the read is too uncertain to act on
    either way and this falls back to MANUAL_REVIEW regardless of what
    ``sticker_complete`` says — a confident "yes" is trusted, but so is a
    confident "no"; only an unsure read gets the benefit of the doubt."""
    try:
        r = classify_fastag_completeness(image, backend=backend, model=model)
    except Exception as e:
        return FastagCompletenessResult(
            decision="MANUAL_REVIEW", checked=False,
            reason=f"completeness check unavailable ({e})", error=str(e),
        )
    if not r.get("checked"):
        return FastagCompletenessResult(
            decision="MANUAL_REVIEW", checked=False,
            reason=f"completeness check unavailable ({r.get('error', '?')})", error=r.get("error"),
        )
    complete = bool(r.get("sticker_complete", False))
    confidence = float(r.get("confidence", 0) or 0)
    read_reason = r.get("reason", "")

    if confidence < conf_min:
        return FastagCompletenessResult(
            decision="MANUAL_REVIEW", checked=True, sticker_complete=complete,
            completeness_confidence=confidence,
            reason=(f"completeness read confidence {confidence:.0f}% < {conf_min:.0f}% "
                    f"— too uncertain to call either way ({read_reason})"),
        )
    if complete:
        return FastagCompletenessResult(
            decision="PASS", checked=True, sticker_complete=True, completeness_confidence=confidence,
            reason=f"sticker appears fully captured ({read_reason})" if read_reason
                   else "sticker appears fully captured",
        )
    return FastagCompletenessResult(
        decision="MANUAL_REVIEW", checked=True, sticker_complete=False, completeness_confidence=confidence,
        reason=(f"sticker may not be fully captured ({read_reason}) — "
                f"uncalibrated VLM judgment, human check"),
    )


def decide_fastag(
    r: dict,
    claimed_fastag_id: str,
    claimed_bank_code: str | None = None,
    max_ocr_edits: int = config.FASTAG_OCR_MAX_CONFUSABLE_EDITS,
) -> FastagCheckResult:
    """Pure decision logic over an already-read dict (``r["checked"]`` must be True
    — see ``classify_fastag_upload``).

    Order of operations:
      1. cross-source disagreement (sources that WERE legibly read disagree with
         EACH OTHER) -> REJECT — a stronger tamper signal than any single mismatch
         against the claimed value
      2. match against the claimed value: QR/barcode (exact, deterministic) first,
         OCR (fuzzy) only as a last resort
      3. nothing readable at all -> MANUAL_REVIEW, not REJECT — a bad photo isn't
         proof of fraud
    """
    read = r["read"]
    claimed_id_norm = _norm(claimed_fastag_id)

    sources: dict[str, str] = {}
    qr_bank_code = None
    for code in read.decoded_codes:
        if code.symbology.upper() == "QRCODE":
            fastag_id, bank_code = parse_qr_payload(code.data)
            if fastag_id:
                sources["qr"] = _norm(fastag_id)
                qr_bank_code = bank_code
        else:
            sources[f"barcode:{code.symbology.lower()}"] = _norm(code.data)
    ocr_val = _norm(read.printed_id_text) if read.printed_id_text else None

    decoded_values = set(sources.values())
    cross_consistent = len(decoded_values) <= 1
    if ocr_val and decoded_values and ocr_val not in decoded_values:
        cross_consistent = False

    if not cross_consistent:
        return FastagCheckResult(
            decision="REJECT",
            checked=True,
            claimed_fastag_id=claimed_fastag_id,
            claimed_bank_code=claimed_bank_code,
            decoded_sources=sources,
            extracted_printed_id=read.printed_id_text,
            reason=(f"identity sources disagree with each other "
                    f"(decoded={sources}, printed_ocr={read.printed_id_text!r}) "
                    f"— possible tampering"),
        )

    matched_via = None
    if sources.get("qr") == claimed_id_norm:
        matched_via = "qr"
    else:
        for k, v in sources.items():
            if k != "qr" and v == claimed_id_norm:
                matched_via = k
                break
    if not matched_via and ocr_val and _fuzzy_match(ocr_val, claimed_id_norm, max_ocr_edits):
        matched_via = "ocr"

    if matched_via:
        decision = "PASS"
        reason = f"fastag id matched via {matched_via}"
        if (matched_via == "qr" and claimed_bank_code and qr_bank_code
                and _norm(qr_bank_code) != _norm(claimed_bank_code)):
            decision = "MANUAL_REVIEW"
            reason += (f"; bank code mismatch ('{qr_bank_code}' vs claimed "
                       f"'{claimed_bank_code}') — verify tag/bank record")
    elif sources or ocr_val:
        decision = "REJECT"
        reason = (f"read identity ({sources or {'ocr': ocr_val}}) doesn't match "
                  f"claimed '{claimed_fastag_id}'")
    else:
        decision = "MANUAL_REVIEW"
        reason = "no barcode, QR, or printed number could be read from this image"

    return FastagCheckResult(
        decision=decision,
        checked=True,
        claimed_fastag_id=claimed_fastag_id,
        claimed_bank_code=claimed_bank_code,
        decoded_sources=sources,
        extracted_printed_id=read.printed_id_text,
        matched_via=matched_via,
        reason=reason,
    )


def decide_qr_only(
    r: dict,
    claimed_fastag_id: str,
    claimed_bank_code: str | None = None,
) -> QrOnlyResult:
    """Pure decision logic for the QR code ALONE — exact match only (the QR's
    built-in error correction makes it damage-tolerant but exactly decoded, no
    fuzzy tolerance needed), independent of ``decide_fastag``'s cross-source
    consistency check (which needs all three sources together)."""
    read = r["read"]
    qr_tag_id = qr_bank_code = None
    for code in read.decoded_codes:
        if code.symbology.upper() == "QRCODE":
            qr_tag_id, qr_bank_code = parse_qr_payload(code.data)
            if qr_tag_id:
                break

    if not qr_tag_id:
        return QrOnlyResult(
            decision="MANUAL_REVIEW", status="UNREADABLE", checked=True,
            claimed_fastag_id=claimed_fastag_id, claimed_bank_code=claimed_bank_code,
            reason="no QR code could be decoded from this image",
        )
    if _norm(qr_tag_id) != _norm(claimed_fastag_id):
        return QrOnlyResult(
            decision="REJECT", status="MISMATCH", checked=True,
            claimed_fastag_id=claimed_fastag_id, claimed_bank_code=claimed_bank_code,
            qr_tag_id=qr_tag_id, qr_bank_code=qr_bank_code,
            reason=f"QR tag id {qr_tag_id!r} != claimed {claimed_fastag_id!r}",
        )
    if claimed_bank_code and qr_bank_code and _norm(qr_bank_code) != _norm(claimed_bank_code):
        return QrOnlyResult(
            decision="MANUAL_REVIEW", status="MATCH", checked=True,
            claimed_fastag_id=claimed_fastag_id, claimed_bank_code=claimed_bank_code,
            qr_tag_id=qr_tag_id, qr_bank_code=qr_bank_code,
            reason=(f"QR tag id matched, but bank code mismatch ({qr_bank_code!r} vs "
                   f"claimed {claimed_bank_code!r}) — verify tag/bank record"),
        )
    return QrOnlyResult(
        decision="PASS", status="MATCH", checked=True,
        claimed_fastag_id=claimed_fastag_id, claimed_bank_code=claimed_bank_code,
        qr_tag_id=qr_tag_id, qr_bank_code=qr_bank_code,
        reason=f"QR tag id matches claimed {claimed_fastag_id!r}",
    )


def decide_printed_digits_only(
    r: dict,
    claimed_barcode: str,
    max_ocr_edits: int = config.FASTAG_OCR_MAX_CONFUSABLE_EDITS,
) -> PrintedDigitsOnlyResult:
    """Pure decision logic for the printed human-readable digits ALONE — fuzzy
    match (OCR is the fuzzy/error-prone source of the three) against a
    separately-claimed barcode value, e.g. from a dedicated handheld barcode
    scan rather than decoded from this same photo. Independent of
    ``decide_fastag``'s cross-source consistency check."""
    printed = r["read"].printed_id_text
    if not printed:
        return PrintedDigitsOnlyResult(
            decision="MANUAL_REVIEW", status="UNREADABLE", checked=True,
            claimed_barcode=claimed_barcode,
            reason="no printed digits could be read from this image",
        )
    if _norm(printed) == _norm(claimed_barcode) or _fuzzy_match(_norm(printed), _norm(claimed_barcode), max_ocr_edits):
        return PrintedDigitsOnlyResult(
            decision="PASS", status="MATCH", checked=True,
            claimed_barcode=claimed_barcode, extracted_printed_id=printed,
            reason=f"printed digits {printed!r} match claimed barcode {claimed_barcode!r}",
        )
    return PrintedDigitsOnlyResult(
        decision="REJECT", status="MISMATCH", checked=True,
        claimed_barcode=claimed_barcode, extracted_printed_id=printed,
        reason=f"printed digits {printed!r} != claimed barcode {claimed_barcode!r}",
    )


def check_qr_only(
    image,
    claimed_fastag_id: str,
    claimed_bank_code: str | None = None,
    backend: str = config.FASTAG_OCR_BACKEND,
    vlm_model: str | None = None,
) -> QrOnlyResult:
    """Read then decide (single-call path) for the QR code alone. ``backend``
    only affects the (unused-here) OCR read; the QR decode itself is
    deterministic either way."""
    r = classify_fastag_upload(image, backend=backend, vlm_model=vlm_model)
    if not r.get("checked"):
        return QrOnlyResult(
            decision="MANUAL_REVIEW", checked=False,
            claimed_fastag_id=claimed_fastag_id, claimed_bank_code=claimed_bank_code,
            reason=f"fastag check unavailable ({r.get('error', '?')})", error=r.get("error"),
        )
    return decide_qr_only(r, claimed_fastag_id, claimed_bank_code)


def check_printed_digits_only(
    image,
    claimed_barcode: str,
    backend: str = config.FASTAG_OCR_BACKEND,
    vlm_model: str | None = None,
    max_ocr_edits: int = config.FASTAG_OCR_MAX_CONFUSABLE_EDITS,
) -> PrintedDigitsOnlyResult:
    """Read then decide (single-call path) for the printed digits alone.
    ``backend`` — "rekognition" (default) | "claude" | "gemini" — selects the
    printed-digit OCR source, same as ``check_fastag_upload``."""
    r = classify_fastag_upload(image, backend=backend, vlm_model=vlm_model)
    if not r.get("checked"):
        return PrintedDigitsOnlyResult(
            decision="MANUAL_REVIEW", checked=False, claimed_barcode=claimed_barcode,
            reason=f"fastag check unavailable ({r.get('error', '?')})", error=r.get("error"),
        )
    return decide_printed_digits_only(r, claimed_barcode, max_ocr_edits)


_SEVERITY = {"REJECT": 2, "MANUAL_REVIEW": 1, "PASS": 0}


def check_fastag_upload(
    image,
    claimed_fastag_id: str,
    claimed_bank_code: str | None = None,
    max_ocr_edits: int = config.FASTAG_OCR_MAX_CONFUSABLE_EDITS,
    backend: str = config.FASTAG_OCR_BACKEND,
    vlm_model: str | None = None,
    claimed_vrn: str | None = None,
    upload_id: str | None = None,
    completeness_backend: str = config.FASTAG_COMPLETENESS_BACKEND,
    completeness_model: str | None = None,
    completeness_conf_min: float = config.FASTAG_COMPLETENESS_CONF_MIN,
) -> FastagCheckResult:
    """Read then decide (single-call path). See ``decide_fastag`` for the decision
    logic and ``classify_fastag_upload`` for the raw reads. ``backend`` — "rekognition"
    (default) | "claude" | "gemini" — selects the printed-digit OCR source only; the
    barcode/QR decode always runs the same way regardless.

    Always also runs ``check_fastag_completeness`` (see module docstring) and folds
    its verdict in, worst-of — unlike duplicate below, this isn't opt-in, since it
    needs nothing beyond the image itself. ``completeness_backend``/
    ``completeness_model``/``completeness_conf_min`` override its defaults
    per-call, independent of the OCR ``backend`` above (Rekognition can't run this
    check at all — see ``config.FASTAG_COMPLETENESS_BACKEND``).

    Pass BOTH ``claimed_vrn`` and ``upload_id`` to also run the cross-upload
    duplicate check (``duplicate_check.py``, ``image_type="fastag"``) and fold its
    verdict into the decision — same opt-in pattern as Q1/side's own duplicate
    checks. Leaving either blank skips it."""
    r = classify_fastag_upload(image, backend=backend, vlm_model=vlm_model)
    if not r.get("checked"):
        return FastagCheckResult(
            decision="MANUAL_REVIEW",
            checked=False,
            claimed_fastag_id=claimed_fastag_id,
            claimed_bank_code=claimed_bank_code,
            reason=f"fastag check unavailable ({r.get('error', '?')})",
            error=r.get("error"),
        )
    result = decide_fastag(r, claimed_fastag_id, claimed_bank_code, max_ocr_edits)

    completeness = check_fastag_completeness(
        image, backend=completeness_backend, model=completeness_model,
        conf_min=completeness_conf_min,
    )
    result = result.model_copy(update={
        "decision": max([result.decision, completeness.decision], key=_SEVERITY.get),
        "reason": f"{result.reason}; completeness: {completeness.reason}",
        "sticker_complete": completeness.sticker_complete,
        "sticker_completeness_confidence": completeness.completeness_confidence,
    })

    if not (claimed_vrn and upload_id):
        return result

    dup = check_duplicate(image, upload_id, claimed_vrn, image_type="fastag")
    return result.model_copy(update={
        "decision": max([result.decision, dup.decision], key=_SEVERITY.get),
        "reason": f"{result.reason}; duplicate: {dup.reason}",
        "duplicate_is_suspect": dup.is_duplicate_suspect,
        "duplicate_matches": dup.duplicate_matches,
    })
