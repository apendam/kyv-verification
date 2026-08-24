"""FASTag validator — does the uploaded FASTag sticker photo belong to this truck?

Three independent representations of the tag's identity are cross-checked, both
against EACH OTHER and against the claimed value:
  - the QR code (``<fastag_id>@<bank_code>``) — decoded directly, most damage-
    tolerant of the three (built-in error correction)
  - the 1D barcode — decoded directly, checksum-backed
  - the printed human-readable digits — read via OCR, the only fuzzy/error-prone
    source of the three

Forging all three consistently (a valid QR + a valid checksummed barcode + matching
printed digits) is a much higher bar than editing the visible digits alone — so a
MISMATCH between sources that were each independently, legibly read is itself a
REJECT-worthy tamper signal, checked BEFORE comparing any of them to the claimed
value. See ``backends/fastag_reader.py`` for the raw reads.

Vehicle-class check: the real sample this was built from shows a printed class code
(e.g. "04") in the corner, not a reliably-photographable tag colour — verify this
holds across samples spanning multiple vehicle classes before trusting it; a colour
classifier (same shape as ``backends/plate_colour.py``) is a reasonable fallback if
it doesn't.
"""
from __future__ import annotations

import re

from truck_extract_match.plate.format import confusable_distance

from vfiv import config
from vfiv.backends.fastag_reader import FastagReadError, parse_qr_payload, read_fastag
from vfiv.schemas import FastagCheckResult


def _norm(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _fuzzy_match(a: str, b: str, max_edits: int) -> bool:
    if not a or not b:
        return False
    return confusable_distance(a, b) <= max_edits


def classify_fastag_upload(image) -> dict:
    """image: file path or PIL.Image. Real barcode/QR decode + OCR — no
    matching/decisioning here, see ``decide_fastag``."""
    try:
        read = read_fastag(image)
    except FastagReadError as e:
        return {"checked": False, "error": str(e)}
    return {"checked": True, "read": read}


def decide_fastag(
    r: dict,
    claimed_fastag_id: str,
    claimed_class_code: str | None = None,
    claimed_bank_code: str | None = None,
    max_ocr_edits: int = config.FASTAG_OCR_MAX_CONFUSABLE_EDITS,
) -> FastagCheckResult:
    """Pure decision logic over an already-read dict (``r["checked"]`` must be True
    — see ``classify_fastag_upload``).

    Order of operations:
      1. class-code mismatch -> instant REJECT (per the "wrong vehicle class ->
         return incorrect immediately" requirement)
      2. cross-source disagreement (sources that WERE legibly read disagree with
         EACH OTHER) -> REJECT — a stronger tamper signal than any single mismatch
         against the claimed value
      3. match against the claimed value: QR/barcode (exact, deterministic) first,
         OCR (fuzzy) only as a last resort
      4. nothing readable at all -> MANUAL_REVIEW, not REJECT — a bad photo isn't
         proof of fraud
    """
    read = r["read"]
    claimed_id_norm = _norm(claimed_fastag_id)

    if claimed_class_code and read.class_code_text:
        if _norm(read.class_code_text) != _norm(claimed_class_code):
            return FastagCheckResult(
                decision="REJECT",
                checked=True,
                claimed_fastag_id=claimed_fastag_id,
                claimed_class_code=claimed_class_code,
                claimed_bank_code=claimed_bank_code,
                extracted_class_code=read.class_code_text,
                reason=(f"tag class code '{read.class_code_text}' != claimed "
                        f"'{claimed_class_code}' — wrong vehicle class"),
            )

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
            claimed_class_code=claimed_class_code,
            claimed_bank_code=claimed_bank_code,
            extracted_class_code=read.class_code_text,
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
        claimed_class_code=claimed_class_code,
        claimed_bank_code=claimed_bank_code,
        extracted_class_code=read.class_code_text,
        decoded_sources=sources,
        extracted_printed_id=read.printed_id_text,
        matched_via=matched_via,
        reason=reason,
    )


def check_fastag_upload(
    image,
    claimed_fastag_id: str,
    claimed_class_code: str | None = None,
    claimed_bank_code: str | None = None,
    max_ocr_edits: int = config.FASTAG_OCR_MAX_CONFUSABLE_EDITS,
) -> FastagCheckResult:
    """Read then decide (single-call path). See ``decide_fastag`` for the decision
    logic and ``classify_fastag_upload`` for the raw reads."""
    r = classify_fastag_upload(image)
    if not r.get("checked"):
        return FastagCheckResult(
            decision="MANUAL_REVIEW",
            checked=False,
            claimed_fastag_id=claimed_fastag_id,
            claimed_class_code=claimed_class_code,
            claimed_bank_code=claimed_bank_code,
            reason=f"fastag check unavailable ({r.get('error', '?')})",
            error=r.get("error"),
        )
    return decide_fastag(r, claimed_fastag_id, claimed_class_code, claimed_bank_code, max_ocr_edits)
