"""VRN (plate-number) + plate-colour validator — Q2: extracted VRN vs claimed
(internal-DB) VRN.

The read step is no longer Claude — it's AWS Rekognition (``backends/rekognition.py``,
copied from truck_verification_pipeline/step14_rekognition_detector.py: real text
detection + the full Indian-plate parsing logic — two-line handling, brand/slogan
rejection, O/I fuzzy-correction) for the plate number, and a real HSV pixel classifier
(``backends/plate_colour.py``, no model weights) on the extracted plate crop for colour.
Both replace Claude entirely; there is no VLM call anywhere in this module now.

The MATCH decision itself reuses, unchanged, the confusion-aware Indian-plate matching
engine from the sibling `truck_extract_match` module (`plate/format.py`): the state-code
whitelist, OCR-confusion groups (0/O/D/Q, 1/I/L, 2/Z, ...), smudge-inference, and grammar
validation — the exact logic already proven for Q2 in the truck verification workflow.
"""
import numpy as np
from truck_extract_match.core import VerificationStatus, blend_score, decide_status
from truck_extract_match.plate.format import match_vrn, normalize_vrn, parse_and_correct

from vfiv import config
from vfiv.backends.plate_colour import classify_plate_colour
from vfiv.backends.rekognition import RekognitionCredentialError, detect_plate
from vfiv.schemas import VrnCheckResult


def classify_plate(image) -> dict:
    """image: file path or PIL.Image. Real Rekognition read + HSV colour, no
    matching/decisioning. Shaped identically to the old Claude-based dict so
    ``decide_vrn`` needs no changes."""
    try:
        det = detect_plate(image)
    except RekognitionCredentialError as e:
        return {"checked": False, "error": str(e)}

    if det is None:
        return {"checked": True, "plate": "", "plate_confidence": 0.0,
                "plate_colour": "unknown", "colour_confidence": 0.0,
                "reason": "no plate detected"}

    colour, colour_conf = classify_plate_colour(np.array(det.crop))
    return {
        "checked": True,
        "plate": det.vrn or "",
        "plate_confidence": round(det.confidence * 100.0, 1),
        "plate_colour": colour,
        "colour_confidence": colour_conf,
        "reason": f"source={det.source}" + ("" if det.vrn else " (unparsed/no VRN text)"),
    }


def decide_vrn(
    r: dict,
    claimed_vrn: str,
    max_confusable_edits: int = config.VRN_MAX_CONFUSABLE_EDITS,
) -> VrnCheckResult:
    """Pure decision logic over an already-read dict (``r["checked"]`` must be True —
    see ``classify_plate``/``classify_combined``). Split out from ``validate_vrn`` so
    the combined single-call prompt (``validators/combined.py``) can reuse this exact
    decisioning without re-deriving it.

    PASS          MATCH — confusion-aware edit distance <= max_confusable_edits
    REJECT        MISMATCH — plate read, but doesn't match the claimed VRN
    MANUAL_REVIEW UNREADABLE — plate not visible / nothing usable read
    """
    norm = normalize_vrn(r["plate"])
    if not norm:
        return VrnCheckResult(
            decision="MANUAL_REVIEW",
            status=VerificationStatus.UNREADABLE.value,
            claimed_vrn=claimed_vrn,
            reason=f"plate not read  [{r['reason']}]",
            checked=True,
            plate_colour=r["plate_colour"],
            colour_confidence=r["colour_confidence"],
        )

    vm = match_vrn(norm, claimed_vrn, max_confusable_edits)
    parse = parse_and_correct(r["plate"])
    status = decide_status(vm.matched, vm.read_norm)
    blended = blend_score(vm.score, r["plate_confidence"] / 100.0)
    decision = "PASS" if status is VerificationStatus.MATCH else "REJECT"

    detail = (f"read '{vm.read_norm}' vs claimed '{vm.claimed_norm}' "
              f"(confusable_dist={vm.distance}, grammar_valid={parse.valid}, "
              f"colour={r['plate_colour']}@{r['colour_confidence']:.0f}%)")
    if vm.inferred:
        detail += " — smudge-inferred match"

    return VrnCheckResult(
        decision=decision,
        status=status.value,
        claimed_vrn=vm.claimed_norm,
        reason=detail,
        checked=True,
        extracted_raw=r["plate"],
        extracted_norm=parse.corrected or vm.read_norm,
        match_score=blended,
        read_confidence=r["plate_confidence"],
        confusable_distance=vm.distance,
        grammar_valid=parse.valid,
        inferred=vm.inferred,
        plate_colour=r["plate_colour"],
        colour_confidence=r["colour_confidence"],
    )


def validate_vrn(
    image,
    claimed_vrn: str,
    max_confusable_edits: int = config.VRN_MAX_CONFUSABLE_EDITS,
) -> VrnCheckResult:
    """Read then decide (single-call path). See ``decide_vrn`` for the decision logic
    and ``classify_plate`` for the VLM call.

    ``claimed_vrn`` is whatever the caller sends alongside the image when invoking
    this module — where that value comes from (a real platform integration vs. a
    manual test-input field) is outside this module's concern.
    """
    r = classify_plate(image)
    if not r.get("checked"):
        return VrnCheckResult(
            decision="MANUAL_REVIEW",
            status=VerificationStatus.UNREADABLE.value,
            claimed_vrn=claimed_vrn,
            reason=f"VRN check unavailable ({r.get('error', '?')})",
            checked=False,
            error=r.get("error"),
        )
    return decide_vrn(r, claimed_vrn, max_confusable_edits)
