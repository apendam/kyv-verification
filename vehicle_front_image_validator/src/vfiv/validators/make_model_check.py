"""Make + model validator — Q3: extracted make/model vs the values accompanying the
upload (same provisioning as VRN — see ``vrn_check.py``: whoever invokes this module
sends the claimed values; no DB lookup happens inside ``vfiv``).

Make comparison is ALWAYS enforced, checked against TWO independent real-model sources:
  - SigLIP 2 zero-shot brand classifier (``backends/siglip.py``, copied from
    truck_front_extractor) — always runs, but in testing misclassified 2 of 3 real Tata
    trucks as "Force Motors" (a known, documented SigLIP limitation: "zero-shot make is
    coarse; needs Qwen or a fine-tuned classifier").
  - AWS Rekognition's painted-brand-text read (a bonus field already extracted during
    Q2's VRN detection — ``backends/rekognition.py``) — real, but only present when a
    legible brand wordmark was actually detected in the image.
Matched if EITHER source agrees with the claimed make (``make_match_via`` reports which),
since each has independent blind spots and this is a mandatory, always-enforced check —
a single misclassifying model shouldn't false-reject a genuinely correct truck.

Model comparison is enforced only when the model read's own confidence is
>= ``MODEL_CONF_MIN`` (default 90) — a low-confidence read isn't trusted enough to hold
against the vehicle. Model designation has no curated list to canonicalise against like
make does, and Qwen2.5-VL (the real option for exact-model reading, ``backends/qwen.py``)
isn't run live in this environment (~16GB weights, minutes/image on CPU) — so this stays
a narrowed Claude VLM call for now, matched via `matching.token_set_ratio` (the same
generic fuzzy primitive `match_make` itself falls back on).
"""
import re

from truck_extract_match.core import VerificationStatus, decide_status
from truck_extract_match.make.aliases import match_make
from truck_extract_match.matching import token_set_ratio

from vfiv import config
from vfiv.backends.image_io import load_rgb_array
from vfiv.backends.rekognition import RekognitionCredentialError, detect_plate
from vfiv.backends.siglip import get_make_classifier
from vfiv.schemas import MakeModelCheckResult
from vfiv.validators.base import call_vlm_json

MODEL_PROMPT = """You are identifying the MODEL designation of an Indian truck/bus from its
front view, for a document-validation platform. The manufacturer (make) is read
separately by a real classifier — focus ONLY on the specific model/variant designation.

Read the MODEL designation if a badge/sticker showing it is legible (e.g. "407", "1616",
"PRO 3015", "LPT 1613", "1617R") — this is often on a small badge near the grille or on a
side panel and may not always be visible/legible from the front; leave it empty rather
than guessing if you're not confident.

Reply with STRICT JSON only:
{"model":"<model designation as read/badged, empty if unreadable>","model_confidence":0-100,"reason":"<short>"}

Definitions:
- model: the specific model/variant designation as badged, if legible. Empty string if
  not shown or not confidently legible — do not guess.
- model_confidence: 0-100, your confidence in the model reading (0 if empty).
- reason: one short phrase citing the deciding factor."""


def classify_make(image) -> dict:
    """image: file path, PIL.Image, or ndarray. Real make read from TWO sources —
    SigLIP zero-shot (always) + Rekognition's brand-text bonus field (best-effort;
    a Rekognition credential failure here just means that vote is unavailable, not a
    tech failure of the whole check — SigLIP alone still runs)."""
    siglip = get_make_classifier().predict(load_rgb_array(image))

    rekog_make = None
    try:
        det = detect_plate(image)
        rekog_make = det.make if det else None
    except RekognitionCredentialError:
        rekog_make = None

    return {
        "checked": True,
        "make_siglip": siglip["make"],
        "make_siglip_confidence": siglip["make_confidence"],
        "make_rekognition": rekog_make,
    }


def classify_model(image, model: str = config.VLM_MODEL) -> dict:
    """image: file path or PIL.Image. Claude reads just the model designation —
    make no longer depends on Claude at all (see ``classify_make``)."""
    r = call_vlm_json(image, MODEL_PROMPT, model)
    if not r.get("checked"):
        return r
    return {
        "checked": True,
        "model": str(r.get("model", "") or "").strip(),
        "model_confidence": float(r.get("model_confidence", 0)),
        "reason": r.get("reason", ""),
    }


def _normalize_model(s: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper()).split())


def _match_model(extracted: str, claimed: str, match_min: float) -> tuple[bool, float]:
    a, b = _normalize_model(extracted), _normalize_model(claimed)
    if not a or not b:
        return False, 0.0
    if a == b:
        return True, 1.0
    score = token_set_ratio(a, b)
    return score >= match_min, round(score, 4)


def decide_make_model(
    r: dict,
    claimed_make: str,
    claimed_model: str | None = None,
    model_conf_min: float = config.MODEL_CONF_MIN,
    model_match_min: float = config.MODEL_MATCH_MIN,
) -> MakeModelCheckResult:
    """Pure decision logic over an already-read dict (``r["checked"]`` must be True —
    see ``classify_make``/``classify_model``).

    PASS          make MATCHes via either real-model source, and model either isn't
                  checked or also MATCHes
    REJECT        make MISMATCHes on BOTH sources, OR model is checked and MISMATCHes
    MANUAL_REVIEW model is checked and UNREADABLE (make is never UNREADABLE here — SigLIP
                  is a closed-set classifier and always returns some brand)
    """
    siglip_match = match_make(r["make_siglip"], claimed_make)
    rekog_make = r.get("make_rekognition")
    rekog_match = match_make(rekog_make, claimed_make) if rekog_make else None

    matched = siglip_match.matched or bool(rekog_match and rekog_match.matched)
    if siglip_match.matched and rekog_match and rekog_match.matched:
        via = "both"
    elif siglip_match.matched:
        via = "siglip"
    elif rekog_match and rekog_match.matched:
        via = "rekognition"
    else:
        via = None
    make_status = decide_status(matched, r["make_siglip"] or rekog_make)

    model_checked = bool(claimed_model) and r["model_confidence"] >= model_conf_min
    model_status = None
    model_matched_score = None
    if model_checked:
        m_matched, score = _match_model(r["model"], claimed_model, model_match_min)
        model_status = decide_status(m_matched, r["model"] or None)
        model_matched_score = score

    detail = (f"make: siglip='{r['make_siglip']}'@{r['make_siglip_confidence']:.0f}% "
              f"({'MATCH' if siglip_match.matched else 'MISMATCH'})")
    detail += (f", rekognition='{rekog_make}' ({'MATCH' if rekog_match.matched else 'MISMATCH'})"
               if rekog_make else ", rekognition=no brand text found")
    detail += f" vs claimed '{claimed_make}' -> {make_status.value}"
    detail += f" via {via}" if via else ""
    if model_checked:
        detail += (f"; model: read '{r['model']}'@{r['model_confidence']:.0f}% vs "
                   f"claimed '{claimed_model}' ({model_status.value})")
    elif claimed_model:
        detail += (f"; model not checked (confidence {r['model_confidence']:.0f}% "
                   f"< {model_conf_min:.0f}%)")

    if model_checked and model_status is VerificationStatus.UNREADABLE:
        decision = "MANUAL_REVIEW"
    elif make_status is VerificationStatus.MISMATCH:
        decision = "REJECT"
    elif model_checked and model_status is VerificationStatus.MISMATCH:
        decision = "REJECT"
    else:
        decision = "PASS"

    return MakeModelCheckResult(
        decision=decision,
        reason=detail,
        checked=True,
        claimed_make=claimed_make,
        make_status=make_status.value,
        extracted_make_siglip=r["make_siglip"] or None,
        siglip_confidence=r["make_siglip_confidence"],
        extracted_make_rekognition=rekog_make,
        make_match_via=via,
        claimed_make_brands=sorted(siglip_match.claimed_brands) or None,
        claimed_model=claimed_model,
        model_checked=model_checked,
        model_status=model_status.value if model_status else None,
        extracted_model_raw=r["model"] or None,
        model_match_score=model_matched_score,
        model_confidence=r["model_confidence"],
    )


def validate_make_model(
    image,
    claimed_make: str,
    claimed_model: str | None = None,
    model_conf_min: float = config.MODEL_CONF_MIN,
    model_match_min: float = config.MODEL_MATCH_MIN,
) -> MakeModelCheckResult:
    """Read then decide. See ``decide_make_model`` for the decision logic.

    Make is real-model based (``classify_make``) and no longer depends on Claude at
    all — a Claude outage only affects the optional model arm, which degrades to "not
    checked" (confidence 0) rather than blocking the make decision.

    ``claimed_model`` is optional — omit it (or leave the model read below
    ``model_conf_min``) to check make only.
    """
    try:
        make_raw = classify_make(image)
    except Exception as e:
        return MakeModelCheckResult(
            decision="MANUAL_REVIEW",
            reason=f"make check unavailable ({e})",
            checked=False,
            claimed_make=claimed_make,
            claimed_model=claimed_model,
            error=str(e),
        )

    model_raw = classify_model(image)
    if not model_raw.get("checked"):
        model_raw = {"model": "", "model_confidence": 0.0,
                     "reason": f"model read unavailable ({model_raw.get('error', '?')})"}

    r = {**make_raw, **model_raw, "checked": True}
    return decide_make_model(r, claimed_make, claimed_model, model_conf_min, model_match_min)
