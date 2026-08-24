"""Side/axle-image validator — does the axle count match, and does this side photo
actually belong to the SAME truck as the separately-uploaded front photo?

Three checks, each independently reported (a single prompt can't run a duplicate
search + a VLM axle count + an OCR/embedding identity check — see combined.py's
docstring for why this has to live in code, same reasoning applies here):

1. Duplicate check — reuses ``validators/duplicate_check.py``'s ``check_duplicate``
   with ``image_type="side"``, only if ``upload_id`` is given. Scoped to the
   "side" corpus only (``backends/vector_store.py``'s ``image_type`` column) so
   it's never compared against front or FASTag embeddings.

2. Axle count — no dedicated CV model wired. Counting axles from a 2D photo is a
   real, non-trivial CV problem in its own right (would need a custom-trained
   wheel/axle detector and a labeled dataset of Indian truck side profiles) — this
   uses a narrowed Claude VLM call instead, the same "no CV model does this
   reliably here" posture as Q1's screenshot/AI-generated checks. Known domain
   traps a single photo can't fully resolve: lift/tag axles raised off the ground,
   and dual/twin wheels on one axle (2 wheels != 2 axles) — the prompt asks the
   model to flag suspected lift axles rather than silently guess.

3. Identity-to-claimed-vehicle — routed by ``SideImageTypeClassifier``
   (``backends/siglip.py``) into three buckets of DECREASING reliability:
     a. vrn_visible       -> re-run Q2's own VRN detector/matcher on this image
                             (strongest — exact identity, reuses ``vrn_check.py``
                             as-is, no new logic).
     b. corner_view       -> make/model match (reuses Q3's SigLIP classifier) PLUS
                             a direct SigLIP embedding cosine-similarity between
                             this crop and the claimed truck's OWN on-file front
                             photo — a 1:1 pairwise compare, NOT a vector-DB
                             nearest-neighbor search. UNCALIBRATED: a general
                             embedding is trained for semantic similarity, not
                             individual-vehicle re-identification, so it may not
                             reliably separate "same truck, different angle" from
                             "different truck, same make/model/colour" — validate
                             ``config.SIDE_IMAGE_SIMILARITY_MIN`` against real
                             labeled pairs before trusting it.
     c. pure_side_profile  -> make/model match ONLY. The individual-vehicle
                             question is NOT solved here — this is the genuinely
                             open piece flagged in the design discussion, not an
                             integration task. A match at this level is capped at
                             MANUAL_REVIEW, never a confident PASS, because two
                             different trucks of the same make/model/colour would
                             pass this too.

Overall ``decision`` takes the worst of whichever checks actually ran — REJECT >
MANUAL_REVIEW > PASS, same ordering as ``combined.py``.
"""
from __future__ import annotations

import numpy as np

from truck_extract_match.make.aliases import match_make

from vfiv import config
from vfiv.backends.image_io import load_rgb_array
from vfiv.backends.siglip import get_make_classifier, get_side_image_type_classifier, get_siglip_model
from vfiv.backends.vehicle import get_vehicle_detector
from vfiv.schemas import SideImageCheckResult
from vfiv.validators.base import call_vlm_json
from vfiv.validators.duplicate_check import check_duplicate
from vfiv.validators.vrn_check import validate_vrn

_SEVERITY = {"REJECT": 2, "MANUAL_REVIEW": 1, "PASS": 0}

AXLE_PROMPT = """You are counting axles on an uploaded side-profile photo of a truck/bus,
for a document-validation platform. Count distinct AXLES (a pair of wheels sharing one
axle line), NOT individual wheels — dual/twin wheels mounted together on one axle still
count as ONE axle. If a lift/tag axle appears raised off the ground (not touching the
road), still include it in the total count, but set "lift_axle_suspected" to true. If the
full wheelbase isn't visible (cropped, obstructed, mid-corner shot), give your best count
from what IS visible rather than refusing, and say so in "reason".

Reply with STRICT JSON only:
{"axle_count":<int>,"confidence":0-100,"lift_axle_suspected":true|false,"reason":"<short>"}"""


def _worst_decision(*decisions: str) -> str:
    """Pure helper — REJECT > MANUAL_REVIEW > PASS, same ordering as combined.py."""
    return max(decisions, key=_SEVERITY.get)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


AXLE_COUNT_BACKENDS = ["claude", "gemini"]


def classify_axle_count(
    image,
    backend: str = config.AXLE_COUNT_BACKEND,
    model: str | None = None,
) -> dict:
    """VLM judgment call — see module docstring for why no dedicated detector is
    wired, and the real limitations (lift axles, dual wheels) this can't fully
    resolve from a single 2D photo. ``backend`` — "claude" (default) | "gemini" —
    same prompt either way, only which model reads it changes."""
    if backend == "claude":
        r = call_vlm_json(image, AXLE_PROMPT, model or config.VLM_MODEL, max_tokens=200)
    elif backend == "gemini":
        from vfiv.backends.gemini import call_gemini_json
        r = call_gemini_json(image, AXLE_PROMPT, model=model)
    else:
        raise ValueError(f"unknown axle-count backend: {backend!r} (expected one of {AXLE_COUNT_BACKENDS})")
    if not r.get("checked"):
        return r
    return {
        "checked": True,
        "axle_count": int(r.get("axle_count", 0) or 0),
        "axle_confidence": float(r.get("confidence", 0) or 0),
        "lift_axle_suspected": bool(r.get("lift_axle_suspected", False)),
        "reason": r.get("reason", ""),
    }


def decide_axle_count(
    r: dict,
    claimed_axle_count: int,
    conf_min: float = config.AXLE_COUNT_CONF_MIN,
) -> dict:
    """Pure decision logic over an already-read dict — MATCH/MISMATCH/UNREADABLE,
    same vocabulary as Q2/Q3's ``VerificationStatus``."""
    if r["axle_confidence"] < conf_min:
        return {
            "status": "UNREADABLE", "decision": "MANUAL_REVIEW",
            "reason": (f"axle read confidence {r['axle_confidence']:.0f}% "
                       f"< {conf_min:.0f}% — needs human count"),
        }
    if r["axle_count"] == claimed_axle_count:
        note = " (lift axle suspected — verify load state)" if r.get("lift_axle_suspected") else ""
        return {
            "status": "MATCH", "decision": "PASS",
            "reason": f"axle count {r['axle_count']} matches claimed {claimed_axle_count}{note}",
        }
    return {
        "status": "MISMATCH", "decision": "REJECT",
        "reason": f"axle count {r['axle_count']} != claimed {claimed_axle_count}",
    }


def _identity_via_vrn(image, claimed_vrn: str) -> tuple[str, str, dict]:
    vrn_result = validate_vrn(image, claimed_vrn)
    detail = {"bucket": "vrn_visible", "vrn_status": vrn_result.status}
    return vrn_result.decision, f"[vrn_visible] {vrn_result.reason}", detail


def _identity_via_corner_view(
    image, claimed_make: str, front_reference_image,
    similarity_min: float = config.SIDE_IMAGE_SIMILARITY_MIN,
) -> tuple[str, str, dict]:
    arr = load_rgb_array(image)
    det = get_vehicle_detector().best_truck(arr)
    crop = arr[det.bbox[1]:det.bbox[3], det.bbox[0]:det.bbox[2]] if det is not None else arr

    siglip_make = get_make_classifier().predict(crop)
    make_match = match_make(siglip_make["make"], claimed_make)

    similarity = None
    if front_reference_image is not None:
        siglip = get_siglip_model()
        emb_a = siglip.embed_image(crop)
        emb_b = siglip.embed_image(load_rgb_array(front_reference_image))
        similarity = _cosine(emb_a, emb_b)

    detail = {"bucket": "corner_view", "make_read": siglip_make["make"],
              "make_matched": make_match.matched, "front_similarity": similarity}

    if not make_match.matched:
        return "REJECT", f"[corner_view] make mismatch (read '{siglip_make['make']}')", detail
    if similarity is not None and similarity < similarity_min:
        return ("MANUAL_REVIEW",
                (f"[corner_view] make matched but front-similarity {similarity:.4f} "
                 f"< {similarity_min:.4f} — uncalibrated signal, human check"), detail)
    reason = "[corner_view] make matched" + (f", front-similarity {similarity:.4f}" if similarity is not None else "")
    return "PASS", reason, detail


def _identity_via_pure_side_profile(image, claimed_make: str) -> tuple[str, str, dict]:
    """Make/model-level ONLY — see module docstring. Never a confident PASS."""
    arr = load_rgb_array(image)
    det = get_vehicle_detector().best_truck(arr)
    crop = arr[det.bbox[1]:det.bbox[3], det.bbox[0]:det.bbox[2]] if det is not None else arr
    siglip_make = get_make_classifier().predict(crop)
    make_match = match_make(siglip_make["make"], claimed_make)
    detail = {"bucket": "pure_side_profile", "make_read": siglip_make["make"],
              "make_matched": make_match.matched}
    if not make_match.matched:
        return "REJECT", f"[pure_side_profile] make mismatch (read '{siglip_make['make']}')", detail
    return ("MANUAL_REVIEW",
            ("[pure_side_profile] make matches, but individual-vehicle identity "
             "isn't verifiable from a bare side profile alone — human check"), detail)


def check_side_image_upload(
    image,
    claimed_vrn: str,
    claimed_make: str,
    claimed_axle_count: int,
    upload_id: str | None = None,
    front_reference_image=None,
    axle_conf_min: float = config.AXLE_COUNT_CONF_MIN,
    side_image_similarity_min: float = config.SIDE_IMAGE_SIMILARITY_MIN,
    axle_backend: str = config.AXLE_COUNT_BACKEND,
    axle_model: str | None = None,
) -> SideImageCheckResult:
    """The single entry point for a side/axle-image upload. Runs duplicate check
    (if ``upload_id`` given), axle count, and identity-binding (routed by
    ``SideImageTypeClassifier``), then takes the worst decision across whichever
    checks ran — see module docstring for the full breakdown.

    ``front_reference_image`` is this truck's own already-accepted front photo,
    used by the corner-view bucket's embedding-similarity arm; without it, that
    bucket falls back to make/model-only (same ceiling as the pure-side-profile
    bucket).

    ``axle_backend`` — "claude" (default) | "gemini" — selects which model reads
    the axle count; the identity/duplicate checks are unaffected by this.
    """
    try:
        arr = load_rgb_array(image)

        dup = check_duplicate(image, upload_id, claimed_vrn, image_type="side") if upload_id else None

        axle_raw = classify_axle_count(image, backend=axle_backend, model=axle_model)
        if axle_raw.get("checked"):
            axle = decide_axle_count(axle_raw, claimed_axle_count, axle_conf_min)
        else:
            axle = {"status": "UNREADABLE", "decision": "MANUAL_REVIEW",
                    "reason": f"axle check unavailable ({axle_raw.get('error', '?')})"}

        bucket_probs = get_side_image_type_classifier().predict(arr)
        bucket = bucket_probs["bucket"]

        if bucket == "vrn_visible":
            identity_decision, identity_reason, identity_detail = _identity_via_vrn(image, claimed_vrn)
        elif bucket == "corner_view":
            identity_decision, identity_reason, identity_detail = _identity_via_corner_view(
                image, claimed_make, front_reference_image, side_image_similarity_min)
        else:
            identity_decision, identity_reason, identity_detail = _identity_via_pure_side_profile(
                image, claimed_make)
    except Exception as e:
        return SideImageCheckResult(
            decision="MANUAL_REVIEW", checked=False,
            claimed_vrn=claimed_vrn, claimed_make=claimed_make, claimed_axle_count=claimed_axle_count,
            reason=f"side-image check unavailable ({e})", error=str(e),
        )

    decisions = [axle["decision"], identity_decision]
    if dup is not None:
        decisions.append(dup.decision)
    overall = _worst_decision(*decisions)

    reason_parts = [f"axle: {axle['reason']}", f"identity: {identity_reason}"]
    if dup is not None:
        reason_parts.append(f"duplicate: {dup.reason}")

    return SideImageCheckResult(
        decision=overall,
        checked=True,
        reason="; ".join(reason_parts),
        claimed_vrn=claimed_vrn,
        claimed_make=claimed_make,
        claimed_axle_count=claimed_axle_count,
        axle_count=axle_raw.get("axle_count"),
        axle_status=axle.get("status"),
        identity_bucket=identity_detail.get("bucket"),
        identity_decision=identity_decision,
        duplicate_is_suspect=dup.is_duplicate_suspect if dup is not None else None,
    )
