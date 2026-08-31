"""Side/axle-image validator — does the axle count match, and does this side photo
actually belong to the SAME truck as the separately-uploaded front photo?

Three checks, each independently reported (a single prompt can't run a duplicate
search + a VLM axle count + an OCR/embedding identity check — see combined.py's
docstring for why this has to live in code, same reasoning applies here):

1. Duplicate check — reuses ``duplicate_check.py``'s ``check_duplicate``
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
     b. corner_view       -> a direct SigLIP embedding cosine-similarity between
                             this crop and the claimed truck's OWN on-file front
                             photo — a 1:1 pairwise compare, NOT a vector-DB
                             nearest-neighbor search. No make check here at all
                             (see below for why) — without a
                             ``front_reference_image`` there's simply nothing to
                             compare against, so this bucket is MANUAL_REVIEW
                             ("unverifiable") rather than falling back to a
                             weaker signal. UNCALIBRATED: a general embedding is
                             trained for semantic similarity, not individual-
                             vehicle re-identification, so it may not reliably
                             separate "same truck, different angle" from
                             "different truck, same make/model/colour" — validate
                             ``config.SIDE_IMAGE_SIMILARITY_MIN`` against real
                             labeled pairs before trusting it. Now this bucket's
                             ONLY signal (see below), so that validation matters
                             more than it used to.
     c. pure_side_profile  -> make/model match ONLY (reuses Q3's SigLIP zero-shot
                             classifier) — this is the ONE bucket that still uses
                             it, since there's nothing else to go on for a bare
                             side profile (no plate, no front grille, no
                             embedding to compare). Deliberately NOT used in
                             corner_view any more: it's a coarse, brand-only
                             zero-shot read (compares the image against 8 fixed
                             brand-name text prompts, never actually reads
                             painted logos/text) that can misfire between
                             visually-similar cab shapes across manufacturers —
                             see ``backends/siglip.py``'s ``MakeClassifier`` — so
                             it should never be the thing that gates a bucket
                             that has a stronger identity signal available. The
                             individual-vehicle question is NOT solved here
                             either way — this is the genuinely open piece
                             flagged in the design discussion, not an
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
from vfiv.schemas import AxleCountResult, SideImageCheckResult, SideImageIdentityResult
from vfiv.base import call_vlm_json
from vfiv.duplicate_check import check_duplicate
from vfiv.front_image.vrn_check import validate_vrn

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


def _color_histogram_similarity(crop_a: np.ndarray, crop_b: np.ndarray, bins: int = 32) -> float:
    """Correlation between two RGB colour histograms, roughly -1..1 (1 = identical
    colour distribution). A real, deterministic paint-colour comparison -- unlike
    the embedding-similarity check's general semantic notion of "looks similar",
    this only cares about colour. Callers MUST pass vehicle-only crops (e.g. from
    ``get_vehicle_detector().best_truck()``), never full uncropped photos -- the
    surrounding road/sky/background would otherwise dominate the histogram and
    swamp the actual paint-colour signal."""
    def _hist(crop: np.ndarray) -> np.ndarray:
        channels = [np.histogram(crop[..., c], bins=bins, range=(0, 255))[0].astype(np.float64)
                    for c in range(3)]
        h = np.concatenate(channels)
        return h / (h.sum() + 1e-8)

    ha, hb = _hist(crop_a), _hist(crop_b)
    ha, hb = ha - ha.mean(), hb - hb.mean()
    denom = float(np.sqrt((ha ** 2).sum() * (hb ** 2).sum()))
    return float((ha * hb).sum() / denom) if denom else 0.0


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


# RC-derived vehicle mapper class -> the axle count that class implies, straight
# from the classification table (same source as the "auto-filled" axle field, just
# one hop removed via the vehicle's body/weight class rather than a direct RC
# field). Several mapper classes share an axle count (e.g. VC20/VC9/VC7/VC5/VC10
# are all 2-axle, split by weight/seat bands, not by axle count) -- that's fine,
# this table only needs to answer "what axle count does THIS class imply", not the
# reverse. "Car" (VC4) has no axle count of its own in the source table.
VEHICLE_MAPPER_AXLE_COUNT: dict[str, int | None] = {
    "VC4": None,   # Car
    "VC20": 2,     # Bus/Truck <7.5T
    "VC9": 2,      # Bus <12T, seats < 32
    "VC7": 2,      # Bus <12T seats > 32, or Bus >12T
    "VC8": 3,      # Bus, 3-axle
    "VC5": 2,      # Truck >7.5T & <12T
    "VC10": 2,     # Truck >12T
    "VC11": 3,     # Truck, 3-axle
    "VC12": 4,     # Truck, 4-axle
    "VC13": 5,     # Truck, 5-axle
    "VC14": 6,     # Truck, 6-axle
    "VC15": 7,     # Truck, 7-axle
}

_UNKNOWN_MAPPER = object()


def decide_axle_source_consistency(
    claimed_axle_count: int,
    axle_source: str,
    vehicle_mapper: str | None,
) -> dict:
    """Pure data-consistency check -- no image involved at all. ``axle_source``
    "auto" means ``claimed_axle_count`` was pulled straight from the RC and is
    trusted as-is. "manual" means a field agent typed it in, so it's cross-checked
    against ``vehicle_mapper``'s own RC-derived fixed axle count instead --
    catching a fabricated count even before the photo is looked at."""
    if axle_source == "auto":
        return {"decision": "PASS", "status": "MATCH",
                "reason": f"axle count {claimed_axle_count} auto-filled from RC — trusted as-is"}
    if axle_source != "manual":
        return {"decision": "MANUAL_REVIEW", "status": "UNREADABLE",
                "reason": f"unknown axle_source {axle_source!r} (expected 'auto' or 'manual')"}
    if not vehicle_mapper:
        return {"decision": "MANUAL_REVIEW", "status": "UNREADABLE",
                "reason": "manually-entered axle count has no vehicle mapper class to cross-check against"}

    expected = VEHICLE_MAPPER_AXLE_COUNT.get(vehicle_mapper, _UNKNOWN_MAPPER)
    if expected is _UNKNOWN_MAPPER:
        return {"decision": "MANUAL_REVIEW", "status": "UNREADABLE",
                "reason": f"unknown vehicle mapper class {vehicle_mapper!r}"}
    if expected is None:
        return {"decision": "MANUAL_REVIEW", "status": "UNREADABLE",
                "reason": f"vehicle mapper class {vehicle_mapper!r} has no defined axle count"}
    if claimed_axle_count == expected:
        return {"decision": "PASS", "status": "MATCH",
                "reason": (f"manually-entered axle count {claimed_axle_count} matches vehicle "
                           f"mapper {vehicle_mapper!r}'s expected {expected}")}
    return {"decision": "REJECT", "status": "MISMATCH",
            "reason": (f"manually-entered axle count {claimed_axle_count} != vehicle mapper "
                       f"{vehicle_mapper!r}'s expected {expected}")}


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


def _truck_crop(image) -> np.ndarray:
    arr = load_rgb_array(image)
    det = get_vehicle_detector().best_truck(arr)
    return arr[det.bbox[1]:det.bbox[3], det.bbox[0]:det.bbox[2]] if det is not None else arr


def _identity_via_corner_view(
    image, front_reference_image,
    similarity_min: float = config.SIDE_IMAGE_SIMILARITY_MIN,
    color_hist_min: float = config.SIDE_IMAGE_COLOR_HIST_MIN,
) -> tuple[str, str, dict]:
    """Identity here rests on TWO signals against this truck's OWN on-file front
    photo -- a SigLIP embedding comparison (general "looks like the same vehicle")
    and a colour-histogram comparison (specifically "same paint colour") -- both
    much stronger than the generic 8-brand make classifier (see
    ``_identity_via_pure_side_profile``), which can't tell THIS Tata from any other
    Tata, only "Tata-shaped or not" (and not even reliably that -- see
    MakeClassifier's docstring). No make check here at all; without a
    ``front_reference_image`` there is nothing to compare against, so identity is
    simply unverifiable from a corner view alone. Both crops are vehicle-only (via
    the detector), never the raw uncropped photo -- background/road/sky colour
    would otherwise swamp the histogram signal."""
    detail = {"bucket": "corner_view", "front_similarity": None, "color_hist_similarity": None}

    if front_reference_image is None:
        return ("MANUAL_REVIEW",
                "[corner_view] no front reference photo supplied — identity isn't "
                "verifiable from a corner view alone", detail)

    crop = _truck_crop(image)
    ref_crop = _truck_crop(front_reference_image)

    siglip = get_siglip_model()
    similarity = _cosine(siglip.embed_image(crop), siglip.embed_image(ref_crop))
    detail["front_similarity"] = similarity

    color_similarity = _color_histogram_similarity(crop, ref_crop)
    detail["color_hist_similarity"] = color_similarity

    failures = []
    if similarity < similarity_min:
        failures.append(f"front-similarity {similarity:.4f} < {similarity_min:.4f}")
    if color_similarity < color_hist_min:
        failures.append(f"colour-histogram similarity {color_similarity:.4f} < {color_hist_min:.4f}")

    if failures:
        return ("MANUAL_REVIEW",
                "[corner_view] " + "; ".join(failures) + " — uncalibrated signal(s), human check",
                detail)
    return ("PASS",
            f"[corner_view] front-similarity {similarity:.4f}, colour-histogram {color_similarity:.4f}",
            detail)


def _identity_via_pure_side_profile(image, claimed_make: str) -> tuple[str, str, dict]:
    """Make/model-level ONLY — see module docstring. Never a confident PASS."""
    crop = _truck_crop(image)
    siglip_make = get_make_classifier().predict(crop)
    make_match = match_make(siglip_make["make"], claimed_make)
    detail = {"bucket": "pure_side_profile", "make_read": siglip_make["make"],
              "make_matched": make_match.matched}
    if not make_match.matched:
        return "REJECT", f"[pure_side_profile] make mismatch (read '{siglip_make['make']}')", detail
    return ("MANUAL_REVIEW",
            ("[pure_side_profile] make matches, but individual-vehicle identity "
             "isn't verifiable from a bare side profile alone — human check"), detail)


def check_axle_count(
    image,
    claimed_axle_count: int,
    backend: str = config.AXLE_COUNT_BACKEND,
    model: str | None = None,
    conf_min: float = config.AXLE_COUNT_CONF_MIN,
    axle_source: str | None = None,
    vehicle_mapper: str | None = None,
) -> AxleCountResult:
    """Axle-count in isolation — classify then decide (see module docstring for why
    no dedicated detector is wired). Reused by ``check_side_image_upload``; exposed
    standalone so it's independently testable from identity-binding/duplicate.

    Pass BOTH ``axle_source`` ("auto" | "manual") and (for "manual")
    ``vehicle_mapper`` to also run ``decide_axle_source_consistency`` and fold its
    verdict in (worst-of) — an RC-derived data check independent of the photo.
    Leaving ``axle_source`` unset skips it entirely (opt-in, same pattern as the
    duplicate check elsewhere)."""
    try:
        raw = classify_axle_count(image, backend=backend, model=model)
    except Exception as e:
        return AxleCountResult(
            decision="MANUAL_REVIEW", checked=False, claimed_axle_count=claimed_axle_count,
            axle_source=axle_source, vehicle_mapper=vehicle_mapper,
            reason=f"axle check unavailable ({e})", error=str(e),
        )
    if not raw.get("checked"):
        return AxleCountResult(
            decision="MANUAL_REVIEW", checked=False, claimed_axle_count=claimed_axle_count,
            axle_source=axle_source, vehicle_mapper=vehicle_mapper,
            reason=f"axle check unavailable ({raw.get('error', '?')})", error=raw.get("error"),
        )
    decided = decide_axle_count(raw, claimed_axle_count, conf_min)

    if axle_source is None:
        return AxleCountResult(
            decision=decided["decision"], status=decided["status"], checked=True,
            claimed_axle_count=claimed_axle_count, axle_count=raw.get("axle_count"),
            axle_confidence=raw.get("axle_confidence"), lift_axle_suspected=raw.get("lift_axle_suspected"),
            reason=decided["reason"],
        )

    consistency = decide_axle_source_consistency(claimed_axle_count, axle_source, vehicle_mapper)
    mapper_expected = VEHICLE_MAPPER_AXLE_COUNT.get(vehicle_mapper) if vehicle_mapper else None
    return AxleCountResult(
        decision=_worst_decision(decided["decision"], consistency["decision"]),
        status=decided["status"], checked=True,
        claimed_axle_count=claimed_axle_count, axle_count=raw.get("axle_count"),
        axle_confidence=raw.get("axle_confidence"), lift_axle_suspected=raw.get("lift_axle_suspected"),
        axle_source=axle_source, vehicle_mapper=vehicle_mapper, mapper_expected_axle_count=mapper_expected,
        reason=f"{decided['reason']}; source-consistency: {consistency['reason']}",
    )


def check_side_identity(
    image,
    claimed_vrn: str,
    claimed_make: str,
    front_reference_image=None,
    similarity_min: float = config.SIDE_IMAGE_SIMILARITY_MIN,
    color_hist_min: float = config.SIDE_IMAGE_COLOR_HIST_MIN,
) -> SideImageIdentityResult:
    """Identity-binding in isolation — routed by ``SideImageTypeClassifier`` into
    vrn_visible / corner_view / pure_side_profile (see module docstring). Reused by
    ``check_side_image_upload``; exposed standalone so it's independently testable
    from axle-count/duplicate.

    ``front_reference_image`` is this truck's own already-accepted front photo,
    used by the corner-view bucket's embedding-similarity AND colour-histogram
    checks — its only identity signals; without it, that bucket is MANUAL_REVIEW
    ("unverifiable") rather than falling back to the weaker make classifier (which
    is reserved for the pure-side-profile bucket only — see module docstring)."""
    try:
        arr = load_rgb_array(image)
        bucket = get_side_image_type_classifier().predict(arr)["bucket"]
        if bucket == "vrn_visible":
            decision, reason, detail = _identity_via_vrn(image, claimed_vrn)
        elif bucket == "corner_view":
            decision, reason, detail = _identity_via_corner_view(
                image, front_reference_image, similarity_min, color_hist_min)
        else:
            decision, reason, detail = _identity_via_pure_side_profile(image, claimed_make)
    except Exception as e:
        return SideImageIdentityResult(
            decision="MANUAL_REVIEW", checked=False, claimed_vrn=claimed_vrn, claimed_make=claimed_make,
            reason=f"identity check unavailable ({e})", error=str(e),
        )
    return SideImageIdentityResult(
        decision=decision, checked=True, claimed_vrn=claimed_vrn, claimed_make=claimed_make,
        identity_bucket=detail.get("bucket"), make_read=detail.get("make_read"),
        make_matched=detail.get("make_matched"), front_similarity=detail.get("front_similarity"),
        color_hist_similarity=detail.get("color_hist_similarity"),
        vrn_status=detail.get("vrn_status"), reason=reason,
    )


def check_side_image_upload(
    image,
    claimed_vrn: str,
    claimed_make: str,
    claimed_axle_count: int,
    upload_id: str | None = None,
    front_reference_image=None,
    axle_conf_min: float = config.AXLE_COUNT_CONF_MIN,
    side_image_similarity_min: float = config.SIDE_IMAGE_SIMILARITY_MIN,
    side_image_color_hist_min: float = config.SIDE_IMAGE_COLOR_HIST_MIN,
    axle_backend: str = config.AXLE_COUNT_BACKEND,
    axle_model: str | None = None,
    axle_source: str | None = None,
    vehicle_mapper: str | None = None,
) -> SideImageCheckResult:
    """The single entry point for a side/axle-image upload. Runs duplicate check
    (if ``upload_id`` given), axle count (``check_axle_count``), and identity-
    binding (``check_side_identity``), then takes the worst decision across
    whichever checks ran — see module docstring for the full breakdown.

    ``axle_backend`` — "claude" (default) | "gemini" — selects which model reads
    the axle count; the identity/duplicate checks are unaffected by this.

    Pass ``axle_source`` ("auto" | "manual") + (for "manual") ``vehicle_mapper`` to
    also run the RC-derived axle-count consistency check — see
    ``check_axle_count``'s docstring.
    """
    try:
        dup = check_duplicate(image, upload_id, claimed_vrn, image_type="side") if upload_id else None
        axle = check_axle_count(image, claimed_axle_count, backend=axle_backend,
                                model=axle_model, conf_min=axle_conf_min,
                                axle_source=axle_source, vehicle_mapper=vehicle_mapper)
        identity = check_side_identity(image, claimed_vrn, claimed_make, front_reference_image,
                                       side_image_similarity_min, side_image_color_hist_min)
    except Exception as e:
        return SideImageCheckResult(
            decision="MANUAL_REVIEW", checked=False,
            claimed_vrn=claimed_vrn, claimed_make=claimed_make, claimed_axle_count=claimed_axle_count,
            reason=f"side-image check unavailable ({e})", error=str(e),
        )

    decisions = [axle.decision, identity.decision]
    if dup is not None:
        decisions.append(dup.decision)
    overall = _worst_decision(*decisions)

    reason_parts = [f"axle: {axle.reason}", f"identity: {identity.reason}"]
    if dup is not None:
        reason_parts.append(f"duplicate: {dup.reason}")

    return SideImageCheckResult(
        decision=overall,
        checked=True,
        reason="; ".join(reason_parts),
        claimed_vrn=claimed_vrn,
        claimed_make=claimed_make,
        claimed_axle_count=claimed_axle_count,
        axle_count=axle.axle_count,
        axle_status=axle.status,
        axle_source=axle.axle_source,
        mapper_expected_axle_count=axle.mapper_expected_axle_count,
        identity_bucket=identity.identity_bucket,
        identity_decision=identity.decision,
        duplicate_is_suspect=dup.is_duplicate_suspect if dup is not None else None,
        duplicate_matches=dup.duplicate_matches if dup is not None else [],
    )
