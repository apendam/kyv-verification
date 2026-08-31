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

3. Identity-to-claimed-vehicle — routed by ``classify_side_image_type`` (a VLM
   call, see its docstring) into three buckets of DECREASING reliability. This
   used to be a SigLIP zero-shot embedding comparison
   (``backends/siglip.py``'s ``SideImageTypeClassifier``, now removed) but that
   kept misrouting a genuine corner-view upload where the side of the cargo box
   dominates the frame — no amount of rewording its text prompts could fix it,
   since reworking the TEXT side of a zero-shot comparison can't change how the
   IMAGE itself embeds. The VLM version instead reasons explicitly about
   windshield/plate visibility and cites its evidence, the same "reasoning over
   embedding-similarity" fix already applied to axle counting above. NO bucket
   uses the make classifier any more (see below for why it was dropped from
   both places that used to lean on it):
     a. vrn_visible       -> re-run Q2's own VRN detector/matcher on this image
                             (strongest — exact identity, reuses ``vrn_check.py``
                             as-is, no new logic).
     b. corner_view       -> a SigLIP embedding cosine-similarity AND a colour-
                             histogram comparison, both against the claimed
                             truck's OWN on-file front photo — direct 1:1
                             pairwise compares, NOT a vector-DB nearest-neighbor
                             search. Without a ``front_reference_image`` there's
                             simply nothing to compare against, so this bucket
                             is MANUAL_REVIEW ("unverifiable"). Both signals are
                             UNCALIBRATED (see ``config.SIDE_IMAGE_SIMILARITY_MIN``/
                             ``SIDE_IMAGE_COLOR_HIST_MIN``) — a general SigLIP
                             embedding is trained for semantic similarity, not
                             individual-vehicle re-identification, so it may not
                             reliably separate "same truck, different angle"
                             from "different truck, same make/model/colour";
                             colour can shift with lighting/exposure between two
                             photos of the same truck. Validate both against
                             real labeled pairs before trusting them in
                             production.
     c. pure_side_profile  -> colour-histogram ONLY, against the same front
                             reference photo — no embedding-similarity check
                             here (SigLIP's embedding is angle-sensitive, so a
                             side profile vs. a front-on photo would likely
                             score low even for the same truck; colour is
                             roughly angle-invariant and is the one corner_view
                             signal that actually transfers). Without a
                             ``front_reference_image``, also MANUAL_REVIEW
                             ("unverifiable") — there's nothing else to go on
                             for a bare side profile (no plate, no front
                             grille). The individual-vehicle question is NOT
                             solved here either way — this is the genuinely
                             open piece flagged in the design discussion, not
                             an integration task — so even a colour MATCH is
                             capped at MANUAL_REVIEW, never a confident PASS
                             (two different trucks of the same colour would
                             pass this too), and (unlike the make classifier
                             this replaced) a colour MISMATCH is also
                             MANUAL_REVIEW rather than an outright REJECT, given
                             the same uncalibrated-threshold caveat as
                             corner_view's colour check.

   Make classifier removed from BOTH buckets above (formerly used in
   pure_side_profile, briefly also in corner_view): it's a coarse, brand-only
   zero-shot read (``backends/siglip.py``'s ``MakeClassifier`` — compares the
   image against 8 fixed brand-name text prompts, never actually reads painted
   logos/text) that can misfire between visually-similar cab shapes across
   manufacturers — confirmed on a real upload during manual testing (a genuine
   Tata read as "Eicher" twice, on separate uploads of the same photo). Its old
   REJECT-on-mismatch behaviour meant a genuine truck could get auto-rejected
   on what amounts to a coin-flip brand read — replaced everywhere by the
   colour-histogram check, a real deterministic signal instead of a zero-shot
   classifier's guess.

Overall ``decision`` takes the worst of whichever checks actually ran — REJECT >
MANUAL_REVIEW > PASS, same ordering as ``combined.py``.
"""
from __future__ import annotations

import numpy as np

from vfiv import config
from vfiv.backends.image_io import load_rgb_array
from vfiv.backends.siglip import get_siglip_model
from vfiv.backends.vehicle import get_vehicle_detector
from vfiv.schemas import AxleCountResult, SideImageCheckResult, SideImageIdentityResult
from vfiv.base import call_vlm_json
from vfiv.duplicate_check import check_duplicate
from vfiv.front_image.vrn_check import validate_vrn

_SEVERITY = {"REJECT": 2, "MANUAL_REVIEW": 1, "PASS": 0}

AXLE_PROMPT = """You are counting axles on an uploaded side-profile photo of a truck/bus,
for a document-validation platform. Real commercial vehicles here range from 2 to 7
axles — get this right across that whole range, not just the common 2-axle case.

An AXLE is one wheel-bearing line running across the vehicle's width. A WHEEL is a
single tire. These are NOT the same thing:
- Dual/twin wheels: two tires mounted side-by-side on the SAME side of the SAME axle
  (common on load-bearing axles for heavier trucks). This is still ONE axle, not two —
  a rear axle with dual wheels has 4 wheels total (2 per side) but is 1 axle.
- A tandem or tridem bogie: two or three axles mounted close together as a group
  (common at the rear of heavier trucks, to spread axle load). EACH axle in the group
  IS a separate axle, even though they sit close together and can look like one wide
  axle or a single "unit" at a glance. Look for evenly-spaced distinct wheel positions
  within the cluster — each one is its own axle, whether or not it also has dual wheels.
- A lift/tag axle: can be raised off the road when unloaded. If it's visibly raised
  (not touching the ground) in this photo, still include it in the total count, but set
  "lift_axle_suspected" to true.
- A steering axle: the frontmost axle, which turns for steering. Most trucks have one,
  but some heavier multi-axle trucks have twin-steer (two axles up front) — don't assume
  there's only ever one axle at the front just because it's usually true.

CRITICAL — count ONLY what you can visually confirm in THIS specific photo, never what
a truck brand/model "typically" has. If you recognize the make/model (e.g. "this looks
like a Tata LPT 1918"), do NOT use that recognition to infer a typical axle
configuration (e.g. "...which usually runs 6x2") — this specific vehicle's real,
registered configuration may be a lighter or heavier trim than the class norm, and this
platform exists precisely to catch that gap, not assume it away. For every axle you
count, you must be able to point to a specific visual feature that establishes it as a
distinct axle position — a separate leaf-spring/suspension mount, a visible gap between
wheel sets, a distinct axle housing — not a general expectation of "trucks like this
usually have N axles." If your only justification for an axle is what's typical for the
vehicle's class rather than something you can point to in the pixels in front of you,
do not count it — lower your confidence and note the ambiguity in "reason" instead.

Worked examples across the range (for calibration only — don't assume the vehicle in
front of you matches one of these):
- 2 axles: one front (steer) axle + one rear axle with dual wheels. 6 wheels, 2 axles.
- 3 axles: one front axle + a 2-axle rear tandem bogie, each with dual wheels. Up to 10
  wheels, 3 axles.
- 4-7 axles: a longer rear bogie (tridem or more), sometimes combined with twin-steer
  front axles. Count every distinct axle LINE, whether isolated or clustered in a bogie.

If the full wheelbase isn't visible (cropped, obstructed, mid-corner shot), give your
best count from what IS visible rather than refusing, and say so in "reason".

Before answering, mentally walk the wheelbase front-to-rear and describe each axle
position you identify, how many wheels sit at it, AND the specific visual feature that
tells you it's a distinct axle (not "typical for this model") (e.g. "front: 1 axle,
single wheels; rear: a single wheel cluster with dual wheels, one visible leaf-spring
mount -> 2 axles total" — or, if genuinely a bogie, "...two separate leaf-spring mounts
visible with a gap between wheel sets -> 3 axles total"). Put that walk-through in
"reason" FIRST, then give the final axle_count based on it — do not just report however
many wheels you can see, and do not pad the count with an axle you inferred from the
vehicle's class rather than saw.

Reply with STRICT JSON only, in this field order:
{"reason":"<short position-by-position walk-through, then your conclusion>","axle_count":<int>,"confidence":0-100,"lift_axle_suspected":true|false}"""


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


SIDE_IMAGE_TYPE_PROMPT = """You are classifying what a side/axle-image upload actually
shows, for a document-validation platform that routes to a different identity-checking
strategy depending on the answer. Decide which ONE of these three categories the photo
belongs to:

- "vrn_visible": the side of the truck is shown with a LEGIBLE license plate /
  registration number visible somewhere in the frame.
- "corner_view": a three-quarter/angled shot where the WINDSHIELD is visible (even
  partially, at an angle) alongside the side of the truck body. The windshield is the
  key test: a true side profile is shot perpendicular to the truck's length, so the
  forward-facing windshield would be edge-on or invisible; if you can see the glass
  panel of the windshield at all, that's a forward-facing viewing-angle component,
  which makes this a corner/three-quarter shot -- even if most of the frame is taken up
  by the long side of the truck body/cargo box.
- "pure_side_profile": shot perpendicular to the truck's length -- the windshield is
  edge-on or not visible at all, and no legible plate is visible either.

If BOTH a legible plate AND a visible windshield are present, prefer "vrn_visible" --
it's the stronger identity signal. Base your answer ONLY on what's actually visible in
THIS specific photo, not on how this kind of upload is typically framed.

Reply with STRICT JSON only, in this field order:
{"reason":"<short -- cite whether the windshield and/or plate are actually visible>","bucket":"vrn_visible"|"corner_view"|"pure_side_profile"}"""

SIDE_IMAGE_TYPE_BACKENDS = ["claude", "gemini"]
SIDE_IMAGE_BUCKETS = ("vrn_visible", "corner_view", "pure_side_profile")


def classify_side_image_type(
    image,
    backend: str = config.SIDE_IMAGE_TYPE_BACKEND,
    model: str | None = None,
) -> dict:
    """Which of the three identity-binding buckets (see module docstring) this
    photo belongs to — a VLM judgment call anchored on windshield/plate
    visibility, not a zero-shot embedding comparison. Replaced the earlier SigLIP-
    based ``SideImageTypeClassifier``: it kept misrouting a genuine corner-view
    upload where the side of the cargo box dominates the frame, and no amount of
    rewording its text prompts could fix that — rewording the TEXT side of a
    zero-shot comparison can't change how the IMAGE itself embeds, so a
    stubbornly "side-profile-shaped" photo stayed misrouted regardless of wording.
    ``backend`` — "claude" (default) | "gemini"."""
    if backend == "claude":
        r = call_vlm_json(image, SIDE_IMAGE_TYPE_PROMPT, model or config.VLM_MODEL, max_tokens=200)
    elif backend == "gemini":
        from vfiv.backends.gemini import call_gemini_json
        r = call_gemini_json(image, SIDE_IMAGE_TYPE_PROMPT, model=model)
    else:
        raise ValueError(f"unknown side-image-type backend: {backend!r} "
                         f"(expected one of {SIDE_IMAGE_TYPE_BACKENDS})")
    return r


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
        r = call_vlm_json(image, AXLE_PROMPT, model or config.VLM_MODEL, max_tokens=400)
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
    same vocabulary as Q2/Q3's ``VerificationStatus``. Every branch appends the
    model's own ``r["reason"]`` (its position-by-position wheelbase walk-through,
    per ``AXLE_PROMPT``) to the decision reason -- without this, a REJECT/PASS
    only ever showed "axle count N != claimed M", with the model's actual
    explanation for *why* it counted N silently discarded, making a wrong count
    impossible to debug from the result alone."""
    read_reason = r.get("reason", "")
    if r["axle_confidence"] < conf_min:
        return {
            "status": "UNREADABLE", "decision": "MANUAL_REVIEW",
            "reason": (f"axle read confidence {r['axle_confidence']:.0f}% "
                       f"< {conf_min:.0f}% — needs human count ({read_reason})"),
        }
    if r["axle_count"] == claimed_axle_count:
        note = " (lift axle suspected — verify load state)" if r.get("lift_axle_suspected") else ""
        return {
            "status": "MATCH", "decision": "PASS",
            "reason": f"axle count {r['axle_count']} matches claimed {claimed_axle_count}{note} ({read_reason})",
        }
    return {
        "status": "MISMATCH", "decision": "REJECT",
        "reason": f"axle count {r['axle_count']} != claimed {claimed_axle_count} ({read_reason})",
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


def _identity_via_pure_side_profile(
    image, front_reference_image,
    color_hist_min: float = config.SIDE_IMAGE_COLOR_HIST_MIN,
) -> tuple[str, str, dict]:
    """Colour-histogram ONLY now — no make classifier here any more (see module
    docstring for why it was dropped from this bucket too). No embedding-
    similarity check either: SigLIP's general embedding is angle-sensitive, so a
    pure side profile vs. a front-on reference photo would likely score low even
    for the exact same truck — colour is roughly angle-invariant (same paint from
    any angle) and is the one signal from ``_identity_via_corner_view``'s toolkit
    that actually transfers here. Like the old make-based version, NEVER a
    confident PASS — individual-vehicle identity genuinely isn't solved here even
    with a colour match (two different trucks of the same colour would pass this
    too) — but UNLIKE the old make-classifier version, also never an outright
    REJECT: this shares corner_view's colour check's uncalibrated-threshold
    caveat, so a mismatch here is a lead for a human, not an auto-reject."""
    detail = {"bucket": "pure_side_profile", "color_hist_similarity": None}

    if front_reference_image is None:
        return ("MANUAL_REVIEW",
                "[pure_side_profile] no front reference photo supplied — identity "
                "isn't verifiable from a bare side profile alone", detail)

    crop = _truck_crop(image)
    ref_crop = _truck_crop(front_reference_image)
    color_similarity = _color_histogram_similarity(crop, ref_crop)
    detail["color_hist_similarity"] = color_similarity

    if color_similarity < color_hist_min:
        return ("MANUAL_REVIEW",
                (f"[pure_side_profile] colour-histogram similarity {color_similarity:.4f} "
                 f"< {color_hist_min:.4f} — possible mismatch, human check"), detail)
    return ("MANUAL_REVIEW",
            (f"[pure_side_profile] colour-histogram {color_similarity:.4f} matches, but "
             "individual-vehicle identity isn't verifiable from a bare side profile "
             "alone — human check"), detail)


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
    type_backend: str = config.SIDE_IMAGE_TYPE_BACKEND,
    type_model: str | None = None,
) -> SideImageIdentityResult:
    """Identity-binding in isolation — routed by ``classify_side_image_type`` (a
    VLM call, see its docstring for why this isn't a SigLIP zero-shot comparison
    any more) into vrn_visible / corner_view / pure_side_profile (see module
    docstring). Reused by ``check_side_image_upload``; exposed standalone so it's
    independently testable from axle-count/duplicate.

    ``front_reference_image`` is this truck's own already-accepted front photo —
    used by corner_view's embedding-similarity AND colour-histogram checks, and by
    pure_side_profile's colour-histogram-only check (no embedding there — see that
    function's docstring for why). Without it, both buckets are MANUAL_REVIEW
    ("unverifiable"); no bucket falls back to the make classifier any more (see
    module docstring for why it was dropped from both)."""
    try:
        type_raw = classify_side_image_type(image, backend=type_backend, model=type_model)
        if not type_raw.get("checked"):
            return SideImageIdentityResult(
                decision="MANUAL_REVIEW", checked=False, claimed_vrn=claimed_vrn, claimed_make=claimed_make,
                reason=f"side-image type classification unavailable ({type_raw.get('error', '?')})",
                error=type_raw.get("error"),
            )
        bucket = type_raw.get("bucket")
        if bucket not in SIDE_IMAGE_BUCKETS:
            return SideImageIdentityResult(
                decision="MANUAL_REVIEW", checked=False, claimed_vrn=claimed_vrn, claimed_make=claimed_make,
                reason=f"side-image type classification returned an unrecognized bucket ({bucket!r})",
            )
        if bucket == "vrn_visible":
            decision, reason, detail = _identity_via_vrn(image, claimed_vrn)
        elif bucket == "corner_view":
            decision, reason, detail = _identity_via_corner_view(
                image, front_reference_image, similarity_min, color_hist_min)
        else:
            decision, reason, detail = _identity_via_pure_side_profile(
                image, front_reference_image, color_hist_min)
    except Exception as e:
        return SideImageIdentityResult(
            decision="MANUAL_REVIEW", checked=False, claimed_vrn=claimed_vrn, claimed_make=claimed_make,
            reason=f"identity check unavailable ({e})", error=str(e),
        )
    routing_reason = type_raw.get("reason", "")
    full_reason = f"{reason} [bucket routing: {routing_reason}]" if routing_reason else reason
    return SideImageIdentityResult(
        decision=decision, checked=True, claimed_vrn=claimed_vrn, claimed_make=claimed_make,
        identity_bucket=detail.get("bucket"), front_similarity=detail.get("front_similarity"),
        color_hist_similarity=detail.get("color_hist_similarity"),
        vrn_status=detail.get("vrn_status"), reason=full_reason,
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
    type_backend: str = config.SIDE_IMAGE_TYPE_BACKEND,
    type_model: str | None = None,
) -> SideImageCheckResult:
    """The single entry point for a side/axle-image upload. Runs duplicate check
    (if ``upload_id`` given), axle count (``check_axle_count``), and identity-
    binding (``check_side_identity``), then takes the worst decision across
    whichever checks ran — see module docstring for the full breakdown.

    ``axle_backend`` — "claude" (default) | "gemini" — selects which model reads
    the axle count. ``type_backend`` — same choices — selects which model routes
    the identity-binding bucket (``classify_side_image_type``); independent of
    ``axle_backend`` since they're separate VLM calls, though callers commonly
    pass the same value for both (see the webapp).

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
                                       side_image_similarity_min, side_image_color_hist_min,
                                       type_backend=type_backend, type_model=type_model)
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
