"""Front-image validator — is the upload a genuine FRONT photo of a truck/bus?

Originally a single Claude VLM call (ported from
truck_verification_pipeline/pose_check.py). Now split across two real sources:
  - vehicle_type / view / is_front / front_complete / confidence: a real CV pipeline
    (YOLOv8 vehicle detector + SigLIP 2 zero-shot pose + a completeness heuristic —
    ``backends/gate.py``, copied from truck_front_extractor's gate/backends). The CV
    gate is authoritative here, same as that project's own architecture — a generic
    COCO-trained YOLO model is documented to be weak on Indian trucks (confirmed in
    testing here too), so swap in a truck-finetuned detector in production.
  - is_screenshot / is_photo_of_photo / ai_generated: still Claude — no CV model in
    the sibling repos does this kind of judgment call, so the prompt is narrowed to
    just these fields instead of re-deriving vehicle/view (which the CV gate already
    determined, more cheaply and without an API round-trip).
Decisioning (``decide_front_image``) is unchanged — it only cares about the merged
dict's fields, not which backend produced them.
"""
from vfiv import config
from vfiv.backends.gate import run_gate
from vfiv.backends.image_io import load_rgb_array
from vfiv.backends.vehicle import decide_vehicle_type_match
from vfiv.schemas import FrontImageResult
from vfiv.base import call_vlm_json

PROMPT = """You are inspecting an uploaded image for a document/image validation platform.
The upload is supposed to be a photograph of the FRONT of a truck/bus, of the REAL
physical vehicle. It may come straight from a camera OR from the phone's photo gallery/
camera roll (either is fine — do not penalise it for coming from a gallery). What matters
is that it is a direct, original photograph of the real vehicle — not a screenshot and not
a re-photograph of an existing picture.

The vehicle type and viewpoint are judged separately (by a real detection model) — focus
ONLY on whether this is a genuine, direct photograph and not a digital/physical re-capture.
Reply with STRICT JSON only:
{"is_screenshot":true|false,"is_photo_of_photo":true|false,"ai_generated":true|false,"ai_confidence":0-100,"reason":"<short>"}

Definitions:
- is_screenshot: true if this is a screenshot, a photo OF a digital screen/monitor, or an
  image embedded in a web/app interface (browser chrome, toolbars, buttons, window
  borders, cursor, status bars, watermarks/overlaid UI text).
- is_photo_of_photo: true if this is a photograph of an existing PRINTED photo, poster, or
  picture of the truck — i.e. someone re-photographed a physical print rather than
  photographing the real vehicle. Look for: visible photo-paper texture/gloss/sheen, a
  physical print's border, edge, curl, or corner, glare/reflection off glossy paper, a
  hand/fingers holding the printout, or the truck's image sitting inside a visibly flat
  rectangular photograph within the frame. (Distinct from is_screenshot, which is about
  digital screens/UI — this is about physical prints/photos.)
- ai_generated: true if the image looks AI-GENERATED / synthetic / CGI / heavily AI-edited.
  Look for: implausible/garbled text on the plate or signage, melted or asymmetric
  badges/logos, impossible reflections or lighting, warped geometry, over-smooth or
  "too perfect" surfaces, fused/extra parts. Be conservative — real photos can look odd.
- ai_confidence: 0-100, how sure you are it is AI-generated (0 if it looks like a real photo).
- reason: one short phrase citing the deciding factor."""


def classify_front_image(image, model: str = config.VLM_MODEL) -> dict:
    """image: file path, PIL.Image, or ndarray. Merges the real CV gate (vehicle/view/
    front/complete/confidence) with Claude's judgment calls (screenshot/photo-of-photo/
    AI-generated) into the single dict shape ``decide_front_image`` expects.

    ``checked`` is False if EITHER source fails — a screenshot/AI-generated verdict
    with no real vehicle classification (or vice versa) isn't a usable result.
    """
    try:
        gate = run_gate(load_rgb_array(image))
    except Exception as e:
        return {"checked": False, "error": f"CV gate unavailable: {e}"}

    claude = call_vlm_json(image, PROMPT, model)
    if not claude.get("checked"):
        return claude

    return {
        "checked": True,
        "vehicle_type": gate["vehicle_type"],
        "view": gate["view"],
        "is_front": gate["is_front"],
        "front_complete": gate["front_complete"],
        "confidence": gate["confidence"],
        "is_screenshot": bool(claude.get("is_screenshot")),
        "is_photo_of_photo": bool(claude.get("is_photo_of_photo")),
        "ai_generated": bool(claude.get("ai_generated")),
        "ai_confidence": float(claude.get("ai_confidence", 0)),
        "reason": claude.get("reason", ""),
    }


def decide_front_image(
    r: dict,
    conf_min: float = config.FRONT_CONF_MIN,
    ai_reject_conf: float = config.FRONT_AI_REJECT_CONF,
    claimed_vehicle_type: str | None = None,
) -> FrontImageResult:
    """Pure decision logic over an already-classified dict (``r["checked"]`` must be
    True — see ``classify_front_image````classify_combined``). Split out from
    ``validate_front_image`` so the combined single-call prompt (``combined.py``)
    can reuse this exact decisioning without re-deriving it.

    PASS          genuine photo · truck/bus · clear, complete FRONT view · confident
    REJECT        fails a hard criterion (screenshot, confident AI, wrong vehicle/view, …)
    MANUAL_REVIEW needs a human (e.g. suspected AI-generated below the reject threshold)

    ``claimed_vehicle_type`` ("truck" | "bus", case-insensitive) is optional — both
    truck and bus VRNs get issued against this platform, so beyond the existing
    "is this even a truck/bus" hard gate below, this also checks whether it's the
    SPECIFIC one claimed (``backends.vehicle.decide_vehicle_type_match``). Only
    runs once every other hard gate has already cleared, and only ever downgrades
    an otherwise-PASS result to MANUAL_REVIEW — never a solo REJECT, since the
    detector here is explicitly documented as weak on Indian trucks/buses (see
    ``decide_vehicle_type_match``'s docstring).
    """
    detail = (f"{r['vehicle_type']} / {r['view']} "
              f"(front={r['is_front']}, complete={r.get('front_complete')}, "
              f"screenshot={r.get('is_screenshot')}, photo_of_photo={r.get('is_photo_of_photo')}, "
              f"ai={r.get('ai_generated')}"
              f"@{r.get('ai_confidence', 0):.0f}%, {r['confidence']:.0f}%) — {r['reason']}")

    def result(decision: str, reason: str, **overrides) -> FrontImageResult:
        extra = {k: v for k, v in r.items() if k != "reason"}
        if claimed_vehicle_type:
            extra["claimed_vehicle_type"] = claimed_vehicle_type.strip().lower()
        extra.update(overrides)
        return FrontImageResult(decision=decision, reason=reason, **extra)

    # Ordered checks -> first match wins.
    if r.get("is_screenshot"):
        return result("REJECT", f"screenshot / photo-of-screen / UI interface  [{detail}]")
    if r.get("is_photo_of_photo"):
        return result("REJECT", f"photo of an existing printed photo, not the real vehicle  [{detail}]")
    if r.get("ai_generated"):
        if r.get("ai_confidence", 0) >= ai_reject_conf:
            return result("REJECT", f"AI-generated image ({r.get('ai_confidence', 0):.0f}%)  [{detail}]")
        return result("MANUAL_REVIEW", (f"possibly AI-generated ({r.get('ai_confidence', 0):.0f}% "
                                          f"< {ai_reject_conf:.0f}%) — human check  [{detail}]"))
    if r["vehicle_type"] not in ("truck", "bus"):
        return result("REJECT", f"main vehicle is '{r['vehicle_type']}', not a truck/bus  [{detail}]")
    if not r["is_front"]:
        return result("REJECT", f"not a front view ('{r['view']}')  [{detail}]")
    if not r.get("front_complete"):
        return result("REJECT", f"front of the truck is incomplete / cut off  [{detail}]")
    if r["confidence"] < conf_min:
        return result("REJECT", f"low confidence ({r['confidence']:.0f}% < {conf_min:.0f}%)  [{detail}]")
    if claimed_vehicle_type:
        type_check = decide_vehicle_type_match(r["vehicle_type"], claimed_vehicle_type)
        if type_check["decision"] != "PASS":
            return result("MANUAL_REVIEW", f"{type_check['reason']}  [{detail}]",
                          vehicle_type_status=type_check["status"])
        return result("PASS", detail, vehicle_type_status=type_check["status"])
    return result("PASS", detail)


def validate_front_image(
    image,
    conf_min: float = config.FRONT_CONF_MIN,
    ai_reject_conf: float = config.FRONT_AI_REJECT_CONF,
    claimed_vehicle_type: str | None = None,
) -> FrontImageResult:
    """Classify then decide (single-call path). See ``decide_front_image`` for the
    decision logic and ``classify_front_image`` for the VLM call."""
    r = classify_front_image(image)
    if not r.get("checked"):
        return FrontImageResult(
            decision="MANUAL_REVIEW",
            reason=f"front-image check unavailable ({r.get('error', '?')})",
            checked=False,
            error=r.get("error"),
            claimed_vehicle_type=claimed_vehicle_type.strip().lower() if claimed_vehicle_type else None,
        )
    return decide_front_image(r, conf_min, ai_reject_conf, claimed_vehicle_type)
