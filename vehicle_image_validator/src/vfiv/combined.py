"""Combined validator — the single entry point per upload, orchestrating Q1 (front-
image gate) + Q2 (VRN + plate colour) + Q3 (make + model).

This is what an external platform registers per image type: one call in, one
consolidated result out, covering every real-model check built so far. "The prompt"
for an image type, from that platform's point of view, is documentation describing
what this entry point checks and which real model performs each piece (see
VALIDATION_SPEC.md) — not literal text sent to a single model. A prompt can only
instruct one model call; it can't itself invoke YOLO, SigLIP, AWS Rekognition, or an
HSV pixel classifier, so the multi-model pipeline has to live in code, not in a prompt.

Q1 gates Q2+Q3 entirely (skip both, saving the AWS/SigLIP calls, if Q1 doesn't PASS —
neither check is meaningful on an image that isn't even a genuine front-of-truck photo).
Once Q1 PASSes, Q2 and Q3 run independently of each other (neither gates the other) so
a single call always returns the complete audit trail for a manual-review UI, rather
than stopping early at the first mismatch. The overall decision takes the worst of the
two: REJECT > MANUAL_REVIEW > PASS.
"""
from vfiv import config
from vfiv.schemas import CombinedResult
from vfiv.front_image.front_image import validate_front_image
from vfiv.front_image.make_model_check import validate_make_model
from vfiv.front_image.vrn_check import validate_vrn

_SEVERITY = {"REJECT": 2, "MANUAL_REVIEW": 1, "PASS": 0}


def validate_upload(
    image,
    claimed_vrn: str,
    claimed_make: str,
    claimed_model: str | None = None,
    conf_min: float = config.FRONT_CONF_MIN,
    ai_reject_conf: float = config.FRONT_AI_REJECT_CONF,
    max_confusable_edits: int = config.VRN_MAX_CONFUSABLE_EDITS,
    model_conf_min: float = config.MODEL_CONF_MIN,
    model_match_min: float = config.MODEL_MATCH_MIN,
) -> CombinedResult:
    """Q1 -> (Q2 and Q3). See the module docstring for the gating/severity rules.

    ``claimed_vrn``/``claimed_make``/``claimed_model`` are whatever the caller sends
    alongside the image when invoking this module — where those values come from (a
    real platform integration vs. a manual test-input field) is outside this module's
    concern. ``claimed_model`` is optional, same as in ``validate_make_model``.
    """
    front = validate_front_image(image, conf_min, ai_reject_conf)
    if front.decision != "PASS":
        return CombinedResult(decision=front.decision, reason=front.reason,
                              checked=front.checked, front=front, error=front.error)

    vrn = validate_vrn(image, claimed_vrn, max_confusable_edits)
    make_model = validate_make_model(image, claimed_make, claimed_model,
                                     model_conf_min, model_match_min)

    decision = max((vrn.decision, make_model.decision), key=_SEVERITY.get)
    reason = f"Q1 PASS; Q2: {vrn.reason}; Q3: {make_model.reason}"
    return CombinedResult(decision=decision, reason=reason, checked=True,
                          front=front, vrn=vrn, make_model=make_model)
