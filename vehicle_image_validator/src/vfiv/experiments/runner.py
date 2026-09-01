"""Runs Q1 / Q2 / Q3 — individually (``run_q1_only````run_q2_only````run_q3_only``,
for testing one stage in isolation) or end-to-end (``run_test_case``, Q1 -> Q2 -> Q3,
for the test/inference interface. Mirrors ``combined.py``'s gating (Q1 must PASS before
Q2/Q3 run; Q2 and Q3 run independently of each other; overall decision is the worst of
the two: REJECT > MANUAL_REVIEW > PASS) but exposes every intermediate backend's raw
output for comparison, which production's ``combined.py`` deliberately doesn't need to.
"""
from truck_extract_match.core import VerificationStatus, decide_status

from vfiv import config
from vfiv.experiments import q1_select, q2_select, q3_select
from vfiv.experiments.schemas import ExperimentResult
from vfiv.schemas import FrontImageResult, VrnCheckResult
from vfiv.duplicate_check import check_duplicate
from vfiv.front_image.front_image import decide_front_image
from vfiv.front_image.make_model_check import _match_model
from vfiv.front_image.vrn_check import decide_vrn

_SEVERITY = {"REJECT": 2, "MANUAL_REVIEW": 1, "PASS": 0}


def _decide_q1(raw1: dict, conf_min: float, ai_reject_conf: float,
               claimed_vehicle_type: str | None = None) -> FrontImageResult:
    if not raw1.get("checked"):
        return FrontImageResult(decision="MANUAL_REVIEW",
                                reason=f"Q1 unavailable ({raw1.get('error', '?')})",
                                checked=False, error=raw1.get("error"))
    return decide_front_image(raw1, conf_min, ai_reject_conf, claimed_vehicle_type)


def _decide_q2(raw2: dict, claimed_vrn: str, max_confusable_edits: int) -> VrnCheckResult:
    if not raw2.get("checked"):
        return VrnCheckResult(decision="MANUAL_REVIEW", status=VerificationStatus.UNREADABLE.value,
                              claimed_vrn=claimed_vrn,
                              reason=f"Q2 unavailable ({raw2.get('error', '?')})",
                              checked=False, error=raw2.get("error"))
    return decide_vrn(raw2, claimed_vrn, max_confusable_edits)


def run_q1_only(
    image,
    q1_backend: str = "real_cv",
    q1_gemini_model: str | None = None,
    conf_min: float = config.FRONT_CONF_MIN,
    ai_reject_conf: float = config.FRONT_AI_REJECT_CONF,
    claimed_vrn: str | None = None,
    upload_id: str | None = None,
    claimed_vehicle_type: str | None = None,
) -> FrontImageResult:
    """Q1 in isolation — no VRN/make needed for the gate itself, useful for
    iterating on the front-gate backend alone. Pass BOTH ``claimed_vrn`` and
    ``upload_id`` to also run the cross-upload duplicate check
    (``duplicate_check.py``, ``image_type="front"``) and fold its
    verdict into the returned decision — same opt-in-via-``upload_id`` pattern as
    ``side_image/side_image_check.py``'s ``check_side_image_upload``. Pass
    ``claimed_vehicle_type`` ("truck" | "bus") to also check the detected
    category against it — see ``front_image.decide_front_image``."""
    raw1 = q1_select.classify_q1(image, q1_backend, gemini_model=q1_gemini_model)
    result = _decide_q1(raw1, conf_min, ai_reject_conf, claimed_vehicle_type)

    if not (claimed_vrn and upload_id):
        return result

    dup = check_duplicate(image, upload_id, claimed_vrn, image_type="front")
    return result.model_copy(update={
        "decision": max([result.decision, dup.decision], key=_SEVERITY.get),
        "reason": f"{result.reason}; duplicate: {dup.reason}",
        "duplicate_is_suspect": dup.is_duplicate_suspect,
        "duplicate_matches": dup.duplicate_matches,
    })


def run_q2_only(
    image,
    claimed_vrn: str,
    q2_backend: str = "rekognition",
    q2_gemini_model: str | None = None,
    max_confusable_edits: int = config.VRN_MAX_CONFUSABLE_EDITS,
) -> VrnCheckResult:
    """Q2 in isolation — no dependency on Q1 passing first, useful for iterating on
    the VRN/colour backend alone against images you already know are front shots."""
    raw2 = q2_select.classify_q2(image, q2_backend, gemini_model=q2_gemini_model)
    return _decide_q2(raw2, claimed_vrn, max_confusable_edits)


def run_q3_only(
    image,
    claimed_make: str,
    claimed_model: str | None = None,
    q3_make_backend: str = "siglip_rekognition",
    q3_model_backend: str = "claude",
    q3_make_gemini_model: str | None = None,
    q3_model_gemini_model: str | None = None,
    model_conf_min: float = config.MODEL_CONF_MIN,
    model_match_min: float = config.MODEL_MATCH_MIN,
) -> dict:
    """Q3 in isolation — no dependency on Q1/Q2, useful for iterating on the make/model
    backend alone. Returns a plain dict (no single production schema covers Q3
    standalone) — the same fields ``run_test_case`` folds into ``ExperimentResult``."""
    votes, vote_errors = q3_select.collect_make_votes(image, q3_make_backend,
                                                      gemini_model=q3_make_gemini_model)
    make_decision = q3_select.decide_make_multi(votes, claimed_make, errors=vote_errors)
    make_status = make_decision["make_status"]

    model_raw = q3_select.classify_model(image, q3_model_backend, gemini_model=q3_model_gemini_model)
    model_checked = (bool(claimed_model) and model_raw.get("checked")
                     and model_raw.get("model_confidence", 0) >= model_conf_min)
    model_status = None
    if model_checked:
        matched, _score = _match_model(model_raw.get("model", ""), claimed_model, model_match_min)
        model_status = decide_status(matched, model_raw.get("model") or None).value

    if make_status == "MISMATCH" or (model_checked and model_status == "MISMATCH"):
        decision = "REJECT"
    elif make_status == "UNREADABLE" or (model_checked and model_status == "UNREADABLE"):
        decision = "MANUAL_REVIEW"
    else:
        decision = "PASS"

    reason = (f"make: {make_status} via {make_decision['matched_via']}; model: "
              + (f"{model_status}" if model_checked else "not checked"))
    if make_decision["errors"]:
        reason += "; make backend error(s): " + "; ".join(
            f"{e['source']}: {e['error']}" for e in make_decision["errors"])
    if model_checked is False and not model_raw.get("checked") and claimed_model:
        reason += f"; model backend error: {model_raw.get('error', 'unavailable')}"

    return {
        "decision": decision, "reason": reason,
        "claimed_make": claimed_make, "make_status": make_status,
        "make_match_via": make_decision["matched_via"], "make_votes": make_decision["votes"],
        "make_errors": make_decision["errors"],
        "claimed_model": claimed_model, "model_checked": model_checked,
        "model_status": model_status, "extracted_model": model_raw.get("model") or None,
        "model_confidence": model_raw.get("model_confidence"),
    }


def run_test_case(
    image,
    claimed_vrn: str,
    claimed_make: str,
    claimed_model: str | None = None,
    q1_backend: str = "real_cv",
    q2_backend: str = "rekognition",
    q3_make_backend: str = "siglip_rekognition",
    q3_model_backend: str = "claude",
    q1_gemini_model: str | None = None,
    q2_gemini_model: str | None = None,
    q3_make_gemini_model: str | None = None,
    q3_model_gemini_model: str | None = None,
    conf_min: float = config.FRONT_CONF_MIN,
    ai_reject_conf: float = config.FRONT_AI_REJECT_CONF,
    max_confusable_edits: int = config.VRN_MAX_CONFUSABLE_EDITS,
    model_conf_min: float = config.MODEL_CONF_MIN,
    model_match_min: float = config.MODEL_MATCH_MIN,
    upload_id: str | None = None,
    claimed_vehicle_type: str | None = None,
) -> ExperimentResult:
    """End-to-end: Q1 -> (Q2 and Q3). Q1 gates Q2+Q3 entirely (skipped, saving the
    Rekognition/SigLIP/Gemini calls, if Q1 doesn't PASS); once Q1 passes, Q2 and Q3 run
    independently of each other. The ``*_gemini_model`` params override the configured
    default (env-var driven, see config.py) for that stage's call — ignored unless
    that stage's backend is ``"gemini"``.

    Pass ``upload_id`` to also run the duplicate check (``image_type="front"``) as
    part of Q1 — ``claimed_vrn`` is already mandatory here (Q2 needs it regardless),
    so unlike the standalone Q1 tab this only needs the one extra id to turn on.
    Pass ``claimed_vehicle_type`` ("truck" | "bus") to also check the detected
    category as part of Q1 — see ``front_image.decide_front_image``.
    """
    front = run_q1_only(image, q1_backend, q1_gemini_model, conf_min, ai_reject_conf,
                        claimed_vrn=claimed_vrn, upload_id=upload_id,
                        claimed_vehicle_type=claimed_vehicle_type)
    if front.decision != "PASS":
        return ExperimentResult(q1_backend=q1_backend, q1=front,
                                overall_decision=front.decision, overall_reason=front.reason)

    vrn = run_q2_only(image, claimed_vrn, q2_backend, q2_gemini_model, max_confusable_edits)
    q3 = run_q3_only(image, claimed_make, claimed_model, q3_make_backend, q3_model_backend,
                     q3_make_gemini_model, q3_model_gemini_model, model_conf_min, model_match_min)

    overall = max((vrn.decision, q3["decision"]), key=_SEVERITY.get)
    reason = (f"Q1({q1_backend}) PASS; Q2({q2_backend}): {vrn.reason}; "
              f"Q3 make({q3_make_backend}): {q3['make_status']} via {q3['make_match_via']}; "
              f"Q3 model({q3_model_backend}): "
              + (f"{q3['model_status']}" if q3["model_checked"] else "not checked"))

    return ExperimentResult(
        q1_backend=q1_backend, q1=front,
        q2_backend=q2_backend, q2=vrn,
        q3_make_backend=q3_make_backend, q3_model_backend=q3_model_backend,
        q3_make_status=q3["make_status"], q3_make_votes=q3["make_votes"],
        q3_model_checked=q3["model_checked"], q3_model_status=q3["model_status"],
        q3_model_extracted=q3["extracted_model"],
        overall_decision=overall, overall_reason=reason,
    )
