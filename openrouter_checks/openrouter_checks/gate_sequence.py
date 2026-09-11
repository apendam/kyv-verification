"""Orchestrates one upload through the exact decision tree in the "KYV Gate
Sequence" flowchart:

  filename heuristic (no model call, no image processing) -- a hit against a
    known AI-generator export name sends straight to manual review; a known
    real-camera/screenshot/WhatsApp naming convention (or no match either
    way) just proceeds -- see filename_check.py for why this is weak and
    narrow-case only
    -> local AI-detector (no model call) -- a cheap, independent signal that
       gates the vision call below; a hit sends straight to manual review
    -> front image (vehicle type + tamper, one call) -- reject outright if the
       detected type isn't a bus/truck at all, or doesn't match what was
       claimed; the tamper judgment now also covers a visible AI-generator
       watermark and a windshield sticker/FASTag that looks composited in
    -> VRN check (unreadable / match / mismatch-similar / mismatch-other),
       plus a plate-physical-genuineness check on a match
    -> maker check, only after a VRN match (unreadable or match proceed;
       mismatch -> manual review; never a reject on its own)
    -> duplicate check, last, only on the path that would otherwise approve

Every step is logged to the `checks` table (model, tokens, cost, verdict) as it
happens — a technical failure (bad response, exhausted retries, network error)
at ANY step routes to manual review and stops immediately, same as every other
terminal outcome, matching the "Check failed technically?" gates in the diagram.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import ai_detector, config, db, duplicate, filename_check, matching, prompts, schemas
from .client import OpenRouterClient, OpenRouterError, OpenRouterInsufficientCredits


@dataclass
class GateResult:
    upload_id: str
    decision: str  # "APPROVED" | "MANUAL_REVIEW" | "REJECT"
    reason: str
    steps: list[dict] = field(default_factory=list)


def _log(conn, upload_id, check_name, model, verdict, detail, *,
         prompt_tokens=0, completion_tokens=0, cost_usd=0.0, latency_ms=0,
         technical_failure=False):
    db.log_check(
        conn, upload_id=upload_id, image_type="front", check_name=check_name,
        model=model, verdict=verdict, detail=detail,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost_usd=cost_usd, latency_ms=latency_ms, technical_failure=technical_failure,
    )


def run_gate_sequence(conn: sqlite3.Connection, client: OpenRouterClient, *,
                       image_path: str | Path, claimed_vrn: str, claimed_make: str,
                       claimed_vehicle_type: str, upload_id: str,
                       vision_model: str = config.DEFAULT_VISION_MODEL,
                       ) -> GateResult:
    steps: list[dict] = []
    claimed_vehicle_type = claimed_vehicle_type.strip().lower()

    def finish(decision: str, reason: str) -> GateResult:
        db.record_result(conn, upload_id=upload_id, decision=decision, reason=reason,
                          claimed_vrn=claimed_vrn, claimed_make=claimed_make)
        return GateResult(upload_id, decision, reason, steps)

    # -- 0. Filename heuristic (no model call, no image processing) -----------
    # Cheapest possible signal -- checked first, before even the local
    # detector. `image_path`'s basename is expected to be the client-provided
    # original filename; see filename_check.py's own docstring for why this
    # is weak and MANUAL_REVIEW-only.
    upload_filename = Path(image_path).name
    if filename_check.matches_ai_generator_filename(upload_filename):
        _log(conn, upload_id, "filename_check", "local:filename_check", "flagged",
             {"filename": upload_filename})
        steps.append({"check": "filename_check", "filename": upload_filename, "outcome": "flagged"})
        return finish("MANUAL_REVIEW",
                      f"filename matches a known AI-generator naming convention ({upload_filename!r})")

    _log(conn, upload_id, "filename_check", "local:filename_check", "clean",
         {"filename": upload_filename,
          "matches_known_camera_pattern": filename_check.matches_known_camera_filename(upload_filename)})
    steps.append({"check": "filename_check", "filename": upload_filename, "outcome": "clean"})

    # -- 1. Local AI-detector (no model call) -- cheap signal gates the -------
    # costlier vision call, same order as the vehicle-type-before-tamper logic
    # below. MANUAL_REVIEW-only: see ai_detector.py for why this is never a
    # reject gate or trusted alone.
    detector_start = time.perf_counter()
    try:
        artificial_score = ai_detector.get_ai_detector().score_artificial(str(image_path))
    except Exception as exc:  # noqa: BLE001 - a bad/corrupt image file, or a model load failure
        _log(conn, upload_id, "ai_detector_check", "local:ai_detector", "technical_failure",
             {"error": str(exc)}, latency_ms=int((time.perf_counter() - detector_start) * 1000),
             technical_failure=True)
        return finish("MANUAL_REVIEW", "AI detector check: technical failure")
    detector_latency_ms = int((time.perf_counter() - detector_start) * 1000)

    flagged = artificial_score >= config.AI_DETECTOR_ARTIFICIAL_THRESHOLD
    _log(conn, upload_id, "ai_detector_check", "local:ai_detector",
         "flagged" if flagged else "clean", {"artificial_score": artificial_score},
         latency_ms=detector_latency_ms)
    steps.append({"check": "ai_detector_check", "artificial_score": artificial_score,
                  "outcome": "flagged" if flagged else "clean"})
    if flagged:
        return finish("MANUAL_REVIEW", f"local AI-detector flagged image (score={artificial_score:.3f})")

    # -- 1. Front image check (vehicle type + tamper, one call) --------------
    try:
        r = client.chat_json(
            model=vision_model, system_prompt=prompts.FRONT_IMAGE_SYSTEM,
            user_text=prompts.front_image_user_text(), image_paths=[image_path],
            json_schema=schemas.FRONT_IMAGE_SCHEMA, schema_name="front_image_check",
        )
    except OpenRouterInsufficientCredits:
        raise
    except OpenRouterError as exc:
        _log(conn, upload_id, "front_image_check", vision_model, "technical_failure",
             {"error": str(exc)}, technical_failure=True)
        return finish("MANUAL_REVIEW", "front image check: technical failure")

    _log(conn, upload_id, "front_image_check", r.model, "ok",
         {**r.data, "claimed_vehicle_type": claimed_vehicle_type},
         prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
         cost_usd=r.cost_usd, latency_ms=r.latency_ms)
    steps.append({"check": "front_image_check", **r.data, "claimed_vehicle_type": claimed_vehicle_type})

    vehicle_bbox = (r.data.get("vehicle_bbox_x_min", 0.0), r.data.get("vehicle_bbox_y_min", 0.0),
                     r.data.get("vehicle_bbox_x_max", 0.0), r.data.get("vehicle_bbox_y_max", 0.0))

    detected_vehicle_type = r.data.get("detected_vehicle_type", "other")
    if detected_vehicle_type == "other":
        return finish("REJECT", "not a bus or truck")
    if detected_vehicle_type != claimed_vehicle_type:
        return finish("REJECT", f"vehicle type mismatch (claimed {claimed_vehicle_type}, "
                                 f"detected {detected_vehicle_type})")
    if r.data.get("is_altered_or_ai_generated", False):
        return finish("MANUAL_REVIEW", "front image flagged")

    # -- 2. VRN check ----------------------------------------------------------
    try:
        r = client.chat_json(
            model=vision_model, system_prompt=prompts.PLATE_READ_SYSTEM,
            user_text=prompts.plate_read_user_text(), image_paths=[image_path],
            json_schema=schemas.PLATE_READ_SCHEMA, schema_name="vrn_check",
        )
    except OpenRouterInsufficientCredits:
        raise
    except OpenRouterError as exc:
        _log(conn, upload_id, "vrn_check", vision_model, "technical_failure",
             {"error": str(exc)}, technical_failure=True)
        return finish("MANUAL_REVIEW", "VRN check: technical failure")

    plate_text = (r.data.get("plate_text") or "").strip()
    if not r.data.get("plate_readable", False) or not plate_text:
        _log(conn, upload_id, "vrn_check", r.model, "unreadable", r.data,
             prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
             cost_usd=r.cost_usd, latency_ms=r.latency_ms)
        steps.append({"check": "vrn_check", **r.data, "outcome": "unreadable"})
        return finish("MANUAL_REVIEW", "no legible VRN read")

    vrn_verdict = matching.classify_vrn(plate_text, claimed_vrn)
    _log(conn, upload_id, "vrn_check", r.model, vrn_verdict.outcome,
         {**r.data, "match_detail": vrn_verdict.__dict__},
         prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
         cost_usd=r.cost_usd, latency_ms=r.latency_ms)
    steps.append({"check": "vrn_check", **r.data, "outcome": vrn_verdict.outcome,
                  "distance": vrn_verdict.distance})

    # Reused by the duplicate check below (last gate) to black out the plate
    # before embedding -- same call that already reads the plate text, so no
    # extra vision call needed for this side of the gate sequence.
    plate_bbox = None
    if r.data.get("plate_visible", False):
        plate_bbox = (r.data.get("bbox_x_min", 0.0), r.data.get("bbox_y_min", 0.0),
                      r.data.get("bbox_x_max", 0.0), r.data.get("bbox_y_max", 0.0))

    if vrn_verdict.outcome == "mismatch_other":
        return finish("REJECT", "VRN mismatch")
    if vrn_verdict.outcome == "mismatch_similar":
        return finish("MANUAL_REVIEW", "similar-char mismatch")
    # else: "match" -> fall through

    if not r.data.get("plate_looks_physically_genuine", True):
        return finish("MANUAL_REVIEW", "plate does not look physically genuine")

    # -> maker check

    # -- 3. Maker check (only reached after a VRN match) ----------------------
    try:
        r = client.chat_json(
            model=vision_model, system_prompt=prompts.MAKER_READ_SYSTEM,
            user_text=prompts.maker_read_user_text(), image_paths=[image_path],
            json_schema=schemas.MAKER_READ_SCHEMA, schema_name="maker_check",
        )
    except OpenRouterInsufficientCredits:
        raise
    except OpenRouterError as exc:
        _log(conn, upload_id, "maker_check", vision_model, "technical_failure",
             {"error": str(exc)}, technical_failure=True)
        return finish("MANUAL_REVIEW", "maker check: technical failure")

    maker_text = (r.data.get("maker_text") or "").strip()
    if not r.data.get("maker_readable", False) or not maker_text:
        # Per the flowchart: an unreadable maker read proceeds toward approval —
        # it never blocks a confirmed VRN match.
        _log(conn, upload_id, "maker_check", r.model, "unreadable", r.data,
             prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
             cost_usd=r.cost_usd, latency_ms=r.latency_ms)
        steps.append({"check": "maker_check", **r.data, "outcome": "unreadable"})
    else:
        maker_verdict = matching.classify_maker(maker_text, claimed_make)
        _log(conn, upload_id, "maker_check", r.model, maker_verdict.outcome,
             {**r.data, "match_detail": maker_verdict.__dict__},
             prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
             cost_usd=r.cost_usd, latency_ms=r.latency_ms)
        steps.append({"check": "maker_check", **r.data, "outcome": maker_verdict.outcome})
        if maker_verdict.outcome == "mismatch":
            return finish("MANUAL_REVIEW", "maker mismatch")
        # else "match" -> fall through

    # -- 4. Duplicate check (last gate, only on the would-approve path) -------
    # Local pHash + (fallback) SigLIP comparison -- no model call, no cost,
    # so there's no OpenRouterError path here the way every other step has one.
    # Still real wall-clock time though (loading/running SigLIP especially),
    # so it's timed and logged like every other step's latency.
    duplicate_start = time.perf_counter()
    try:
        dup_result = duplicate.check_duplicate(
            conn, image_path=str(image_path), image_type="front",
            claimed_vrn=claimed_vrn, exclude_upload_id=upload_id, plate_bbox=plate_bbox,
            vehicle_bbox=vehicle_bbox,
        )
    except Exception as exc:  # noqa: BLE001 - a bad/corrupt image file, most likely
        _log(conn, upload_id, "duplicate_check", "local:duplicate_check", "technical_failure",
             {"error": str(exc)}, latency_ms=int((time.perf_counter() - duplicate_start) * 1000),
             technical_failure=True)
        return finish("MANUAL_REVIEW", "duplicate check: technical failure")
    duplicate_latency_ms = int((time.perf_counter() - duplicate_start) * 1000)

    _log(conn, upload_id, "duplicate_check", f"local:{dup_result.signal}",
         "duplicate" if dup_result.is_duplicate else "clean",
         {"signal": dup_result.signal,
          "best_match_upload_id": dup_result.best_match_upload_id,
          "best_match_score": dup_result.best_match_score,
          "reason": dup_result.reason,
          "matches": [{"upload_id": m.upload_id, "score": m.score, "claimed_vrn": m.claimed_vrn}
                      for m in dup_result.matches]},
         latency_ms=duplicate_latency_ms)
    steps.append({"check": "duplicate_check", "outcome": dup_result.reason})

    if dup_result.is_duplicate:
        return finish("MANUAL_REVIEW", "possible photo reuse")

    return finish("APPROVED", "all checks passed")
