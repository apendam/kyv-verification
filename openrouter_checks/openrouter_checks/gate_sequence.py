"""Orchestrates one upload through the exact decision tree in the "KYV Gate
Sequence" flowchart:

  front image (vehicle type + tamper, one call) -- reject outright if the
    detected type isn't a bus/truck at all, or doesn't match what was claimed
    -> VRN check (unreadable / match / mismatch-similar / mismatch-other)
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
from dataclasses import dataclass, field
from pathlib import Path

from . import config, db, duplicate, matching, prompts, schemas
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
    # else: "match" -> fall through to maker check

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
    try:
        dup_result = duplicate.check_duplicate(
            conn, image_path=str(image_path), image_type="front",
            claimed_vrn=claimed_vrn, exclude_upload_id=upload_id, plate_bbox=plate_bbox,
        )
    except Exception as exc:  # noqa: BLE001 - a bad/corrupt image file, most likely
        _log(conn, upload_id, "duplicate_check", "local:duplicate_check", "technical_failure",
             {"error": str(exc)}, technical_failure=True)
        return finish("MANUAL_REVIEW", "duplicate check: technical failure")

    _log(conn, upload_id, "duplicate_check", f"local:{dup_result.signal}",
         "duplicate" if dup_result.is_duplicate else "clean",
         {"signal": dup_result.signal,
          "best_match_upload_id": dup_result.best_match_upload_id,
          "best_match_score": dup_result.best_match_score,
          "reason": dup_result.reason})
    steps.append({"check": "duplicate_check", "outcome": dup_result.reason})

    if dup_result.is_duplicate:
        return finish("MANUAL_REVIEW", "possible photo reuse")

    return finish("APPROVED", "all checks passed")
