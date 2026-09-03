"""Orchestrates one side/axle upload through the agreed decision tree:

  1. Vehicle type + tamper (one vision call) -- wrong type or a flagged image
     rejects outright.
  2. Framing: full side of the vehicle (cabin AND axle) visible? -- reject if
     either end is out of frame.
  3. Axle count vs. claim -- reject on any mismatch.
  4. VRN read (plate OR painted/stencilled text, never guessed) --
     conflicting renderings or an unreadable VRN both go to manual review;
     a readable VRN runs the same confusable-edit-distance match the
     front-image flow uses (match / mismatch_similar / mismatch_other).
  5. Duplicate check (pHash -> SigLIP cascade, image_type="side"), only on
     the path that would otherwise approve -- same as the front-image flow.

No front-image lookup and no visual cross-check: front and side images
arrive together and are reviewed front-first, so an unreadable VRN here
goes straight to manual review rather than through a fallback that would
rarely have anything to check against.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, db, duplicate, matching, prompts, schemas
from .client import OpenRouterClient, OpenRouterError, OpenRouterInsufficientCredits

IMAGE_TYPE = "side"


@dataclass
class SideResult:
    upload_id: str
    decision: str  # "APPROVED" | "MANUAL_REVIEW" | "REJECT"
    reason: str
    steps: list[dict] = field(default_factory=list)


def run_side_sequence(conn: sqlite3.Connection, client: OpenRouterClient, *,
                       image_path: str | Path, claimed_vehicle_type: str, claimed_axle_count: int,
                       claimed_vrn: str, upload_id: str,
                       vision_model: str = config.DEFAULT_VISION_MODEL) -> SideResult:
    steps: list[dict] = []
    claimed_vehicle_type = claimed_vehicle_type.strip().lower()
    claimed_vrn_norm = (claimed_vrn or "").strip().upper()

    def finish(decision: str, reason: str) -> SideResult:
        db.record_result(conn, upload_id=upload_id, decision=decision, reason=reason,
                          claimed_vehicle_type=claimed_vehicle_type, claimed_axle_count=claimed_axle_count,
                          claimed_vrn=claimed_vrn_norm)
        return SideResult(upload_id, decision, reason, steps)

    def call(check_name: str, system_prompt: str, user_text: str, json_schema: dict):
        start = time.perf_counter()
        try:
            r = client.chat_json(model=vision_model, system_prompt=system_prompt, user_text=user_text,
                                  image_paths=[image_path], json_schema=json_schema, schema_name=check_name)
        except OpenRouterInsufficientCredits:
            raise
        except OpenRouterError as exc:
            db.log_check(conn, upload_id=upload_id, image_type=IMAGE_TYPE, check_name=check_name,
                         model=vision_model, verdict="technical_failure", detail={"error": str(exc)},
                         latency_ms=int((time.perf_counter() - start) * 1000), technical_failure=True)
            return None
        db.log_check(conn, upload_id=upload_id, image_type=IMAGE_TYPE, check_name=check_name, model=r.model,
                     verdict="ok", detail=r.data, prompt_tokens=r.prompt_tokens,
                     completion_tokens=r.completion_tokens, cost_usd=r.cost_usd, latency_ms=r.latency_ms)
        steps.append({"check": check_name, **r.data})
        return r

    # -- 1. Vehicle type + tamper ----------------------------------------------
    r = call("type_tamper_check", prompts.TYPE_TAMPER_SYSTEM, prompts.type_tamper_user_text(),
             schemas.TYPE_TAMPER_SCHEMA)
    if r is None:
        return finish("MANUAL_REVIEW", "type/tamper check: technical failure")

    detected_type = r.data.get("detected_vehicle_type", "other")
    if detected_type == "other":
        return finish("REJECT", "not a bus or truck")
    if detected_type != claimed_vehicle_type:
        return finish("REJECT", f"vehicle type mismatch (claimed {claimed_vehicle_type}, "
                                 f"detected {detected_type})")
    if r.data.get("is_altered_or_ai_generated", False):
        return finish("REJECT", "image flagged as altered/AI-generated")

    # -- 2. Framing --------------------------------------------------------------
    r = call("framing_check", prompts.FRAMING_SYSTEM, prompts.framing_user_text(), schemas.FRAMING_SCHEMA)
    if r is None:
        return finish("MANUAL_REVIEW", "framing check: technical failure")
    if not r.data.get("full_side_visible", False):
        return finish("REJECT", "full side of vehicle not visible")

    # -- 3. Axle count -------------------------------------------------------------
    r = call("axle_count_check", prompts.AXLE_COUNT_SYSTEM, prompts.axle_count_user_text(),
             schemas.AXLE_COUNT_SCHEMA)
    if r is None:
        return finish("MANUAL_REVIEW", "axle count check: technical failure")
    detected_axle_count = r.data.get("detected_axle_count")
    if detected_axle_count != claimed_axle_count:
        return finish("REJECT", f"axle count mismatch (claimed {claimed_axle_count}, "
                                 f"detected {detected_axle_count})")

    # -- 4. VRN read + match --------------------------------------------------------
    r = call("vrn_read", prompts.SIDE_VRN_READ_SYSTEM, prompts.side_vrn_read_user_text(),
             schemas.SIDE_VRN_READ_SCHEMA)
    if r is None:
        return finish("MANUAL_REVIEW", "VRN read: technical failure")

    vrn_bbox = None
    if r.data.get("vrn_visible", False):
        vrn_bbox = (r.data.get("bbox_x_min", 0.0), r.data.get("bbox_y_min", 0.0),
                    r.data.get("bbox_x_max", 0.0), r.data.get("bbox_y_max", 0.0))

    if r.data.get("conflicting_renderings", False):
        return finish("MANUAL_REVIEW", "VRN renderings disagree (plate vs. painted)")

    vrn_text = (r.data.get("vrn_text") or "").strip()
    if not r.data.get("vrn_readable", False) or not vrn_text:
        return finish("MANUAL_REVIEW", "no legible VRN read")

    vrn_verdict = matching.classify_vrn(vrn_text, claimed_vrn_norm)
    db.log_check(conn, upload_id=upload_id, image_type=IMAGE_TYPE, check_name="vrn_check",
                 model="local:classify_vrn", verdict=vrn_verdict.outcome, detail=vrn_verdict.__dict__)
    steps.append({"check": "vrn_check", "outcome": vrn_verdict.outcome, "distance": vrn_verdict.distance})

    if vrn_verdict.outcome == "mismatch_other":
        return finish("REJECT", "VRN mismatch")
    if vrn_verdict.outcome == "mismatch_similar":
        return finish("MANUAL_REVIEW", "similar-char mismatch")
    # else "match" -> fall through to the duplicate check

    # -- 5. Duplicate check (last gate, only on the would-approve path) -------------
    dup_start = time.perf_counter()
    try:
        dup_result = duplicate.check_duplicate(
            conn, image_path=str(image_path), image_type=IMAGE_TYPE, claimed_vrn=claimed_vrn_norm,
            exclude_upload_id=upload_id, plate_bbox=vrn_bbox,
        )
    except Exception as exc:  # noqa: BLE001 - a bad/corrupt image file, most likely
        db.log_check(conn, upload_id=upload_id, image_type=IMAGE_TYPE, check_name="duplicate_check",
                     model="local:duplicate_check", verdict="technical_failure", detail={"error": str(exc)},
                     latency_ms=int((time.perf_counter() - dup_start) * 1000), technical_failure=True)
        return finish("MANUAL_REVIEW", "duplicate check: technical failure")
    dup_latency_ms = int((time.perf_counter() - dup_start) * 1000)

    db.log_check(conn, upload_id=upload_id, image_type=IMAGE_TYPE, check_name="duplicate_check",
                 model=f"local:{dup_result.signal}", verdict="duplicate" if dup_result.is_duplicate else "clean",
                 detail={"signal": dup_result.signal, "best_match_upload_id": dup_result.best_match_upload_id,
                          "best_match_score": dup_result.best_match_score, "reason": dup_result.reason,
                          "matches": [{"upload_id": m.upload_id, "score": m.score, "claimed_vrn": m.claimed_vrn}
                                      for m in dup_result.matches]},
                 latency_ms=dup_latency_ms)
    steps.append({"check": "duplicate_check", "outcome": dup_result.reason})

    if dup_result.is_duplicate:
        return finish("MANUAL_REVIEW", "possible photo reuse")

    return finish("APPROVED", "all checks passed")
