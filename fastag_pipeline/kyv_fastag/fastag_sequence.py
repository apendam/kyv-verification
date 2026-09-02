"""Orchestrates one FASTag upload through the agreed decision tree:

  1. Front check (one vision call): fully framed AND not altered/AI-generated
     -> either failing sends to manual review.
  2. Barcode & QR readability (local decode, zxingcpp -- no model call).
     Both unreadable -> manual review. Otherwise branch on which is readable
     (path_taken: "barcode_only" | "qr_only" | "both_readable"):
       - qr_only (barcode unreadable): skip straight to QR parsing.
       - barcode_only / both_readable: BARCODE CHECK first --
           mismatch -> reject (unconfirmed severity, flagged to revisit).
           match, QR unreadable (barcode_only): fall back to the barcode's
             own checksum validity (zxingcpp's `valid` flag) -- valid ->
             approve, invalid -> manual review.
           match, QR readable (both_readable): QR PARSING as the
             cross-check -- Tag ID AND bank code both matching claim ->
             approve, either mismatching -> reject.

Every step is logged to the `checks` table regardless of outcome, same
technical-failure -> manual-review posture as the front-image gate sequence.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import barcode_qr, config, db, prompts, schemas
from .client import OpenRouterClient, OpenRouterError, OpenRouterInsufficientCredits


@dataclass
class FastagResult:
    upload_id: str
    decision: str  # "APPROVED" | "MANUAL_REVIEW" | "REJECT"
    reason: str
    path_taken: str | None  # "both_unreadable" | "barcode_only" | "qr_only" | "both_readable"
    steps: list[dict] = field(default_factory=list)


def _norm(value: str | None) -> str:
    return (value or "").strip().upper()


def run_fastag_sequence(conn: sqlite3.Connection, client: OpenRouterClient, *,
                         image_path: str | Path, claimed_vrn: str, claimed_barcode: str,
                         claimed_tag_id: str, claimed_bank_code: str, upload_id: str,
                         vision_model: str = config.DEFAULT_VISION_MODEL) -> FastagResult:
    """`claimed_vrn` is carried through to the audit trail (results table,
    batch report) only -- it's not a gate in this flow (no VRN check step),
    just the record of which vehicle this FASTag was claimed against.
    """
    steps: list[dict] = []
    claimed_vrn = _norm(claimed_vrn)
    claimed_barcode, claimed_tag_id, claimed_bank_code = (
        _norm(claimed_barcode), _norm(claimed_tag_id), _norm(claimed_bank_code))

    def finish(decision: str, reason: str, path_taken: str | None = None) -> FastagResult:
        db.record_result(conn, upload_id=upload_id, decision=decision, reason=reason,
                          path_taken=path_taken, claimed_vrn=claimed_vrn, claimed_barcode=claimed_barcode,
                          claimed_tag_id=claimed_tag_id, claimed_bank_code=claimed_bank_code)
        return FastagResult(upload_id, decision, reason, path_taken, steps)

    def log_and_step(check_name: str, verdict: str, detail: dict, **log_kwargs) -> None:
        db.log_check(conn, upload_id=upload_id, check_name=check_name, model="local:zxingcpp",
                     verdict=verdict, detail=detail, **log_kwargs)
        steps.append({"check": check_name, **detail, "outcome": verdict})

    # -- 1. Front check (framing + tamper), one vision call --------------------
    front_start = time.perf_counter()
    try:
        r = client.chat_json(
            model=vision_model, system_prompt=prompts.FASTAG_FRONT_SYSTEM,
            user_text=prompts.fastag_front_user_text(), image_paths=[image_path],
            json_schema=schemas.FASTAG_FRONT_SCHEMA, schema_name="fastag_front_check",
        )
    except OpenRouterInsufficientCredits:
        raise
    except OpenRouterError as exc:
        db.log_check(conn, upload_id=upload_id, check_name="fastag_front_check", model=vision_model,
                     verdict="technical_failure", detail={"error": str(exc)},
                     latency_ms=int((time.perf_counter() - front_start) * 1000), technical_failure=True)
        return finish("MANUAL_REVIEW", "front check: technical failure")

    db.log_check(conn, upload_id=upload_id, check_name="fastag_front_check", model=r.model,
                 verdict="ok", detail=r.data, prompt_tokens=r.prompt_tokens,
                 completion_tokens=r.completion_tokens, cost_usd=r.cost_usd, latency_ms=r.latency_ms)
    steps.append({"check": "fastag_front_check", **r.data})

    if not r.data.get("fastag_fully_framed", False):
        return finish("MANUAL_REVIEW", "FASTag not fully framed")
    if r.data.get("is_altered_or_ai_generated", False):
        return finish("MANUAL_REVIEW", "front image flagged")

    # -- 2. Barcode & QR readability (local decode, no model call) -------------
    decode_start = time.perf_counter()
    try:
        barcode_result = barcode_qr.read_barcode(str(image_path))
        qr_result = barcode_qr.read_qr(str(image_path))
    except Exception as exc:  # noqa: BLE001 - a bad/corrupt image file, most likely
        db.log_check(conn, upload_id=upload_id, check_name="barcode_qr_readability",
                     model="local:zxingcpp", verdict="technical_failure", detail={"error": str(exc)},
                     latency_ms=int((time.perf_counter() - decode_start) * 1000), technical_failure=True)
        return finish("MANUAL_REVIEW", "barcode/QR decode: technical failure")
    decode_latency_ms = int((time.perf_counter() - decode_start) * 1000)

    db.log_check(conn, upload_id=upload_id, check_name="barcode_qr_readability", model="local:zxingcpp",
                 verdict="ok", detail={"barcode_readable": barcode_result.readable,
                                        "qr_readable": qr_result.readable}, latency_ms=decode_latency_ms)
    steps.append({"check": "barcode_qr_readability", "barcode_readable": barcode_result.readable,
                  "qr_readable": qr_result.readable})

    if not barcode_result.readable and not qr_result.readable:
        return finish("MANUAL_REVIEW", "barcode and QR both unreadable", path_taken="both_unreadable")

    if barcode_result.readable and qr_result.readable:
        path_taken = "both_readable"
    elif barcode_result.readable:
        path_taken = "barcode_only"
    else:
        path_taken = "qr_only"

    def qr_parsing_and_finish() -> FastagResult:
        tag_id_match = _norm(qr_result.tag_id) == claimed_tag_id
        bank_code_match = _norm(qr_result.bank_code) == claimed_bank_code
        both_match = tag_id_match and bank_code_match
        log_and_step("qr_parsing", "match" if both_match else "mismatch",
                     {"tag_id_value_read": qr_result.tag_id, "claimed_tag_id": claimed_tag_id,
                      "bank_code_value_read": qr_result.bank_code,
                      "claimed_bank_code": claimed_bank_code})
        if both_match:
            return finish("APPROVED", "QR Tag ID and bank code both matched", path_taken)
        return finish("REJECT", "QR Tag ID or bank code mismatch", path_taken)

    # -- qr_only: barcode unreadable, skip straight to QR parsing --------------
    if path_taken == "qr_only":
        return qr_parsing_and_finish()

    # -- barcode_only / both_readable: BARCODE CHECK first ----------------------
    barcode_match = _norm(barcode_result.value) == claimed_barcode
    log_and_step("barcode_check", "match" if barcode_match else "mismatch",
                 {"barcode_value_read": barcode_result.value, "claimed_barcode": claimed_barcode})
    if not barcode_match:
        # Unconfirmed severity -- agreed to revisit once real prompt/decode
        # data is in, per the design discussion. REJECT for now.
        return finish("REJECT", "barcode mismatch", path_taken)

    if path_taken == "barcode_only":
        # No QR to cross-check against -- fall back to the barcode's own
        # checksum validity (zxingcpp's `valid` flag: can the decoder
        # re-derive the same value from the stripe pattern's own checksum?).
        log_and_step("barcode_stripe_validity", "valid" if barcode_result.valid else "invalid",
                     {"checksum_valid": barcode_result.valid})
        if barcode_result.valid:
            return finish("APPROVED", "barcode matched, checksum-valid, QR unreadable", path_taken)
        return finish("MANUAL_REVIEW", "barcode matched but checksum invalid, QR unreadable", path_taken)

    # path_taken == "both_readable": QR parsing as the independent cross-check
    return qr_parsing_and_finish()
