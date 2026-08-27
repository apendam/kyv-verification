"""Test/inference interface — individual and bulk testing, either end-to-end or one
stage (Q1/Q2/Q3) at a time, with a selectable backend per stage, so different CV/LLM
combinations can be compared side by side. Uses Gradio, matching
``truck_verification_pipeline``'s own choice for this kind of dev/test tool in this
ecosystem.

This is a TESTING tool, not the production entry point — the production entry point is
``vfiv.validate_upload`` (``validators/combined.py``), which always uses today's default
backends. Run with:

    python -m vfiv.webapp
"""
from __future__ import annotations

import io
import uuid

import gradio as gr
import pandas as pd
import requests
from PIL import Image

from vfiv import config
from vfiv.backends.fastag_reader import FASTAG_OCR_BACKENDS, parse_qr_payload
from vfiv.backends.vector_store import DuplicateStoreError, count_by_type
from vfiv.experiments import q1_select, q2_select, q3_select
from vfiv.experiments.runner import run_q1_only, run_q2_only, run_q3_only, run_test_case
from vfiv.validators.duplicate_check import check_duplicate
from vfiv.validators.fastag_image.fastag_check import check_fastag_upload, classify_fastag_upload
from vfiv.validators.side_image.side_image_check import (
    AXLE_COUNT_BACKENDS,
    check_axle_count,
    check_side_identity,
    check_side_image_upload,
)

Q1_CHOICES = q1_select.Q1_BACKENDS
Q2_CHOICES = q2_select.Q2_BACKENDS
Q3_MAKE_CHOICES = q3_select.Q3_MAKE_BACKENDS
Q3_MODEL_CHOICES = q3_select.Q3_MODEL_BACKENDS

# Common Gemini 2.5 model ids — the dropdown also accepts a typed custom value (e.g. a
# future model id), so this list is a convenience, not an exhaustive whitelist.
GEMINI_MODEL_CHOICES = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

# Plain, document-style theme — modeled on how a Blackbuck PRD actually reads (see
# blackbuckPrdBuilder's references/style/tone-and-conventions.md): a plain white page,
# black text, restrained bold/colour used only "for the exact state a system ends up
# in" (their words, re: bolding a decision like **Tag Hotlisted**), tables for
# structure rather than decoration, and no brand chrome — this is a working document,
# not a marketing surface. System font stack, not a loaded display face, for the same
# reason: keep it plain.
_DOC_GREEN = "#1a7f37"
_DOC_RED = "#b42318"
_DOC_AMBER = "#9a6700"

_DECISION_STYLE = {
    "PASS": _DOC_GREEN,
    "REJECT": _DOC_RED,
    "MANUAL_REVIEW": _DOC_AMBER,
}

CUSTOM_CSS = """
:root {
    --doc-bg: #ffffff;
    --doc-surface: #f6f6f6;
    --doc-text: #1a1a1a;
    --doc-muted: #5f5f5f;
    --doc-border: #d0d0d0;
}

.gradio-container {
    background: var(--doc-bg) !important;
    color: var(--doc-text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    font-family: inherit !important;
    font-weight: 700 !important;
    letter-spacing: normal !important;
    color: var(--doc-text) !important;
}

.gradio-container .prose, .gradio-container p, .gradio-container span, .gradio-container label {
    color: var(--doc-text) !important;
}

.gradio-container .prose strong {
    color: var(--doc-text) !important;
}
.gradio-container .prose code {
    color: var(--doc-text) !important;
    background: var(--doc-surface) !important;
    border: 1px solid var(--doc-border) !important;
    border-radius: 3px !important;
}

.gradio-container .block, .gradio-container .form {
    background: var(--doc-surface) !important;
    border: 1px solid var(--doc-border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

/* The small label caption Gradio renders above a component (e.g. "Upload image",
   "Full result") ships with its own light background, under either a ".float" class
   (image/file-type inputs) or a "block-label" test-id (JSON/dataframe/etc. outputs) --
   spelled out explicitly rather than left to inheritance, same reasoning as before. */
.gradio-container label.float, .gradio-container .float,
.gradio-container [data-testid="block-label"] {
    background: var(--doc-surface) !important;
    color: var(--doc-muted) !important;
}

/* Tab bars -- targeted by the ARIA role (stable across Gradio versions) rather than
   Gradio's internal class names, which changed between major versions. */
.gradio-container button[role="tab"] {
    color: var(--doc-muted) !important;
    font-weight: 500 !important;
}
.gradio-container button[role="tab"].selected {
    color: var(--doc-text) !important;
    border-bottom: 2px solid var(--doc-text) !important;
}
/* The selected-tab underline is actually drawn by an ::after bar (Tailwind blue-500
   hardcoded, not theme-driven), layered on top of the border-bottom above -- override
   it directly rather than relying on the border alone. */
.gradio-container button[role="tab"]::after {
    background: var(--doc-text) !important;
}

.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: var(--doc-bg) !important;
    color: var(--doc-text) !important;
    border: 1px solid var(--doc-border) !important;
    border-radius: 4px !important;
}

/* Dropdown popover -- Gradio builds this as a plain <ul class="options"><li
   class="item"> list, not a native <select>, so the input/textarea/select rule
   above never reaches it; it renders with the browser's default light background
   otherwise (harmless now that the rest of the page is also light, but kept
   explicit so hover/selected states stay legible). */
.gradio-container ul.options {
    background: var(--doc-bg) !important;
    border: 1px solid var(--doc-border) !important;
}
.gradio-container ul.options li.item {
    color: var(--doc-text) !important;
    background: var(--doc-bg) !important;
}
.gradio-container ul.options li.item.selected,
.gradio-container ul.options li.item.active {
    background: var(--doc-surface) !important;
    color: var(--doc-text) !important;
}
.gradio-container ul.options li.item:hover {
    background: #eaeaea !important;
    color: var(--doc-text) !important;
}

.gradio-container button.primary {
    background: var(--doc-text) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
}
.gradio-container button.primary:hover {
    background: #3a3a3a !important;
}

/* Results Dataframe -- renders its own <table> with a light default background,
   independent of the .block/.form panel styling above. */
.gradio-container table, .gradio-container th, .gradio-container td {
    background: var(--doc-bg) !important;
    color: var(--doc-text) !important;
    border-color: var(--doc-border) !important;
}
.gradio-container th {
    background: var(--doc-surface) !important;
    font-weight: 700 !important;
}

.gradio-container button.secondary {
    background: var(--doc-bg) !important;
    color: var(--doc-text) !important;
    border: 1px solid var(--doc-border) !important;
    border-radius: 4px !important;
}
"""


def _banner(decision: str, reason: str) -> str:
    """Bold, restrained-colour text for the decision -- no pill/badge chrome, same
    "bold only for the exact state a system ends up in" convention a PRD itself
    follows, not a marketing-style status chip."""
    color = _DECISION_STYLE.get(decision, "#1a1a1a")
    return f'### <span style="color:{color};">{decision}</span>\n\n{reason}'


def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _clean_optional(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def _upload_id_or_random(value) -> str:
    """An explicit ``upload_id`` (if the user gave one, e.g. a real DB row id worth
    tracing later) always wins; otherwise a fresh random one, so the duplicate check
    still runs and stores without making the user think up an id for a one-off test."""
    return _clean_optional(value) or f"webapp-{uuid.uuid4().hex}"


def _require_columns(df: pd.DataFrame, required: set[str], all_columns_hint: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise gr.Error(f"CSV is missing required column(s): {sorted(missing)}. "
                       f"Expected: {all_columns_hint}.")


def _run_bulk_generic(csv_path, row_fn, extra_out_cols: list[str], required_cols: set[str],
                      all_columns_hint: str, progress):
    """Shared bulk-runner: reads the CSV, applies ``row_fn(image, row) -> dict`` per
    row (catching per-row errors so one bad row doesn't kill the batch), and writes a
    results CSV. ``row_fn``'s returned dict is merged into the output row alongside
    the standard image_url/decision/reason columns."""
    if csv_path is None:
        raise gr.Error("Upload a CSV first.")
    df = pd.read_csv(csv_path)
    _require_columns(df, required_cols, all_columns_hint)

    rows = []
    for _, row in progress.tqdm(list(df.iterrows()), desc="Running test cases"):
        image_url = row["image_url"]
        base = {"image_url": image_url}
        try:
            image = _fetch_image(image_url)
            base.update(row_fn(image, row))
        except Exception as e:  # noqa: BLE001 - one bad row shouldn't kill the whole batch
            base["decision"] = "ERROR"
            base["reason"] = str(e)
            for col in extra_out_cols:
                base.setdefault(col, None)
        rows.append(base)

    out_df = pd.DataFrame(rows)
    out_path = "bulk_results.csv"
    out_df.to_csv(out_path, index=False)
    return out_df, out_path


# --- Q1 only ------------------------------------------------------------------

def run_q1_individual(image, q1_backend, q1_gemini_model, claimed_vrn=None, upload_id=None):
    if image is None:
        return "### Upload an image first.", {}
    claimed_vrn = _clean_optional(claimed_vrn)
    result = run_q1_only(image, q1_backend, q1_gemini_model, claimed_vrn=claimed_vrn,
                         upload_id=_upload_id_or_random(upload_id) if claimed_vrn else None)
    return _banner(result.decision, result.reason), result.model_dump()


def run_q1_bulk(csv_path, q1_backend, q1_gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        claimed_vrn = _clean_optional(row.get("truck_number"))
        r = run_q1_only(image, q1_backend, q1_gemini_model, claimed_vrn=claimed_vrn,
                        upload_id=_upload_id_or_random(row.get("upload_id")) if claimed_vrn else None)
        return {"decision": r.decision, "reason": r.reason, "vehicle_type": r.vehicle_type,
               "view": r.view, "is_front": r.is_front, "front_complete": r.front_complete,
               "confidence": r.confidence, "duplicate_is_suspect": r.duplicate_is_suspect}

    return _run_bulk_generic(
        csv_path, row_fn,
        ["vehicle_type", "view", "is_front", "front_complete", "confidence", "duplicate_is_suspect"],
        {"image_url"}, "image_url, truck_number (optional), upload_id (optional)", progress)


# --- Q2 only ------------------------------------------------------------------

def run_q2_individual(image, truck_number, q2_backend, q2_gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number:
        return "### Truck number is required.", {}
    result = run_q2_only(image, truck_number, q2_backend, q2_gemini_model)
    return _banner(result.decision, result.reason), result.model_dump()


def run_q2_bulk(csv_path, q2_backend, q2_gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        r = run_q2_only(image, str(row["truck_number"]), q2_backend, q2_gemini_model)
        return {"truck_number": row["truck_number"], "decision": r.decision, "reason": r.reason,
               "status": r.status, "extracted_raw": r.extracted_raw, "plate_colour": r.plate_colour,
               "inferred": r.inferred}

    return _run_bulk_generic(csv_path, row_fn,
                             ["status", "extracted_raw", "plate_colour", "inferred"],
                             {"image_url", "truck_number"}, "image_url, truck_number", progress)


# --- Q3 only ------------------------------------------------------------------

def run_q3_individual(image, make, model, q3_make_backend, q3_model_backend,
                      q3_make_gemini_model, q3_model_gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not make:
        return "### Make is required.", {}
    result = run_q3_only(image, make, model or None, q3_make_backend, q3_model_backend,
                         q3_make_gemini_model, q3_model_gemini_model)
    return _banner(result["decision"], result["reason"]), result


def run_q3_bulk(csv_path, q3_make_backend, q3_model_backend, q3_make_gemini_model,
               q3_model_gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        r = run_q3_only(image, str(row["make"]), _clean_optional(row.get("model")),
                        q3_make_backend, q3_model_backend, q3_make_gemini_model, q3_model_gemini_model)
        return {"make": row["make"], "decision": r["decision"], "reason": r["reason"],
               "make_status": r["make_status"], "make_match_via": r["make_match_via"],
               "model_checked": r["model_checked"], "model_status": r["model_status"],
               "extracted_model": r["extracted_model"]}

    return _run_bulk_generic(csv_path, row_fn,
                             ["make_status", "make_match_via", "model_checked", "model_status",
                              "extracted_model"],
                             {"image_url", "make"}, "image_url, make, model (optional)", progress)


# --- End-to-end -----------------------------------------------------------------

def run_e2e_individual(image, truck_number, make, model, q1_backend, q2_backend,
                       q3_make_backend, q3_model_backend,
                       q1_gemini_model, q2_gemini_model, q3_make_gemini_model, q3_model_gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number or not make:
        return "### Truck number and make are both required.", {}

    result = run_test_case(
        image, truck_number, make, model or None,
        q1_backend=q1_backend, q2_backend=q2_backend,
        q3_make_backend=q3_make_backend, q3_model_backend=q3_model_backend,
        q1_gemini_model=q1_gemini_model, q2_gemini_model=q2_gemini_model,
        q3_make_gemini_model=q3_make_gemini_model, q3_model_gemini_model=q3_model_gemini_model,
    )
    return _banner(result.overall_decision, result.overall_reason), result.model_dump()


def run_e2e_bulk(csv_path, q1_backend, q2_backend, q3_make_backend, q3_model_backend,
                 q1_gemini_model, q2_gemini_model, q3_make_gemini_model, q3_model_gemini_model,
                 progress=gr.Progress()):
    def row_fn(image, row):
        r = run_test_case(
            image, str(row["truck_number"]), str(row["make"]), _clean_optional(row.get("model")),
            q1_backend=q1_backend, q2_backend=q2_backend,
            q3_make_backend=q3_make_backend, q3_model_backend=q3_model_backend,
            q1_gemini_model=q1_gemini_model, q2_gemini_model=q2_gemini_model,
            q3_make_gemini_model=q3_make_gemini_model, q3_model_gemini_model=q3_model_gemini_model,
        )
        return {"truck_number": row["truck_number"], "make": row["make"],
               "decision": r.overall_decision, "reason": r.overall_reason,
               "q1_vehicle_type": r.q1.vehicle_type,
               "q2_status": r.q2.status if r.q2 else None,
               "q3_make_status": r.q3_make_status, "q3_make_via": r.q3_make_votes}

    return _run_bulk_generic(csv_path, row_fn,
                             ["q1_vehicle_type", "q2_status", "q3_make_status", "q3_make_via"],
                             {"image_url", "truck_number", "make"},
                             "image_url, truck_number, make, model (optional)", progress)


# --- FASTag: QR / barcode read only (deterministic decode, no claim to compare) ---

def _decoded_codes_detail(read):
    sources = {}
    qr_bank_code = None
    for code in read.decoded_codes:
        if code.symbology.upper() == "QRCODE":
            fastag_id, bank_code = parse_qr_payload(code.data)
            if fastag_id:
                sources["qr"] = fastag_id
                qr_bank_code = bank_code
        else:
            sources[f"barcode:{code.symbology.lower()}"] = code.data
    return {
        "decoded_codes": [{"symbology": c.symbology, "data": c.data} for c in read.decoded_codes],
        "parsed_identity": sources,
        "qr_bank_code": qr_bank_code,
    }


def run_fastag_raw_individual(image):
    """QR/barcode decode only -- no claimed value to compare against, and no
    decision: the fraud-check logic (decide_fastag) needs all three sources
    together to judge cross-consistency, so this bucket never fabricates a
    partial pass/fail -- it only shows what was actually decoded."""
    if image is None:
        return "### Upload an image first.", {}
    r = classify_fastag_upload(image)
    if not r.get("checked"):
        return f"### Read unavailable\n\n{r.get('error', '?')}", {}
    detail = _decoded_codes_detail(r["read"])
    n = len(detail["decoded_codes"])
    summary = f"### {n} code(s) decoded" if n else "### Nothing decoded from this image"
    return summary, detail


def run_fastag_raw_bulk(csv_path, progress=gr.Progress()):
    def row_fn(image, row):
        r = classify_fastag_upload(image)
        if not r.get("checked"):
            return {"error": r.get("error")}
        return _decoded_codes_detail(r["read"])

    return _run_bulk_generic(csv_path, row_fn, ["decoded_codes", "parsed_identity", "qr_bank_code"],
                             {"image_url"}, "image_url", progress)


# --- FASTag: printed-digit OCR only (backend-selectable, no claim to compare) -----

def run_fastag_ocr_individual(image, backend, gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    r = classify_fastag_upload(image, backend=backend, vlm_model=gemini_model if backend == "gemini" else None)
    if not r.get("checked"):
        return f"### Read unavailable\n\n{r.get('error', '?')}", {}
    printed = r["read"].printed_id_text
    summary = f"### Printed digits\n\n`{printed}`" if printed else "### Nothing legible read from this image"
    return summary, {"extracted_printed_id": printed, "backend": backend}


def run_fastag_ocr_bulk(csv_path, backend, gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        r = classify_fastag_upload(image, backend=backend, vlm_model=gemini_model if backend == "gemini" else None)
        if not r.get("checked"):
            return {"error": r.get("error")}
        return {"extracted_printed_id": r["read"].printed_id_text}

    return _run_bulk_generic(csv_path, row_fn, ["extracted_printed_id"], {"image_url"}, "image_url", progress)


# --- FASTag: end-to-end ------------------------------------------------------------

def run_fastag_individual(image, fastag_id, bank_code, backend, gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not fastag_id:
        return "### FASTag id is required.", {}
    result = check_fastag_upload(image, fastag_id, bank_code or None, backend=backend,
                                 vlm_model=gemini_model if backend == "gemini" else None)
    return _banner(result.decision, result.reason), result.model_dump()


def run_fastag_bulk(csv_path, backend, gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        r = check_fastag_upload(image, str(row["fastag_id"]), _clean_optional(row.get("bank_code")),
                                backend=backend, vlm_model=gemini_model if backend == "gemini" else None)
        return {"fastag_id": row["fastag_id"], "decision": r.decision, "reason": r.reason,
               "matched_via": r.matched_via, "decoded_sources": r.decoded_sources,
               "extracted_printed_id": r.extracted_printed_id}

    return _run_bulk_generic(csv_path, row_fn,
                             ["matched_via", "decoded_sources", "extracted_printed_id"],
                             {"image_url", "fastag_id"}, "image_url, fastag_id, bank_code (optional)", progress)


# --- Side/axle: axle count only ----------------------------------------------------

def run_axle_individual(image, axle_count, axle_backend, gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if axle_count is None:
        return "### Claimed axle count is required.", {}
    result = check_axle_count(image, int(axle_count), backend=axle_backend,
                              model=gemini_model if axle_backend == "gemini" else None)
    return _banner(result.decision, result.reason), result.model_dump()


def run_axle_bulk(csv_path, axle_backend, gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        r = check_axle_count(image, int(row["axle_count"]), backend=axle_backend,
                             model=gemini_model if axle_backend == "gemini" else None)
        return {"axle_count_claimed": row["axle_count"], "decision": r.decision, "reason": r.reason,
               "axle_count": r.axle_count, "status": r.status, "lift_axle_suspected": r.lift_axle_suspected}

    return _run_bulk_generic(csv_path, row_fn, ["axle_count", "status", "lift_axle_suspected"],
                             {"image_url", "axle_count"}, "image_url, axle_count", progress)


# --- Side/axle: identity binding only -----------------------------------------------

def run_identity_individual(image, truck_number, make, front_reference):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number or not make:
        return "### Truck number and make are both required.", {}
    result = check_side_identity(image, truck_number, make, front_reference_image=front_reference)
    return _banner(result.decision, result.reason), result.model_dump()


def run_identity_bulk(csv_path, progress=gr.Progress()):
    def row_fn(image, row):
        front_ref = None
        front_ref_url = _clean_optional(row.get("front_reference_url"))
        if front_ref_url:
            front_ref = _fetch_image(front_ref_url)
        r = check_side_identity(image, str(row["truck_number"]), str(row["make"]),
                                front_reference_image=front_ref)
        return {"truck_number": row["truck_number"], "make": row["make"], "decision": r.decision,
               "reason": r.reason, "identity_bucket": r.identity_bucket}

    return _run_bulk_generic(
        csv_path, row_fn, ["identity_bucket"], {"image_url", "truck_number", "make"},
        "image_url, truck_number, make, front_reference_url (optional)", progress)


# --- Side/axle: end-to-end ----------------------------------------------------------

def run_side_individual(image, truck_number, make, axle_count, upload_id, front_reference,
                        axle_backend, gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number or not make or axle_count is None:
        return "### Truck number, make, and axle count are all required.", {}
    result = check_side_image_upload(
        image, truck_number, make, int(axle_count),
        upload_id=_upload_id_or_random(upload_id), front_reference_image=front_reference,
        axle_backend=axle_backend, axle_model=gemini_model if axle_backend == "gemini" else None,
    )
    return _banner(result.decision, result.reason), result.model_dump()


def run_side_bulk(csv_path, axle_backend, gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        front_ref = None
        front_ref_url = _clean_optional(row.get("front_reference_url"))
        if front_ref_url:
            front_ref = _fetch_image(front_ref_url)
        r = check_side_image_upload(
            image, str(row["truck_number"]), str(row["make"]), int(row["axle_count"]),
            upload_id=_upload_id_or_random(row.get("upload_id")), front_reference_image=front_ref,
            axle_backend=axle_backend, axle_model=gemini_model if axle_backend == "gemini" else None,
        )
        return {"truck_number": row["truck_number"], "make": row["make"], "decision": r.decision,
               "reason": r.reason, "axle_count": r.axle_count, "axle_status": r.axle_status,
               "identity_bucket": r.identity_bucket, "identity_decision": r.identity_decision}

    return _run_bulk_generic(
        csv_path, row_fn, ["axle_count", "axle_status", "identity_bucket", "identity_decision"],
        {"image_url", "truck_number", "make", "axle_count"},
        "image_url, truck_number, make, axle_count, front_reference_url (optional), upload_id (optional)",
        progress,
    )


# --- Reference-image library (duplicate-detection corpus) -------------------------

def run_duplicate_check(image, upload_id, truck_number, image_type, top_k, similarity_min, store):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number:
        return "### Truck number is required.", {}
    upload_id = upload_id or f"webapp-{truck_number}-{image_type}"
    result = check_duplicate(image, upload_id, truck_number, image_type=image_type,
                             top_k=int(top_k), similarity_min=float(similarity_min), store=bool(store))
    return _banner(result.decision, result.reason), result.model_dump()


def run_duplicate_bulk(csv_path, image_type, top_k, similarity_min, store, progress=gr.Progress()):
    def row_fn(image, row):
        upload_id = _clean_optional(row.get("upload_id")) or f"webapp-{row['truck_number']}-{image_type}"
        r = check_duplicate(image, upload_id, str(row["truck_number"]), image_type=image_type,
                            top_k=int(top_k), similarity_min=float(similarity_min), store=bool(store))
        return {"truck_number": row["truck_number"], "decision": r.decision, "reason": r.reason,
               "is_duplicate_suspect": r.is_duplicate_suspect, "best_match_id": r.best_match_id,
               "best_match_similarity": r.best_match_similarity, "best_match_vrn": r.best_match_vrn}

    return _run_bulk_generic(
        csv_path, row_fn,
        ["is_duplicate_suspect", "best_match_id", "best_match_similarity", "best_match_vrn"],
        {"image_url", "truck_number"}, "image_url, truck_number, upload_id (optional)", progress,
    )


def refresh_library_stats():
    try:
        counts = count_by_type()
    except DuplicateStoreError as e:
        return f"### Reference library unavailable\n\n{e}"
    if not counts:
        return "### Reference library is empty (no images stored yet for any type)."
    lines = "\n".join(f"- **{t}**: {n} image(s)" for t, n in sorted(counts.items()))
    return f"### Reference library contents\n\n{lines}"


# --- UI --------------------------------------------------------------------------

def _gemini_model_row(*, q1=False, q2=False, q3_make=False, q3_model=False):
    """A collapsed accordion with only the Gemini-model pickers relevant to the tab
    it's placed in. Returns the components created (in the order requested)."""
    comps = []
    with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
        with gr.Row():
            if q1:
                comps.append(gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL_Q1,
                                         allow_custom_value=True, label="Q1 Gemini model"))
            if q2:
                comps.append(gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL_Q2,
                                         allow_custom_value=True, label="Q2 Gemini model"))
            if q3_make:
                comps.append(gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL_Q3_MAKE,
                                         allow_custom_value=True, label="Q3 make Gemini model"))
            if q3_model:
                comps.append(gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL_Q3_MODEL,
                                         allow_custom_value=True, label="Q3 model Gemini model"))
    return comps


with gr.Blocks(title="Vehicle Front-Image Validator — Test Interface") as demo:
    gr.Markdown(
        "# Vehicle Front-Image Validator — Test Interface\n"
        "Test **one stage at a time** (fastest way to iterate on a single backend) or "
        "**end-to-end**. The **production** entry point (`vfiv.validate_upload`) always "
        "uses today's defaults regardless of what you pick here."
    )

    with gr.Tabs():
        # === Front Image (Q1 gate + Q2 VRN/colour + Q3 make/model + end-to-end) ===
        with gr.Tab("Front Image"):
            gr.Markdown(
                "All four sub-tabs test the same uploaded **front photo** — the gate "
                "(genuine/complete front view), the VRN/plate colour read, and the "
                "make/model classifier — one stage at a time, or chained end-to-end."
            )
            with gr.Tabs():
                # --- Q1 ---------------------------------------------------------------
                with gr.Tab("Q1 — front-image gate"):
                    q1_dd = gr.Dropdown(Q1_CHOICES, value="real_cv", label="Q1 backend")
                    (q1_gm_dd,) = _gemini_model_row(q1=True)

                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    q1_image_in = gr.Image(type="pil", label="Upload image")
                                    q1_vrn_in = gr.Textbox(
                                        label="Truck number / VRN (optional — fill in to also run the "
                                              "duplicate check against the 'front' reference library; "
                                              "leave blank to skip it)")
                                    q1_upload_id_in = gr.Textbox(
                                        label="Upload id (optional — auto-generated if left blank; only "
                                              "needed if you want this stored under a specific id)")
                                    q1_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    q1_decision_out = gr.Markdown()
                                    q1_json_out = gr.JSON(label="Full result")
                            q1_run_btn.click(
                                run_q1_individual,
                                inputs=[q1_image_in, q1_dd, q1_gm_dd, q1_vrn_in, q1_upload_id_in],
                                outputs=[q1_decision_out, q1_json_out])

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`, `truck_number` (optional), `upload_id` "
                                       "(optional). Fill in `truck_number` on a row to also run the "
                                       "duplicate check for it — `upload_id` is auto-generated if left blank.")
                            q1_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            q1_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            q1_table_out = gr.Dataframe(label="Results")
                            q1_download_out = gr.File(label="Download results CSV")
                            q1_bulk_btn.click(run_q1_bulk, inputs=[q1_csv_in, q1_dd, q1_gm_dd],
                                              outputs=[q1_table_out, q1_download_out])

                # --- Q2 ---------------------------------------------------------------
                with gr.Tab("Q2 — VRN + colour"):
                    q2_dd = gr.Dropdown(Q2_CHOICES, value="rekognition", label="Q2 backend")
                    (q2_gm_dd,) = _gemini_model_row(q2=True)

                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    q2_image_in = gr.Image(type="pil", label="Upload image")
                                    q2_vrn_in = gr.Textbox(label="Truck number (VRN)")
                                    q2_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    q2_decision_out = gr.Markdown()
                                    q2_json_out = gr.JSON(label="Full result")
                            q2_run_btn.click(run_q2_individual, inputs=[q2_image_in, q2_vrn_in, q2_dd, q2_gm_dd],
                                             outputs=[q2_decision_out, q2_json_out])

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`, `truck_number`. One row per test case.")
                            q2_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            q2_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            q2_table_out = gr.Dataframe(label="Results")
                            q2_download_out = gr.File(label="Download results CSV")
                            q2_bulk_btn.click(run_q2_bulk, inputs=[q2_csv_in, q2_dd, q2_gm_dd],
                                              outputs=[q2_table_out, q2_download_out])

                # --- Q3 ---------------------------------------------------------------
                with gr.Tab("Q3 — make + model"):
                    with gr.Row():
                        q3m_dd = gr.Dropdown(Q3_MAKE_CHOICES, value="siglip_rekognition", label="Q3 make backend")
                        q3mo_dd = gr.Dropdown(Q3_MODEL_CHOICES, value="claude", label="Q3 model backend")
                    q3m_gm_dd, q3mo_gm_dd = _gemini_model_row(q3_make=True, q3_model=True)

                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    q3_image_in = gr.Image(type="pil", label="Upload image")
                                    q3_make_in = gr.Textbox(label="Make")
                                    q3_model_in = gr.Textbox(label="Model (optional)")
                                    q3_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    q3_decision_out = gr.Markdown()
                                    q3_json_out = gr.JSON(label="Full result")
                            q3_run_btn.click(
                                run_q3_individual,
                                inputs=[q3_image_in, q3_make_in, q3_model_in, q3m_dd, q3mo_dd, q3m_gm_dd, q3mo_gm_dd],
                                outputs=[q3_decision_out, q3_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`, `make`, `model` (optional). "
                                       "One row per test case.")
                            q3_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            q3_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            q3_table_out = gr.Dataframe(label="Results")
                            q3_download_out = gr.File(label="Download results CSV")
                            q3_bulk_btn.click(
                                run_q3_bulk,
                                inputs=[q3_csv_in, q3m_dd, q3mo_dd, q3m_gm_dd, q3mo_gm_dd],
                                outputs=[q3_table_out, q3_download_out],
                            )

                # --- End-to-end ---------------------------------------------------------
                with gr.Tab("End-to-end (Q1 → Q2 → Q3)"):
                    with gr.Row():
                        e2e_q1_dd = gr.Dropdown(Q1_CHOICES, value="real_cv", label="Q1 backend")
                        e2e_q2_dd = gr.Dropdown(Q2_CHOICES, value="rekognition", label="Q2 backend")
                        e2e_q3m_dd = gr.Dropdown(Q3_MAKE_CHOICES, value="siglip_rekognition",
                                                 label="Q3 make backend")
                        e2e_q3mo_dd = gr.Dropdown(Q3_MODEL_CHOICES, value="claude", label="Q3 model backend")
                    e2e_q1_gm, e2e_q2_gm, e2e_q3m_gm, e2e_q3mo_gm = _gemini_model_row(
                        q1=True, q2=True, q3_make=True, q3_model=True)

                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    e2e_image_in = gr.Image(type="pil", label="Upload image")
                                    e2e_vrn_in = gr.Textbox(label="Truck number (VRN)")
                                    e2e_make_in = gr.Textbox(label="Make")
                                    e2e_model_in = gr.Textbox(label="Model (optional)")
                                    e2e_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    e2e_decision_out = gr.Markdown()
                                    e2e_json_out = gr.JSON(label="Full result")
                            e2e_run_btn.click(
                                run_e2e_individual,
                                inputs=[e2e_image_in, e2e_vrn_in, e2e_make_in, e2e_model_in,
                                       e2e_q1_dd, e2e_q2_dd, e2e_q3m_dd, e2e_q3mo_dd,
                                       e2e_q1_gm, e2e_q2_gm, e2e_q3m_gm, e2e_q3mo_gm],
                                outputs=[e2e_decision_out, e2e_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown(
                                "CSV columns: `image_url`, `truck_number`, `make`, `model` (optional). "
                                "One row per test case."
                            )
                            e2e_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            e2e_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            e2e_table_out = gr.Dataframe(label="Results")
                            e2e_download_out = gr.File(label="Download results CSV")
                            e2e_bulk_btn.click(
                                run_e2e_bulk,
                                inputs=[e2e_csv_in, e2e_q1_dd, e2e_q2_dd, e2e_q3m_dd, e2e_q3mo_dd,
                                       e2e_q1_gm, e2e_q2_gm, e2e_q3m_gm, e2e_q3mo_gm],
                                outputs=[e2e_table_out, e2e_download_out],
                            )

        # === Side/Axle Image (axle count + identity binding + duplicate + e2e) ===
        with gr.Tab("Side/Axle Image"):
            gr.Markdown(
                "Each sub-tab tests one check in isolation against the same uploaded "
                "**side/axle photo** — axle count, identity-to-claimed-vehicle binding, "
                "duplicate-photo reuse — or the full end-to-end decision that combines "
                "all three (worst of the three wins)."
            )
            axle_dd = gr.Dropdown(AXLE_COUNT_BACKENDS, value=config.AXLE_COUNT_BACKEND,
                                  label="Axle-count model backend")
            with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
                axle_gm_dd = gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL,
                                         allow_custom_value=True, label="Gemini model")

            with gr.Tabs():
                # --- Axle count only --------------------------------------------------
                with gr.Tab("Axle count"):
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    axle_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                                    axle_count_in = gr.Number(label="Claimed axle count", precision=0)
                                    axle_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    axle_decision_out = gr.Markdown()
                                    axle_json_out = gr.JSON(label="Full result")
                            axle_run_btn.click(
                                run_axle_individual,
                                inputs=[axle_image_in, axle_count_in, axle_dd, axle_gm_dd],
                                outputs=[axle_decision_out, axle_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`, `axle_count`. One row per test case.")
                            axle_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            axle_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            axle_table_out = gr.Dataframe(label="Results")
                            axle_download_out = gr.File(label="Download results CSV")
                            axle_bulk_btn.click(run_axle_bulk, inputs=[axle_csv_in, axle_dd, axle_gm_dd],
                                               outputs=[axle_table_out, axle_download_out])

                # --- Identity binding only ---------------------------------------------
                with gr.Tab("Identity binding"):
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    identity_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                                    identity_vrn_in = gr.Textbox(label="Claimed truck number (VRN)")
                                    identity_make_in = gr.Textbox(label="Claimed make")
                                    identity_front_ref_in = gr.Image(
                                        type="pil",
                                        label="On-file front photo (optional, corner_view bucket only)")
                                    identity_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    identity_decision_out = gr.Markdown()
                                    identity_json_out = gr.JSON(label="Full result")
                            identity_run_btn.click(
                                run_identity_individual,
                                inputs=[identity_image_in, identity_vrn_in, identity_make_in,
                                       identity_front_ref_in],
                                outputs=[identity_decision_out, identity_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown(
                                "CSV columns: `image_url`, `truck_number`, `make`, "
                                "`front_reference_url` (optional). One row per test case."
                            )
                            identity_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            identity_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            identity_table_out = gr.Dataframe(label="Results")
                            identity_download_out = gr.File(label="Download results CSV")
                            identity_bulk_btn.click(run_identity_bulk, inputs=[identity_csv_in],
                                                   outputs=[identity_table_out, identity_download_out])

                # --- Duplicate check only -----------------------------------------------
                with gr.Tab("Duplicate check"):
                    gr.Markdown(
                        "Checks/stores against the `side` reference corpus only (see the "
                        "**Reference Images** tab) — never compared against front or FASTag "
                        "images. Needs `VFIV_PGVECTOR_DSN` set to a reachable Postgres+pgvector "
                        "instance."
                    )
                    side_dup_type_state = gr.State("side")
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    side_dup_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                                    side_dup_vrn_in = gr.Textbox(label="Truck number (VRN)")
                                    side_dup_upload_id_in = gr.Textbox(
                                        label="Upload id (optional — auto-generated if left blank)")
                                    with gr.Row():
                                        side_dup_topk_in = gr.Number(label="Top-K neighbors to check",
                                                                    value=5, precision=0)
                                        side_dup_simmin_in = gr.Number(
                                            label="Similarity threshold", value=config.DUPLICATE_SIMILARITY_MIN)
                                    side_dup_store_in = gr.Checkbox(
                                        value=True, label="Store this image in the reference library "
                                                          "(uncheck for a pure lookup that doesn't grow "
                                                          "the corpus)")
                                    side_dup_run_btn = gr.Button("Check / Store", variant="primary")
                                with gr.Column():
                                    side_dup_decision_out = gr.Markdown()
                                    side_dup_json_out = gr.JSON(label="Full result")
                            side_dup_run_btn.click(
                                run_duplicate_check,
                                inputs=[side_dup_image_in, side_dup_upload_id_in, side_dup_vrn_in,
                                       side_dup_type_state, side_dup_topk_in, side_dup_simmin_in,
                                       side_dup_store_in],
                                outputs=[side_dup_decision_out, side_dup_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`, `truck_number`, `upload_id` (optional). "
                                       "One row per test case.")
                            side_dup_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            side_dup_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            side_dup_table_out = gr.Dataframe(label="Results")
                            side_dup_download_out = gr.File(label="Download results CSV")
                            side_dup_bulk_btn.click(
                                run_duplicate_bulk,
                                inputs=[side_dup_csv_in, side_dup_type_state, side_dup_topk_in,
                                       side_dup_simmin_in, side_dup_store_in],
                                outputs=[side_dup_table_out, side_dup_download_out],
                            )

                # --- End-to-end ----------------------------------------------------------
                with gr.Tab("End-to-end"):
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    side_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                                    side_vrn_in = gr.Textbox(label="Claimed truck number (VRN)")
                                    side_make_in = gr.Textbox(label="Claimed make")
                                    side_axle_in = gr.Number(label="Claimed axle count", precision=0)
                                    side_upload_id_in = gr.Textbox(
                                        label="Upload id (optional — auto-generated if left blank; the "
                                              "duplicate check always runs)")
                                    side_front_ref_in = gr.Image(
                                        type="pil",
                                        label="On-file front photo (optional, corner_view bucket only)")
                                    side_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    side_decision_out = gr.Markdown()
                                    side_json_out = gr.JSON(label="Full result")
                            side_run_btn.click(
                                run_side_individual,
                                inputs=[side_image_in, side_vrn_in, side_make_in, side_axle_in,
                                       side_upload_id_in, side_front_ref_in, axle_dd, axle_gm_dd],
                                outputs=[side_decision_out, side_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown(
                                "CSV columns: `image_url`, `truck_number`, `make`, `axle_count`, "
                                "`front_reference_url` (optional), `upload_id` (optional). "
                                "One row per test case."
                            )
                            side_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            side_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            side_table_out = gr.Dataframe(label="Results")
                            side_download_out = gr.File(label="Download results CSV")
                            side_bulk_btn.click(run_side_bulk, inputs=[side_csv_in, axle_dd, axle_gm_dd],
                                               outputs=[side_table_out, side_download_out])

        # === FASTag (QR/barcode read + printed-digit OCR + end-to-end) ===========
        with gr.Tab("FASTag"):
            gr.Markdown(
                "Each sub-tab tests one piece in isolation against the same uploaded "
                "**FASTag photo** — the deterministic QR/barcode decode, the printed-digit "
                "OCR read — or the full end-to-end decision, which cross-checks all three "
                "sources against each other AND the claimed value (a disagreement between "
                "legibly-read sources is itself a REJECT, checked before the isolated "
                "sub-tabs here could even show you). The isolated sub-tabs below show raw "
                "reads only, with no pass/fail of their own, since that cross-check needs "
                "all three sources together."
            )
            fastag_dd = gr.Dropdown(FASTAG_OCR_BACKENDS, value=config.FASTAG_OCR_BACKEND,
                                    label="Printed-digit OCR backend")
            with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
                fastag_gm_dd = gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL,
                                           allow_custom_value=True, label="Gemini model")

            with gr.Tabs():
                # --- QR / Barcode read only ---------------------------------------------
                with gr.Tab("QR / Barcode read"):
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    fastag_raw_image_in = gr.Image(type="pil", label="Upload FASTag photo")
                                    fastag_raw_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    fastag_raw_decision_out = gr.Markdown()
                                    fastag_raw_json_out = gr.JSON(label="Full result")
                            fastag_raw_run_btn.click(
                                run_fastag_raw_individual,
                                inputs=[fastag_raw_image_in],
                                outputs=[fastag_raw_decision_out, fastag_raw_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`. One row per test case.")
                            fastag_raw_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            fastag_raw_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            fastag_raw_table_out = gr.Dataframe(label="Results")
                            fastag_raw_download_out = gr.File(label="Download results CSV")
                            fastag_raw_bulk_btn.click(
                                run_fastag_raw_bulk, inputs=[fastag_raw_csv_in],
                                outputs=[fastag_raw_table_out, fastag_raw_download_out])

                # --- Printed-digit OCR only ---------------------------------------------
                with gr.Tab("Printed digits (OCR)"):
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    fastag_ocr_image_in = gr.Image(type="pil", label="Upload FASTag photo")
                                    fastag_ocr_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    fastag_ocr_decision_out = gr.Markdown()
                                    fastag_ocr_json_out = gr.JSON(label="Full result")
                            fastag_ocr_run_btn.click(
                                run_fastag_ocr_individual,
                                inputs=[fastag_ocr_image_in, fastag_dd, fastag_gm_dd],
                                outputs=[fastag_ocr_decision_out, fastag_ocr_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`. One row per test case.")
                            fastag_ocr_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            fastag_ocr_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            fastag_ocr_table_out = gr.Dataframe(label="Results")
                            fastag_ocr_download_out = gr.File(label="Download results CSV")
                            fastag_ocr_bulk_btn.click(
                                run_fastag_ocr_bulk, inputs=[fastag_ocr_csv_in, fastag_dd, fastag_gm_dd],
                                outputs=[fastag_ocr_table_out, fastag_ocr_download_out])

                # --- End-to-end ----------------------------------------------------------
                with gr.Tab("End-to-end"):
                    with gr.Tabs():
                        with gr.Tab("Individual"):
                            with gr.Row():
                                with gr.Column():
                                    fastag_image_in = gr.Image(type="pil", label="Upload FASTag photo")
                                    fastag_id_in = gr.Textbox(label="Claimed FASTag id")
                                    fastag_bank_in = gr.Textbox(label="Claimed bank code (optional)")
                                    fastag_run_btn = gr.Button("Run", variant="primary")
                                with gr.Column():
                                    fastag_decision_out = gr.Markdown()
                                    fastag_json_out = gr.JSON(label="Full result")
                            fastag_run_btn.click(
                                run_fastag_individual,
                                inputs=[fastag_image_in, fastag_id_in, fastag_bank_in, fastag_dd, fastag_gm_dd],
                                outputs=[fastag_decision_out, fastag_json_out],
                            )

                        with gr.Tab("Bulk (CSV)"):
                            gr.Markdown("CSV columns: `image_url`, `fastag_id`, `bank_code` (optional). "
                                       "One row per test case.")
                            fastag_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                            fastag_bulk_btn = gr.Button("Run bulk test", variant="primary")
                            fastag_table_out = gr.Dataframe(label="Results")
                            fastag_download_out = gr.File(label="Download results CSV")
                            fastag_bulk_btn.click(
                                run_fastag_bulk, inputs=[fastag_csv_in, fastag_dd, fastag_gm_dd],
                                outputs=[fastag_table_out, fastag_download_out])

        # --- Reference-image library --------------------------------------------
        with gr.Tab("Reference Images"):
            gr.Markdown(
                "Seed the duplicate-detection corpus (`validators/duplicate_check.py`) with known-good "
                "truck images, or check a new image against what's already stored — **without** running "
                "Q1/Q2/Q3. `front` / `side` / `fastag` are separate corpora and never compared against "
                "each other. Needs `VFIV_PGVECTOR_DSN` set to a reachable Postgres+pgvector instance."
            )
            dup_image_type_dd = gr.Dropdown(config.IMAGE_TYPES, value="front", label="Image type")
            dup_stats_out = gr.Markdown()
            dup_refresh_btn = gr.Button("Refresh library stats")
            dup_refresh_btn.click(refresh_library_stats, inputs=[], outputs=[dup_stats_out])

            with gr.Tabs():
                with gr.Tab("Individual"):
                    with gr.Row():
                        with gr.Column():
                            dup_image_in = gr.Image(type="pil", label="Upload image")
                            dup_vrn_in = gr.Textbox(label="Truck number (VRN)")
                            dup_upload_id_in = gr.Textbox(
                                label="Upload id (optional — auto-generated from VRN + image type if blank)")
                            with gr.Row():
                                dup_topk_in = gr.Number(label="Top-K neighbors to check", value=5, precision=0)
                                dup_simmin_in = gr.Number(label="Similarity threshold",
                                                          value=config.DUPLICATE_SIMILARITY_MIN)
                            dup_store_in = gr.Checkbox(
                                value=True, label="Store this image in the reference library "
                                                  "(uncheck for a pure lookup that doesn't grow the corpus)")
                            dup_run_btn = gr.Button("Check / Store", variant="primary")
                        with gr.Column():
                            dup_decision_out = gr.Markdown()
                            dup_json_out = gr.JSON(label="Full result")
                    dup_run_btn.click(
                        run_duplicate_check,
                        inputs=[dup_image_in, dup_upload_id_in, dup_vrn_in, dup_image_type_dd,
                               dup_topk_in, dup_simmin_in, dup_store_in],
                        outputs=[dup_decision_out, dup_json_out],
                    )

                with gr.Tab("Bulk (CSV)"):
                    gr.Markdown("CSV columns: `image_url`, `truck_number`, `upload_id` (optional). "
                               "One row per reference image — uses the image type and settings above.")
                    dup_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    dup_bulk_btn = gr.Button("Run bulk test", variant="primary")
                    dup_table_out = gr.Dataframe(label="Results")
                    dup_download_out = gr.File(label="Download results CSV")
                    dup_bulk_btn.click(
                        run_duplicate_bulk,
                        inputs=[dup_csv_in, dup_image_type_dd, dup_topk_in, dup_simmin_in, dup_store_in],
                        outputs=[dup_table_out, dup_download_out],
                    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CUSTOM_CSS)
