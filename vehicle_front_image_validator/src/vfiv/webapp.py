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

import gradio as gr
import pandas as pd
import requests
from PIL import Image

from vfiv import config
from vfiv.backends.fastag_reader import FASTAG_OCR_BACKENDS
from vfiv.backends.vector_store import DuplicateStoreError, count_by_type
from vfiv.experiments import q1_select, q2_select, q3_select
from vfiv.experiments.runner import run_q1_only, run_q2_only, run_q3_only, run_test_case
from vfiv.validators.duplicate_check import check_duplicate
from vfiv.validators.fastag_check import check_fastag_upload
from vfiv.validators.side_image_check import AXLE_COUNT_BACKENDS, check_side_image_upload

Q1_CHOICES = q1_select.Q1_BACKENDS
Q2_CHOICES = q2_select.Q2_BACKENDS
Q3_MAKE_CHOICES = q3_select.Q3_MAKE_BACKENDS
Q3_MODEL_CHOICES = q3_select.Q3_MODEL_BACKENDS

# Common Gemini 2.5 model ids — the dropdown also accepts a typed custom value (e.g. a
# future model id), so this list is a convenience, not an exhaustive whitelist.
GEMINI_MODEL_CHOICES = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

# Brand tokens pulled from blackbuck.com's own compiled Webflow CSS (its :root custom
# properties and most-used hex values) — dark-first UI, a signature turquoise accent,
# a bold display face for headings over a clean geometric sans for everything else.
# The exact brand fonts (Drukwide / Generalsans) are commercial and not embedded here;
# Archivo Black / Inter are free stand-ins with a similar weight and character.
_BB_TURQUOISE = "#0dcbc2"
_BB_CRIMSON = "#d71e48"
_BB_YELLOW = "#ffc130"

_DECISION_STYLE = {
    "PASS": (_BB_TURQUOISE, "#000"),
    "REJECT": (_BB_CRIMSON, "#fff"),
    "MANUAL_REVIEW": (_BB_YELLOW, "#000"),
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bb-black: #0a0a0a;
    --bb-panel: #151515;
    --bb-panel-2: #1e1e1e;
    --bb-white: #ffffff;
    --bb-muted: #a8a8a8;
    --bb-turquoise: #0dcbc2;
    --bb-turquoise-dark: #0b8e88;
    --bb-border: #2a2a2a;
}

.gradio-container {
    background: var(--bb-black) !important;
    color: var(--bb-white) !important;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    font-family: 'Archivo Black', 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
    color: var(--bb-white) !important;
}

.gradio-container .prose, .gradio-container p, .gradio-container span, .gradio-container label {
    color: var(--bb-white) !important;
}

/* Gradio's markdown typography gives <strong>/<code> their own explicit (dark) colour
   that plain inheritance from .prose doesn't override -- target them directly. */
.gradio-container .prose strong {
    color: var(--bb-white) !important;
}
.gradio-container .prose code {
    color: var(--bb-turquoise) !important;
    background: var(--bb-panel-2) !important;
}

.gradio-container .block, .gradio-container .form {
    background: var(--bb-panel) !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 14px !important;
}

/* The small label caption Gradio renders above a component (e.g. "Upload image",
   "Full result") ships with its own light background, under either a ".float" class
   (image/file-type inputs) or a "block-label" test-id (JSON/dataframe/etc. outputs) --
   without this override both render as an unreadable white-on-white bar once the text
   colour above is forced to white. */
.gradio-container label.float, .gradio-container .float,
.gradio-container [data-testid="block-label"] {
    background: var(--bb-panel-2) !important;
    color: var(--bb-white) !important;
}

/* Tab bars -- targeted by the ARIA role (stable across Gradio versions) rather than
   Gradio's internal class names, which changed between major versions. */
.gradio-container button[role="tab"] {
    color: var(--bb-muted) !important;
    font-weight: 600 !important;
}
.gradio-container button[role="tab"].selected {
    color: var(--bb-turquoise) !important;
    border-bottom: 2px solid var(--bb-turquoise) !important;
}

.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: var(--bb-panel-2) !important;
    color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 10px !important;
}

/* Dropdown popover -- Gradio builds this as a plain <ul class="options"><li
   class="item"> list, not a native <select>, so the input/textarea/select rule
   above never reaches it; it renders with the browser's default light background
   otherwise. */
.gradio-container ul.options {
    background: var(--bb-panel-2) !important;
    border: 1px solid var(--bb-border) !important;
}
.gradio-container ul.options li.item {
    color: var(--bb-white) !important;
    background: var(--bb-panel-2) !important;
}
/* The currently-selected option ships a literal Tailwind bg-gray-100 utility class
   (a hardcoded light colour, not theme-driven) that otherwise wins over the rule
   above -- override it explicitly rather than relying on cascade order. */
.gradio-container ul.options li.item.selected,
.gradio-container ul.options li.item.active {
    background: var(--bb-panel-2) !important;
    color: var(--bb-white) !important;
}
.gradio-container ul.options li.item:hover {
    background: var(--bb-turquoise) !important;
    color: #000 !important;
}

.gradio-container button.primary {
    background: var(--bb-turquoise) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
}
.gradio-container button.primary:hover {
    background: var(--bb-turquoise-dark) !important;
    color: #fff !important;
}

/* Results Dataframe -- renders its own <table> with a light default background,
   independent of the .block/.form panel styling above. */
.gradio-container table, .gradio-container th, .gradio-container td {
    background: var(--bb-panel) !important;
    color: var(--bb-white) !important;
    border-color: var(--bb-border) !important;
}

.gradio-container button.secondary {
    background: transparent !important;
    color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 999px !important;
}
"""


def _banner(decision: str, reason: str) -> str:
    bg, fg = _DECISION_STYLE.get(decision, ("#333", "#fff"))
    badge = (f'<span style="background:{bg};color:{fg};padding:0.35em 0.9em;'
             f'border-radius:999px;font-weight:700;letter-spacing:0.02em;">{decision}</span>')
    return f"### {badge}\n\n{reason}"


def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _clean_optional(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


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

def run_q1_individual(image, q1_backend, q1_gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    result = run_q1_only(image, q1_backend, q1_gemini_model)
    return _banner(result.decision, result.reason), result.model_dump()


def run_q1_bulk(csv_path, q1_backend, q1_gemini_model, progress=gr.Progress()):
    def row_fn(image, row):
        r = run_q1_only(image, q1_backend, q1_gemini_model)
        return {"decision": r.decision, "reason": r.reason, "vehicle_type": r.vehicle_type,
               "view": r.view, "is_front": r.is_front, "front_complete": r.front_complete,
               "confidence": r.confidence}

    return _run_bulk_generic(csv_path, row_fn,
                             ["vehicle_type", "view", "is_front", "front_complete", "confidence"],
                             {"image_url"}, "image_url", progress)


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


# --- FASTag only ------------------------------------------------------------------

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


# --- Side/axle image only ----------------------------------------------------------

def run_side_individual(image, truck_number, make, axle_count, upload_id, front_reference,
                        axle_backend, gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number or not make or axle_count is None:
        return "### Truck number, make, and axle count are all required.", {}
    result = check_side_image_upload(
        image, truck_number, make, int(axle_count),
        upload_id=upload_id or None, front_reference_image=front_reference,
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
            upload_id=_clean_optional(row.get("upload_id")), front_reference_image=front_ref,
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
        # --- Q1 ---------------------------------------------------------------
        with gr.Tab("Q1 — front-image gate"):
            q1_dd = gr.Dropdown(Q1_CHOICES, value="real_cv", label="Q1 backend")
            (q1_gm_dd,) = _gemini_model_row(q1=True)

            with gr.Tabs():
                with gr.Tab("Individual"):
                    with gr.Row():
                        with gr.Column():
                            q1_image_in = gr.Image(type="pil", label="Upload image")
                            q1_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            q1_decision_out = gr.Markdown()
                            q1_json_out = gr.JSON(label="Full result")
                    q1_run_btn.click(run_q1_individual, inputs=[q1_image_in, q1_dd, q1_gm_dd],
                                     outputs=[q1_decision_out, q1_json_out])

                with gr.Tab("Bulk (CSV)"):
                    gr.Markdown("CSV columns: `image_url`. One row per test case.")
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
                    gr.Markdown("CSV columns: `image_url`, `make`, `model` (optional). One row per test case.")
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
                e2e_q3m_dd = gr.Dropdown(Q3_MAKE_CHOICES, value="siglip_rekognition", label="Q3 make backend")
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

        # --- FASTag ---------------------------------------------------------------
        with gr.Tab("FASTag"):
            fastag_dd = gr.Dropdown(FASTAG_OCR_BACKENDS, value=config.FASTAG_OCR_BACKEND,
                                    label="Printed-digit OCR backend")
            with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
                fastag_gm_dd = gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL,
                                           allow_custom_value=True, label="Gemini model")

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
                    fastag_bulk_btn.click(run_fastag_bulk, inputs=[fastag_csv_in, fastag_dd, fastag_gm_dd],
                                         outputs=[fastag_table_out, fastag_download_out])

        # --- Side/axle image --------------------------------------------------------
        with gr.Tab("Side/axle image"):
            axle_dd = gr.Dropdown(AXLE_COUNT_BACKENDS, value=config.AXLE_COUNT_BACKEND,
                                  label="Axle-count model backend")
            with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
                axle_gm_dd = gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL,
                                         allow_custom_value=True, label="Gemini model")

            with gr.Tabs():
                with gr.Tab("Individual"):
                    with gr.Row():
                        with gr.Column():
                            side_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                            side_vrn_in = gr.Textbox(label="Claimed truck number (VRN)")
                            side_make_in = gr.Textbox(label="Claimed make")
                            side_axle_in = gr.Number(label="Claimed axle count", precision=0)
                            side_upload_id_in = gr.Textbox(
                                label="Upload id (optional — leave blank to skip the duplicate check)")
                            side_front_ref_in = gr.Image(
                                type="pil", label="On-file front photo (optional, corner_view bucket only)")
                            side_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            side_decision_out = gr.Markdown()
                            side_json_out = gr.JSON(label="Full result")
                    side_run_btn.click(
                        run_side_individual,
                        inputs=[side_image_in, side_vrn_in, side_make_in, side_axle_in, side_upload_id_in,
                               side_front_ref_in, axle_dd, axle_gm_dd],
                        outputs=[side_decision_out, side_json_out],
                    )

                with gr.Tab("Bulk (CSV)"):
                    gr.Markdown(
                        "CSV columns: `image_url`, `truck_number`, `make`, `axle_count`, "
                        "`front_reference_url` (optional), `upload_id` (optional). One row per test case."
                    )
                    side_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    side_bulk_btn = gr.Button("Run bulk test", variant="primary")
                    side_table_out = gr.Dataframe(label="Results")
                    side_download_out = gr.File(label="Download results CSV")
                    side_bulk_btn.click(run_side_bulk, inputs=[side_csv_in, axle_dd, axle_gm_dd],
                                       outputs=[side_table_out, side_download_out])

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
