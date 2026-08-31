"""Test/inference interface — one stage at a time or end-to-end, with a selectable
backend per stage, so different CV/LLM combinations can be compared side by side.
Uses Gradio, matching ``truck_verification_pipeline``'s own choice for this kind of
dev/test tool in this ecosystem.

This is a TESTING tool, not the production entry point — the production entry point is
``vfiv.validate_upload`` (``combined.py``), which always uses today's default
backends. Run with:

    python -m vfiv.webapp
"""
from __future__ import annotations

import uuid

import gradio as gr

from vfiv import config
from vfiv.backends.fastag_reader import FASTAG_OCR_BACKENDS, parse_qr_payload
from vfiv.backends.vector_store import DuplicateStoreError, count_by_type
from vfiv.experiments import q1_select, q2_select, q3_select
from vfiv.experiments.runner import run_q1_only, run_q2_only, run_q3_only, run_test_case
from vfiv.duplicate_check import check_duplicate, store_reference_image
from vfiv.fastag_image.fastag_check import (
    check_fastag_upload,
    classify_fastag_upload,
    decide_printed_digits_only,
    decide_qr_only,
)
from vfiv.side_image.side_image_check import (
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


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _upload_id_or_random(value: str | None) -> str:
    """An explicit ``upload_id`` (if the user gave one, e.g. a real DB row id worth
    tracing later) always wins; otherwise a fresh random one, so the duplicate check
    still runs and stores without making the user think up an id for a one-off test."""
    return _clean_optional(value) or f"webapp-{uuid.uuid4().hex}"


# --- Q1 only ------------------------------------------------------------------

def run_q1_individual(image, q1_backend, q1_gemini_model, claimed_vrn=None, upload_id=None):
    if image is None:
        return "### Upload an image first.", {}
    claimed_vrn = _clean_optional(claimed_vrn)
    result = run_q1_only(image, q1_backend, q1_gemini_model, claimed_vrn=claimed_vrn,
                         upload_id=_upload_id_or_random(upload_id) if claimed_vrn else None)
    return _banner(result.decision, result.reason), result.model_dump()


# --- Q2 only ------------------------------------------------------------------

def run_q2_individual(image, truck_number, q2_backend, q2_gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number:
        return "### Truck number is required.", {}
    result = run_q2_only(image, truck_number, q2_backend, q2_gemini_model)
    return _banner(result.decision, result.reason), result.model_dump()


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


# --- End-to-end -----------------------------------------------------------------

def run_e2e_individual(image, truck_number, make, model, q1_backend, q2_backend,
                       q3_make_backend, q3_model_backend,
                       q1_gemini_model, q2_gemini_model, q3_make_gemini_model, q3_model_gemini_model,
                       upload_id=None):
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
        upload_id=_upload_id_or_random(upload_id),
    )
    return _banner(result.overall_decision, result.overall_reason), result.model_dump()


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


def run_fastag_raw_individual(image, backend, gemini_model, claimed_fastag_id=None, claimed_bank_code=None):
    """QR/barcode decode -- always shows the raw decode. The full cross-source
    fraud check (decide_fastag, needing all three sources together to judge
    tamper-consistency) still only lives in the End-to-end tab; but with a
    claimed Tag ID, this now ALSO runs the QR-only verdict (exact match against
    the QR's decoded tag id, see decide_qr_only) -- narrower than the full
    check, but a real pass/fail on its own. Leave the Tag ID blank for the old
    raw-read-only behaviour.

    The QR/barcode decode itself is unaffected by ``backend`` (it's a
    deterministic algorithm, not a model call) -- but classify_fastag_upload
    ALSO reads the printed digits internally regardless of which tab is asking,
    so ``backend`` still needs to be threaded through here to avoid always
    defaulting to "rekognition" (and its AWS dependency) even when the shared
    dropdown above says otherwise."""
    if image is None:
        return "### Upload an image first.", {}
    r = classify_fastag_upload(image, backend=backend, vlm_model=gemini_model if backend == "gemini" else None)
    if not r.get("checked"):
        return f"### Read unavailable\n\n{r.get('error', '?')}", {}
    detail = _decoded_codes_detail(r["read"])
    claimed_fastag_id = _clean_optional(claimed_fastag_id)
    if not claimed_fastag_id:
        n = len(detail["decoded_codes"])
        summary = f"### {n} code(s) decoded" if n else "### Nothing decoded from this image"
        return summary, detail
    verdict = decide_qr_only(r, claimed_fastag_id, _clean_optional(claimed_bank_code))
    return _banner(verdict.decision, verdict.reason), {**detail, **verdict.model_dump()}


# --- FASTag: printed-digit OCR only (backend-selectable) --------------------------

def run_fastag_ocr_individual(image, backend, gemini_model, claimed_barcode=None):
    """Printed-digit OCR -- always shows the raw read. With a claimed barcode
    value (e.g. from a separate handheld barcode scan, not decoded from this
    same photo), also runs the printed-digits-only verdict (fuzzy match, see
    decide_printed_digits_only) -- independent of the full cross-source check.
    Leave the barcode field blank for the old raw-read-only behaviour."""
    if image is None:
        return "### Upload an image first.", {}
    r = classify_fastag_upload(image, backend=backend, vlm_model=gemini_model if backend == "gemini" else None)
    if not r.get("checked"):
        return f"### Read unavailable\n\n{r.get('error', '?')}", {}
    printed = r["read"].printed_id_text
    claimed_barcode = _clean_optional(claimed_barcode)
    if not claimed_barcode:
        summary = f"### Printed digits\n\n`{printed}`" if printed else "### Nothing legible read from this image"
        return summary, {"extracted_printed_id": printed, "backend": backend}
    verdict = decide_printed_digits_only(r, claimed_barcode)
    return _banner(verdict.decision, verdict.reason), verdict.model_dump()


# --- FASTag: end-to-end ------------------------------------------------------------

def run_fastag_individual(image, fastag_id, bank_code, backend, gemini_model,
                          claimed_vrn=None, upload_id=None):
    if image is None:
        return "### Upload an image first.", {}
    if not fastag_id:
        return "### FASTag id is required.", {}
    claimed_vrn = _clean_optional(claimed_vrn)
    result = check_fastag_upload(image, fastag_id, bank_code or None, backend=backend,
                                 vlm_model=gemini_model if backend == "gemini" else None,
                                 claimed_vrn=claimed_vrn,
                                 upload_id=_upload_id_or_random(upload_id) if claimed_vrn else None)
    return _banner(result.decision, result.reason), result.model_dump()


# --- Side/axle: axle count only ----------------------------------------------------

def run_axle_individual(image, axle_count, axle_backend, gemini_model, axle_source, vehicle_mapper):
    if image is None:
        return "### Upload an image first.", {}
    if axle_count is None:
        return "### Claimed axle count is required.", {}
    result = check_axle_count(image, int(axle_count), backend=axle_backend,
                              model=gemini_model if axle_backend == "gemini" else None,
                              axle_source=_clean_optional(axle_source),
                              vehicle_mapper=_clean_optional(vehicle_mapper))
    return _banner(result.decision, result.reason), result.model_dump()


# --- Side/axle: identity binding only -----------------------------------------------

def run_identity_individual(image, truck_number, make, front_reference, axle_backend, gemini_model):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number or not make:
        return "### Truck number and make are both required.", {}
    result = check_side_identity(image, truck_number, make, front_reference_image=front_reference,
                                 type_backend=axle_backend,
                                 type_model=gemini_model if axle_backend == "gemini" else None)
    return _banner(result.decision, result.reason), result.model_dump()


# --- Side/axle: duplicate check only -------------------------------------------------

def run_side_duplicate_individual(image, truck_number, upload_id, top_k, similarity_min, store):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number:
        return "### Truck number is required.", {}
    result = check_duplicate(image, _upload_id_or_random(upload_id), truck_number, image_type="side",
                             top_k=int(top_k), similarity_min=float(similarity_min), store=bool(store))
    return _banner(result.decision, result.reason), result.model_dump()


# --- Side/axle: end-to-end ----------------------------------------------------------

def run_side_individual(image, truck_number, make, axle_count, upload_id, front_reference,
                        axle_backend, gemini_model, axle_source, vehicle_mapper):
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number or not make or axle_count is None:
        return "### Truck number, make, and axle count are all required.", {}
    result = check_side_image_upload(
        image, truck_number, make, int(axle_count),
        upload_id=_upload_id_or_random(upload_id), front_reference_image=front_reference,
        axle_backend=axle_backend, axle_model=gemini_model if axle_backend == "gemini" else None,
        axle_source=_clean_optional(axle_source), vehicle_mapper=_clean_optional(vehicle_mapper),
        type_backend=axle_backend, type_model=gemini_model if axle_backend == "gemini" else None,
    )
    return _banner(result.decision, result.reason), result.model_dump()


# --- Reference-image library (legacy-data seeding, NOT a check) --------------------

def run_reference_store(image, truck_number, image_type, upload_id):
    """Vectorize and store ONLY -- e.g. importing a legacy dump of photos that
    predates this vector-DB setup. The actual duplicate CHECK happens later, when a
    real front/side/FASTag upload is tested (see the duplicate-check bucket under
    Side/Axle Image, or the optional VRN field on Q1/FASTag's end-to-end) -- not
    here, and this never produces a PASS/REJECT/MANUAL_REVIEW decision."""
    if image is None:
        return "### Upload an image first.", {}
    if not truck_number:
        return "### Truck number is required.", {}
    result = store_reference_image(image, _upload_id_or_random(upload_id), truck_number,
                                   image_type=image_type)
    if not result.stored:
        return f"### Not stored\n\n{result.reason}", result.model_dump()
    return f"### Stored\n\nSaved under `{result.upload_id}` ({image_type}).", result.model_dump()


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


with gr.Blocks(title="Vehicle Image Validator — Test Interface") as demo:
    gr.Markdown(
        "# Vehicle Image Validator — Test Interface\n"
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

                # --- Q2 ---------------------------------------------------------------
                with gr.Tab("Q2 — VRN + colour"):
                    q2_dd = gr.Dropdown(Q2_CHOICES, value="rekognition", label="Q2 backend")
                    (q2_gm_dd,) = _gemini_model_row(q2=True)

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

                # --- Q3 ---------------------------------------------------------------
                with gr.Tab("Q3 — make + model"):
                    with gr.Row():
                        q3m_dd = gr.Dropdown(Q3_MAKE_CHOICES, value="siglip_rekognition", label="Q3 make backend")
                        q3mo_dd = gr.Dropdown(Q3_MODEL_CHOICES, value="claude", label="Q3 model backend")
                    q3m_gm_dd, q3mo_gm_dd = _gemini_model_row(q3_make=True, q3_model=True)

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

                    with gr.Row():
                        with gr.Column():
                            e2e_image_in = gr.Image(type="pil", label="Upload image")
                            e2e_vrn_in = gr.Textbox(label="Truck number (VRN)")
                            e2e_make_in = gr.Textbox(label="Make")
                            e2e_model_in = gr.Textbox(label="Model (optional)")
                            e2e_upload_id_in = gr.Textbox(
                                label="Upload id (optional — auto-generated if left blank; the "
                                      "duplicate check against the 'front' reference library always "
                                      "runs as part of Q1)")
                            e2e_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            e2e_decision_out = gr.Markdown()
                            e2e_json_out = gr.JSON(label="Full result")
                    e2e_run_btn.click(
                        run_e2e_individual,
                        inputs=[e2e_image_in, e2e_vrn_in, e2e_make_in, e2e_model_in,
                               e2e_q1_dd, e2e_q2_dd, e2e_q3m_dd, e2e_q3mo_dd,
                               e2e_q1_gm, e2e_q2_gm, e2e_q3m_gm, e2e_q3mo_gm, e2e_upload_id_in],
                        outputs=[e2e_decision_out, e2e_json_out],
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
                                  label="Model backend (axle count read + identity-bucket routing)")
            with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
                axle_gm_dd = gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL,
                                         allow_custom_value=True, label="Gemini model")

            with gr.Tabs():
                # --- Axle count only --------------------------------------------------
                with gr.Tab("Axle count"):
                    with gr.Row():
                        with gr.Column():
                            axle_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                            axle_count_in = gr.Number(label="Claimed axle count", precision=0)
                            axle_source_in = gr.Dropdown(
                                ["", "auto", "manual"], value="",
                                label="Axle count source (optional — leave blank to skip the RC "
                                      "cross-check entirely)")
                            axle_vehicle_mapper_in = gr.Textbox(
                                label="Vehicle mapper class (e.g. VC12 — only used when source is "
                                      "\"manual\")")
                            axle_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            axle_decision_out = gr.Markdown()
                            axle_json_out = gr.JSON(label="Full result")
                    axle_run_btn.click(
                        run_axle_individual,
                        inputs=[axle_image_in, axle_count_in, axle_dd, axle_gm_dd,
                               axle_source_in, axle_vehicle_mapper_in],
                        outputs=[axle_decision_out, axle_json_out],
                    )

                # --- Identity binding only ---------------------------------------------
                with gr.Tab("Identity binding"):
                    with gr.Row():
                        with gr.Column():
                            identity_image_in = gr.Image(type="pil", label="Upload side/axle photo")
                            identity_vrn_in = gr.Textbox(label="Claimed truck number (VRN)")
                            identity_make_in = gr.Textbox(label="Claimed make")
                            identity_front_ref_in = gr.Image(
                                type="pil", label="On-file front photo (optional, corner_view bucket only)")
                            identity_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            identity_decision_out = gr.Markdown()
                            identity_json_out = gr.JSON(label="Full result")
                    identity_run_btn.click(
                        run_identity_individual,
                        inputs=[identity_image_in, identity_vrn_in, identity_make_in, identity_front_ref_in,
                               axle_dd, axle_gm_dd],
                        outputs=[identity_decision_out, identity_json_out],
                    )

                # --- Duplicate check only -----------------------------------------------
                with gr.Tab("Duplicate check"):
                    gr.Markdown(
                        "Checks against the `side` reference corpus only (see the "
                        "**Reference Images** tab) — never compared against front or FASTag "
                        "images. Needs `VFIV_PGVECTOR_DSN` set to a reachable Postgres+pgvector "
                        "instance."
                    )
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
                                value=True, label="Also store this image in the reference library "
                                                  "(uncheck for a pure lookup that doesn't grow "
                                                  "the corpus)")
                            side_dup_run_btn = gr.Button("Check", variant="primary")
                        with gr.Column():
                            side_dup_decision_out = gr.Markdown()
                            side_dup_json_out = gr.JSON(label="Full result")
                    side_dup_run_btn.click(
                        run_side_duplicate_individual,
                        inputs=[side_dup_image_in, side_dup_vrn_in, side_dup_upload_id_in,
                               side_dup_topk_in, side_dup_simmin_in, side_dup_store_in],
                        outputs=[side_dup_decision_out, side_dup_json_out],
                    )

                # --- End-to-end ----------------------------------------------------------
                with gr.Tab("End-to-end"):
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
                                type="pil", label="On-file front photo (optional, corner_view bucket only)")
                            side_axle_source_in = gr.Dropdown(
                                ["", "auto", "manual"], value="",
                                label="Axle count source (optional — leave blank to skip the RC "
                                      "cross-check entirely)")
                            side_vehicle_mapper_in = gr.Textbox(
                                label="Vehicle mapper class (e.g. VC12 — only used when source is "
                                      "\"manual\")")
                            side_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            side_decision_out = gr.Markdown()
                            side_json_out = gr.JSON(label="Full result")
                    side_run_btn.click(
                        run_side_individual,
                        inputs=[side_image_in, side_vrn_in, side_make_in, side_axle_in,
                               side_upload_id_in, side_front_ref_in, axle_dd, axle_gm_dd,
                               side_axle_source_in, side_vehicle_mapper_in],
                        outputs=[side_decision_out, side_json_out],
                    )

        # === FASTag (QR/barcode read + printed-digit OCR + end-to-end) ===========
        with gr.Tab("FASTag"):
            gr.Markdown(
                "Each sub-tab tests one piece in isolation against the same uploaded "
                "**FASTag photo** — the deterministic QR/barcode decode, the printed-digit "
                "OCR read — or the full end-to-end decision, which cross-checks all three "
                "sources against each other AND the claimed value (a disagreement between "
                "legibly-read sources is itself a REJECT, checked before the isolated "
                "sub-tabs here could even show you). The isolated sub-tabs below always show "
                "the raw read; give them a claimed value and they'll also return a narrower "
                "pass/fail verdict on that ONE source alone (no cross-source tamper check — "
                "that still only lives in End-to-end)."
            )
            fastag_dd = gr.Dropdown(FASTAG_OCR_BACKENDS, value=config.FASTAG_OCR_BACKEND,
                                    label="Printed-digit OCR backend")
            with gr.Accordion("Gemini model (only used if the backend above is \"gemini\")", open=False):
                fastag_gm_dd = gr.Dropdown(GEMINI_MODEL_CHOICES, value=config.GEMINI_MODEL,
                                           allow_custom_value=True, label="Gemini model")

            with gr.Tabs():
                # --- QR / Barcode read (+ optional QR-only verdict) --------------------
                with gr.Tab("QR / Barcode read"):
                    gr.Markdown(
                        "Always shows the raw QR/barcode decode. Fill in a claimed **Tag "
                        "ID** to also get a pass/fail verdict on the QR code alone (exact "
                        "match, parsed from the QR's `<TagID>@<bank_code>` payload) — this "
                        "is narrower than the End-to-end tab's full cross-source check, "
                        "which also verifies the barcode/OCR agree with each other."
                    )
                    with gr.Row():
                        with gr.Column():
                            fastag_raw_image_in = gr.Image(type="pil", label="Upload FASTag photo")
                            fastag_raw_tagid_in = gr.Textbox(
                                label="Claimed Tag ID (optional — leave blank for a raw "
                                      "read with no verdict)")
                            fastag_raw_bank_in = gr.Textbox(label="Claimed bank code (optional)")
                            fastag_raw_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            fastag_raw_decision_out = gr.Markdown()
                            fastag_raw_json_out = gr.JSON(label="Full result")
                    fastag_raw_run_btn.click(
                        run_fastag_raw_individual,
                        inputs=[fastag_raw_image_in, fastag_dd, fastag_gm_dd,
                               fastag_raw_tagid_in, fastag_raw_bank_in],
                        outputs=[fastag_raw_decision_out, fastag_raw_json_out],
                    )

                # --- Printed-digit OCR (+ optional printed-digits-only verdict) --------
                with gr.Tab("Printed digits (OCR)"):
                    gr.Markdown(
                        "Always shows the raw OCR read. Fill in a claimed **barcode "
                        "number** (e.g. from a separate handheld barcode scan, not decoded "
                        "from this same photo) to also get a fuzzy-match pass/fail verdict "
                        "on the printed digits alone."
                    )
                    with gr.Row():
                        with gr.Column():
                            fastag_ocr_image_in = gr.Image(type="pil", label="Upload FASTag photo")
                            fastag_ocr_barcode_in = gr.Textbox(
                                label="Claimed barcode number (optional — leave blank for a "
                                      "raw read with no verdict)")
                            fastag_ocr_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            fastag_ocr_decision_out = gr.Markdown()
                            fastag_ocr_json_out = gr.JSON(label="Full result")
                    fastag_ocr_run_btn.click(
                        run_fastag_ocr_individual,
                        inputs=[fastag_ocr_image_in, fastag_dd, fastag_gm_dd, fastag_ocr_barcode_in],
                        outputs=[fastag_ocr_decision_out, fastag_ocr_json_out],
                    )

                # --- End-to-end ----------------------------------------------------------
                with gr.Tab("End-to-end"):
                    with gr.Row():
                        with gr.Column():
                            fastag_image_in = gr.Image(type="pil", label="Upload FASTag photo")
                            fastag_id_in = gr.Textbox(label="Claimed FASTag id")
                            fastag_bank_in = gr.Textbox(label="Claimed bank code (optional)")
                            fastag_vrn_in = gr.Textbox(
                                label="Truck number / VRN (optional — fill in to also run the "
                                      "duplicate check against the 'fastag' reference library; "
                                      "leave blank to skip it)")
                            fastag_upload_id_in = gr.Textbox(
                                label="Upload id (optional — auto-generated if left blank; only "
                                      "needed if you want this stored under a specific id)")
                            fastag_run_btn = gr.Button("Run", variant="primary")
                        with gr.Column():
                            fastag_decision_out = gr.Markdown()
                            fastag_json_out = gr.JSON(label="Full result")
                    fastag_run_btn.click(
                        run_fastag_individual,
                        inputs=[fastag_image_in, fastag_id_in, fastag_bank_in, fastag_dd, fastag_gm_dd,
                               fastag_vrn_in, fastag_upload_id_in],
                        outputs=[fastag_decision_out, fastag_json_out],
                    )

        # --- Reference-image library --------------------------------------------
        with gr.Tab("Reference Images"):
            gr.Markdown(
                "**Seeding only — no check happens here.** Vectorizes an uploaded image and "
                "stores it in the reference corpus (`duplicate_check.py`) — e.g. importing a "
                "legacy dump of photos from before this vector-DB setup existed. The actual "
                "duplicate CHECK against this corpus happens later, as part of testing a real "
                "front/side/FASTag upload (the optional VRN field on Q1/FASTag's end-to-end, or "
                "the **Duplicate check** bucket under Side/Axle Image). `front` / `side` / "
                "`fastag` are separate corpora and never compared against each other. Needs "
                "`VFIV_PGVECTOR_DSN` set to a reachable Postgres+pgvector instance."
            )
            dup_image_type_dd = gr.Dropdown(config.IMAGE_TYPES, value="front", label="Image type")
            dup_stats_out = gr.Markdown()
            dup_refresh_btn = gr.Button("Refresh library stats")
            dup_refresh_btn.click(refresh_library_stats, inputs=[], outputs=[dup_stats_out])

            with gr.Row():
                with gr.Column():
                    dup_image_in = gr.Image(type="pil", label="Upload image")
                    dup_vrn_in = gr.Textbox(label="Truck number (VRN)")
                    dup_upload_id_in = gr.Textbox(
                        label="Upload id (optional — auto-generated if left blank; only needed if "
                              "you want this stored under a specific id, e.g. a real legacy DB row id)")
                    dup_store_btn = gr.Button("Store", variant="primary")
                with gr.Column():
                    dup_decision_out = gr.Markdown()
                    dup_json_out = gr.JSON(label="Full result")
            dup_store_btn.click(
                run_reference_store,
                inputs=[dup_image_in, dup_vrn_in, dup_image_type_dd, dup_upload_id_in],
                outputs=[dup_decision_out, dup_json_out],
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CUSTOM_CSS)
