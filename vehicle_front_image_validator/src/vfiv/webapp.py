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
from vfiv.experiments import q1_select, q2_select, q3_select
from vfiv.experiments.runner import run_q1_only, run_q2_only, run_q3_only, run_test_case

Q1_CHOICES = q1_select.Q1_BACKENDS
Q2_CHOICES = q2_select.Q2_BACKENDS
Q3_MAKE_CHOICES = q3_select.Q3_MAKE_BACKENDS
Q3_MODEL_CHOICES = q3_select.Q3_MODEL_BACKENDS

# Common Gemini 2.5 model ids — the dropdown also accepts a typed custom value (e.g. a
# future model id), so this list is a convenience, not an exhaustive whitelist.
GEMINI_MODEL_CHOICES = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

_DECISION_COLOUR = {"PASS": "🟢", "REJECT": "🔴", "MANUAL_REVIEW": "🟠"}


def _banner(decision: str, reason: str) -> str:
    emoji = _DECISION_COLOUR.get(decision, "")
    return f"## {emoji} {decision}\n\n{reason}"


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


if __name__ == "__main__":
    demo.launch()
