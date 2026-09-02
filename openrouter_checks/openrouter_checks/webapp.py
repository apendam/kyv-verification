"""Front end for the OpenRouter-backed KYV gate sequence — Gradio, matching
``vehicle_front_image_validator/src/vfiv/webapp.py``'s own choice of tool and
its exact BlackBuck dark theme (same CSS tokens/rules, copied rather than
re-derived, so this doesn't visually drift from the sibling package's test UI).

Three tabs, per the approved mockup:
  - **Check Image**  — run one upload through the full gate sequence.
  - **Seed Reference** — add a known-good image to the duplicate-check corpus.
  - **Model Catalog**  — the models.json you maintain by hand, plus a
    "verify against OpenRouter" lookup for typed-in model ids.

Run with:

    python -m openrouter_checks.webapp
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before `from . import config` reads OPENROUTER_API_KEY

import gradio as gr
from PIL import Image

from . import config, db, models
from .client import OpenRouterClient, OpenRouterInsufficientCredits
from .gate_sequence import run_gate_sequence

# Gradio uploads land in an ephemeral temp dir (cleared on restart, maybe
# sooner) — fine for a Check Image run (used once, then discarded), but a
# seeded reference image needs to still be viewable later, so it's copied
# here on seed and this path (not the temp one) is what gets stored in the
# DB and shown in the gallery.
REFERENCE_STORE_DIR = Path(__file__).resolve().parent.parent / "reference_images"
REFERENCE_STORE_DIR.mkdir(exist_ok=True)

# --- theme: light, macOS-native (System Settings / Finder style) -------------
# No webfont import -- -apple-system/BlinkMacSystemFont resolve to the real
# SF Pro on macOS/Safari/Chrome, which is a better "feels native" win than
# any web font could be, and drops an external network dependency.

_BB_CRIMSON = "#d71e48"
_BB_CRIMSON_DARK = "#a8152f"
_BB_YELLOW = "#ffc130"

_DECISION_STYLE = {
    "APPROVED": (_BB_CRIMSON, "#fff"),
    "REJECT": (_BB_CRIMSON_DARK, "#fff"),
    "MANUAL_REVIEW": (_BB_YELLOW, "#000"),
}

TITLEBAR_HTML = """
<div style="display:flex; align-items:center; gap:8px; padding:10px 16px;
            background:#e8e8ed; border-radius:10px 10px 0 0; margin:-1px -1px 0;
            border-bottom:1px solid #d2d2d7;">
  <span style="width:12px;height:12px;border-radius:50%;background:#ff5f57;display:inline-block;"></span>
  <span style="width:12px;height:12px;border-radius:50%;background:#febc2e;display:inline-block;"></span>
  <span style="width:12px;height:12px;border-radius:50%;background:#28c840;display:inline-block;"></span>
</div>
"""

CUSTOM_CSS = """
:root {
    --bb-black: #f5f5f7;
    --bb-panel: #ffffff;
    --bb-panel-2: #f5f5f7;
    --bb-white: #1d1d1f;
    --bb-muted: #6e6e73;
    --bb-turquoise: #d71e48;
    --bb-turquoise-dark: #a8152f;
    --bb-border: #d2d2d7;
    --mac-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

.gradio-container {
    background: var(--bb-black) !important;
    color: var(--bb-white) !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
    color: var(--bb-white) !important;
}

.gradio-container .prose, .gradio-container p, .gradio-container span, .gradio-container label {
    color: var(--bb-white) !important;
}

.gradio-container .prose strong {
    color: var(--bb-white) !important;
}
.gradio-container .prose code {
    color: var(--bb-turquoise) !important;
    background: var(--bb-panel-2) !important;
    border-radius: 4px !important;
}

.gradio-container .block, .gradio-container .form {
    background: var(--bb-panel) !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--mac-shadow) !important;
}

.gradio-container label.float, .gradio-container .float,
.gradio-container [data-testid="block-label"] {
    background: transparent !important;
    color: var(--bb-muted) !important;
}

/* Top tab row restyled as a macOS toolbar segmented control rather than an
   underlined web tab bar. */
.gradio-container .tab-container[role="tablist"] {
    background: #e8e8ed !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 0 !important;
    display: inline-flex !important;
}
.gradio-container button[role="tab"] {
    color: var(--bb-muted) !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
    margin: 0 !important;
}
.gradio-container button[role="tab"].selected {
    color: var(--bb-white) !important;
    background: #ffffff !important;
    box-shadow: var(--mac-shadow) !important;
}
/* Gradio paints the selected-tab indicator as a blue ::after bar (not a
   border), which the .selected rule above never reaches — the white chip
   background already shows which tab is active, so just hide it. */
.gradio-container button[role="tab"].selected::after {
    background: transparent !important;
}

.gradio-container input:not([type=radio]):not([type=checkbox]),
.gradio-container textarea, .gradio-container select {
    background: #ffffff !important;
    color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 8px !important;
}
.gradio-container input:not([type=radio]):not([type=checkbox]):focus,
.gradio-container textarea:focus {
    border-color: var(--bb-turquoise) !important;
    outline: 2px solid color-mix(in srgb, var(--bb-turquoise) 25%, transparent) !important;
    outline-offset: 0 !important;
}

/* Radio group restyled as a macOS segmented control (one pill track, the
   selected option gets a white "chip" with a soft shadow) instead of the
   default row of separately-bordered pills. */
.gradio-container fieldset .wrap {
    background: #e8e8ed !important;
    border-radius: 8px !important;
    padding: 3px !important;
    display: inline-flex !important;
    gap: 0 !important;
}
.gradio-container fieldset label {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
    color: var(--bb-muted) !important;
}
.gradio-container fieldset label.selected {
    background: #ffffff !important;
    color: var(--bb-white) !important;
    box-shadow: var(--mac-shadow) !important;
}
.gradio-container input[type=radio], .gradio-container input[type=checkbox] {
    accent-color: var(--bb-turquoise) !important;
}
/* accent-color alone isn't enough — Gradio's own :checked rule sets
   background-color directly (its default blue) with equal or higher
   specificity, so the painted dot stayed blue regardless of accent-color. */
.gradio-container input[type=radio]:checked, .gradio-container input[type=checkbox]:checked {
    background-color: var(--bb-turquoise) !important;
    border-color: var(--bb-turquoise) !important;
}

.gradio-container ul.options {
    background: #ffffff !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 8px !important;
    box-shadow: var(--mac-shadow) !important;
}
.gradio-container ul.options li.item {
    color: var(--bb-white) !important;
    background: #ffffff !important;
}
.gradio-container ul.options li.item.selected,
.gradio-container ul.options li.item.active {
    background: #f5f5f7 !important;
    color: var(--bb-white) !important;
}
.gradio-container ul.options li.item:hover {
    background: var(--bb-turquoise) !important;
    color: #fff !important;
}

.gradio-container button.primary {
    background: var(--bb-turquoise) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    box-shadow: var(--mac-shadow) !important;
}
.gradio-container button.primary:hover {
    background: var(--bb-turquoise-dark) !important;
    color: #fff !important;
}

.gradio-container table, .gradio-container th, .gradio-container td {
    background: var(--bb-panel) !important;
    color: var(--bb-white) !important;
    border-color: var(--bb-border) !important;
}

/* Modern Gradio Dataframe renders its scrollable body as a virtualized stack
   of plain divs (.table-wrap > .virtual-table-viewport > .virtual-row >
   .body-cell), not <tbody>/<tr>/<td> — the rule above never reaches it, so it
   falls back to the browser's default white background (white-on-white). */
.gradio-container .table-wrap, .gradio-container .virtual-table-viewport,
.gradio-container .virtual-body, .gradio-container .virtual-row,
.gradio-container .body-cell, .gradio-container .header-table,
.gradio-container .header-cell {
    background: var(--bb-panel) !important;
    color: var(--bb-white) !important;
    border-color: var(--bb-border) !important;
}
.gradio-container .virtual-row.row-odd {
    background: var(--bb-panel-2) !important;
}

.gradio-container button.secondary {
    background: #ffffff !important;
    color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important;
    border-radius: 7px !important;
    box-shadow: var(--mac-shadow) !important;
}
"""


def _banner(decision: str, reason: str) -> str:
    bg, fg = _DECISION_STYLE.get(decision, ("#333", "#fff"))
    badge = (f'<span style="background:{bg};color:{fg};padding:0.35em 0.9em;'
             f'border-radius:999px;font-weight:700;letter-spacing:0.02em;">{decision}</span>')
    return f"### {badge}\n\n{reason}"


def _describe_upload(file_path: str | None) -> tuple[Image.Image | None, str]:
    """Given the filepath Gradio hands back from a file/image upload, returns
    (preview image, a 'name.jpg · 214 KB' label) — the filename-and-preview
    the mockup asked for.
    """
    if not file_path:
        return None, ""
    path = Path(file_path)
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = None
    size_kb = path.stat().st_size / 1024
    return image, f"**{path.name}** · {size_kb:.0f} KB"


def _format_steps(steps: list[dict]) -> str:
    if not steps:
        return "_No steps ran._"
    lines = []
    for i, step in enumerate(steps, start=1):
        check = step.get("check", "?")
        outcome = step.get("outcome", step.get("verdict", ""))
        detail = ", ".join(
            f"{k}={v!r}" for k, v in step.items()
            if k not in {"check", "reasoning"} and v not in (None, "")
        )
        lines.append(f"**{i}. {check}** — `{outcome}`  \n<span style='color:var(--bb-muted);font-size:13px'>{detail}</span>")
    return "\n\n".join(lines)


# --- Check Image tab ----------------------------------------------------------

def on_check_image_upload(file_path):
    image, label = _describe_upload(file_path)
    return image, label


def run_check_image(image_file, vrn, make, upload_id, vision_model, embed_model):
    if not image_file:
        return "### Upload an image first.", "", ""
    if not vrn or not make:
        return "### VRN and make are both required.", "", ""
    upload_id = (upload_id or "").strip() or f"check-{vrn.strip()}-{int(time.time())}"

    conn = db.connect(config.DEFAULT_DB_PATH)
    try:
        client = OpenRouterClient()
    except Exception as exc:
        return f"### Client error\n\n{exc}", "", ""

    try:
        result = run_gate_sequence(
            conn, client, image_path=image_file, claimed_vrn=vrn.strip(),
            claimed_make=make.strip(), upload_id=upload_id,
            vision_model=vision_model, embed_model=embed_model,
        )
    except OpenRouterInsufficientCredits as exc:
        return f"### Stopped — out of OpenRouter credits\n\n{exc}", "", ""
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return f"### Error\n\n{exc}", "", ""

    t = client.totals
    summary = (f"upload_id: `{upload_id}` &middot; {t.calls} call(s) &middot; "
               f"{t.prompt_tokens + t.completion_tokens} tokens &middot; ${t.cost_usd:.5f}")
    return _banner(result.decision, result.reason), _format_steps(result.steps), summary


# --- Seed Reference tab --------------------------------------------------------

def on_seed_upload(file_path):
    image, label = _describe_upload(file_path)
    return image, label


def refresh_repo_stats() -> str:
    conn = db.connect(config.DEFAULT_DB_PATH)
    stats = db.reference_stats(conn)
    if not stats:
        return "_Repository is empty — nothing seeded yet._"
    return "\n".join(f"- **{t}**: {n} image(s)" for t, n in sorted(stats.items()))


def _persist_reference_image(image_file: str, upload_id: str, image_type: str) -> str:
    """Copies the uploaded file out of Gradio's ephemeral temp dir into
    REFERENCE_STORE_DIR under a stable name, so it's still there to view
    later — Gradio's own temp files aren't guaranteed to survive a restart.
    """
    src = Path(image_file)
    dest = REFERENCE_STORE_DIR / f"{upload_id}__{image_type}{src.suffix or '.jpg'}"
    shutil.copyfile(src, dest)
    return str(dest)


def refresh_gallery(image_type: str) -> list[tuple[str, str]]:
    """Thumbnails of every stored reference image of one type, captioned
    with its upload ID (+ claimed VRN if set) — lets you view the individual
    images behind a repository-stats count, not just the total.
    """
    conn = db.connect(config.DEFAULT_DB_PATH)
    rows = db.list_reference_images(conn, image_type)
    gallery = []
    for r in rows:
        if not Path(r["image_path"]).is_file():
            continue  # a pre-persistence row whose temp file is long gone
        caption = r["upload_id"] + (f" · {r['claimed_vrn']}" if r["claimed_vrn"] else "")
        gallery.append((r["image_path"], caption))
    return gallery


def run_seed(image_file, image_type, upload_id, vrn, embed_model):
    stats = refresh_repo_stats()
    gallery = refresh_gallery(image_type)
    if not image_file:
        return "### Upload an image first.", stats, gallery

    vrn = (vrn or "").strip() or None
    upload_id = (upload_id or "").strip()
    if not upload_id:
        # Same fallback vfiv/webapp.py uses: one reference per VRN+type by
        # default (re-seeding the same VRN's front image replaces it, rather
        # than piling up duplicates), falling back to a timestamp if even
        # the VRN was left blank.
        upload_id = f"ref-{vrn}-{image_type}" if vrn else f"ref-{image_type}-{int(time.time())}"

    conn = db.connect(config.DEFAULT_DB_PATH)
    try:
        client = OpenRouterClient()
        result = client.embed(model=embed_model, image_path=image_file)
    except OpenRouterInsufficientCredits as exc:
        return f"### Stopped — out of OpenRouter credits\n\n{exc}", stats, gallery
    except Exception as exc:  # noqa: BLE001
        return f"### Error\n\n{exc}", stats, gallery

    stored_path = _persist_reference_image(image_file, upload_id, image_type)
    db.insert_reference_image(
        conn, upload_id=upload_id, image_type=image_type, image_path=stored_path,
        claimed_vrn=vrn, embedding=result.vector, embed_model=result.model,
    )
    msg = (f"### Added\n\n`{upload_id}` &middot; **{image_type}** &middot; "
           f"${result.cost_usd:.5f} &middot; {result.prompt_tokens} tokens")
    return msg, refresh_repo_stats(), refresh_gallery(image_type)


# --- Model Catalog tab ----------------------------------------------------------

def _catalog_table():
    rows = models.load_catalog()
    return [[m["id"], m.get("label", m["id"]),
             "✓" if m.get("vision") else "—",
             "✓" if m.get("embedding") else "—"] for m in rows]


def do_verify_model(model_id: str):
    if not model_id or not model_id.strip():
        return "Enter a model id first (e.g. `mistralai/pixtral-large-2411`).", False, False
    model_id = model_id.strip()
    try:
        all_models = models.list_all_models()
        info = models.fetch_model_info(model_id, all_models)
    except Exception as exc:  # noqa: BLE001
        return f"Lookup against OpenRouter failed: {exc}", False, False
    if info is None:
        suggestions = models.find_similar_models(model_id, all_models)
        msg = (f"`{model_id}` was not found — OpenRouter ids are exact and case-sensitive "
               f"(`provider/lowercase-slug`), not the bold display name shown on the site.")
        if suggestions:
            lines = "\n".join(f"- `{m['id']}` — {m.get('name', '')}" for m in suggestions)
            msg += f"\n\nDid you mean:\n\n{lines}"
        else:
            msg += " No close match found either — double check the id at openrouter.ai/models."
        return msg, False, False

    arch = info.get("architecture", {})
    in_modalities = arch.get("input_modalities", [])
    out_modalities = arch.get("output_modalities", [])
    is_vision = "image" in in_modalities and "embeddings" not in out_modalities
    is_embedding = "embeddings" in out_modalities
    pricing = info.get("pricing", {})
    warn = ""
    if is_embedding and "image" not in in_modalities:
        warn = (" **Note:** this embedding model only takes text input — it will reject the "
                 "images this app sends for the duplicate check and seeding.")
    msg = (f"Found **{info.get('name', model_id)}** (`{model_id}`) &middot; "
           f"input: {in_modalities}, output: {out_modalities} &middot; "
           f"prompt ${pricing.get('prompt', '?')}/tok, completion ${pricing.get('completion', '?')}/tok."
           f"{warn} Check the box(es) below and add it to the catalog if it's what you want.")
    return msg, is_vision, is_embedding


def do_add_to_catalog(model_id: str, label: str, is_vision: bool, is_embedding: bool):
    if not model_id or not model_id.strip():
        return "Verify a model id first.", _catalog_table()
    models.add_to_catalog(model_id.strip(), label=label.strip() or None,
                          vision=bool(is_vision), embedding=bool(is_embedding))
    return f"Added `{model_id.strip()}` to `models.json`.", _catalog_table()


# --- UI --------------------------------------------------------------------------

with gr.Blocks(title="KYV · OpenRouter Checks", theme=gr.themes.Base(), css=CUSTOM_CSS) as demo:
    gr.HTML(TITLEBAR_HTML)
    gr.Markdown(
        "# KYV · OpenRouter Checks\n"
        "Front-image gate sequence, run against OpenRouter — model swappable per run. "
        "Vision/embedding model choices come from `models.json`; verify a new model id "
        "against OpenRouter on the **Model Catalog** tab before using it here."
    )

    with gr.Tabs():
        # --- Check Image ---------------------------------------------------
        with gr.Tab("Check Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    ci_file_in = gr.File(label="Upload image", file_types=["image"], type="filepath")
                    ci_preview = gr.Image(label="Preview", interactive=False)
                    ci_filename = gr.Markdown()
                    ci_vrn_in = gr.Textbox(label="Claimed VRN")
                    ci_make_in = gr.Textbox(label="Claimed make")
                    ci_upload_id_in = gr.Textbox(
                        label="Upload ID (optional — auto-generated from the VRN if left blank)")
                    with gr.Row():
                        ci_vision_dd = gr.Dropdown(models.vision_models(), value=config.DEFAULT_VISION_MODEL,
                                                   allow_custom_value=True, label="Vision model")
                        ci_embed_dd = gr.Dropdown(models.embed_models(), value=config.DEFAULT_EMBED_MODEL,
                                                  allow_custom_value=True, label="Embedding model")
                    ci_run_btn = gr.Button("Run gate sequence", variant="primary")
                with gr.Column(scale=1):
                    ci_banner_out = gr.Markdown()
                    ci_steps_out = gr.Markdown(label="Steps")
                    ci_summary_out = gr.Markdown()

            ci_file_in.change(on_check_image_upload, inputs=[ci_file_in], outputs=[ci_preview, ci_filename])
            ci_run_btn.click(
                run_check_image,
                inputs=[ci_file_in, ci_vrn_in, ci_make_in, ci_upload_id_in, ci_vision_dd, ci_embed_dd],
                outputs=[ci_banner_out, ci_steps_out, ci_summary_out],
            )

        # --- Seed Reference --------------------------------------------------
        with gr.Tab("Seed Reference"):
            gr.Markdown("Add a known-good image to the duplicate-check repository.")
            with gr.Row():
                with gr.Column(scale=1):
                    sr_type_in = gr.Radio(["front", "fastag", "side"], value="front", label="Image type")
                    sr_file_in = gr.File(label="Reference image", file_types=["image"], type="filepath")
                    sr_preview = gr.Image(label="Preview", interactive=False)
                    sr_filename = gr.Markdown()
                    sr_upload_id_in = gr.Textbox(
                        label="Upload ID (optional — auto-generated from VRN + image type if left blank)")
                    sr_vrn_in = gr.Textbox(label="Claimed VRN")
                    sr_embed_dd = gr.Dropdown(models.embed_models(), value=config.DEFAULT_EMBED_MODEL,
                                              allow_custom_value=True, label="Embedding model")
                    sr_run_btn = gr.Button("Add to repository", variant="primary")
                with gr.Column(scale=1):
                    sr_result_out = gr.Markdown()
                    gr.Markdown("**Repository stats**")
                    sr_stats_out = gr.Markdown(value=refresh_repo_stats)
                    gr.Markdown("**Stored images — click one to view full size**")
                    sr_gallery_out = gr.Gallery(
                        value=lambda: refresh_gallery("front"), columns=3, height=320,
                        object_fit="cover", show_label=False,
                    )

            sr_file_in.change(on_seed_upload, inputs=[sr_file_in], outputs=[sr_preview, sr_filename])
            sr_type_in.change(refresh_gallery, inputs=[sr_type_in], outputs=[sr_gallery_out])
            sr_run_btn.click(
                run_seed,
                inputs=[sr_file_in, sr_type_in, sr_upload_id_in, sr_vrn_in, sr_embed_dd],
                outputs=[sr_result_out, sr_stats_out, sr_gallery_out],
            )

        # --- Model Catalog -----------------------------------------------------
        with gr.Tab("Model Catalog"):
            gr.Markdown(
                "Loaded from `models.json` — edit that file directly, or verify a model id "
                "against OpenRouter below and add it here."
            )
            mc_table_out = gr.Dataframe(
                headers=["id", "label", "vision", "embedding"],
                value=_catalog_table, label="Catalog", interactive=False,
            )
            mc_refresh_btn = gr.Button("Refresh", variant="secondary")
            mc_refresh_btn.click(_catalog_table, inputs=[], outputs=[mc_table_out])

            gr.Markdown("**Verify a model against OpenRouter**")
            with gr.Row():
                mc_id_in = gr.Textbox(label="Model id (provider/model-slug)",
                                      placeholder="mistralai/pixtral-large-2411")
                mc_label_in = gr.Textbox(label="Display label (optional)")
            mc_verify_btn = gr.Button("Verify against OpenRouter", variant="secondary")
            mc_verify_out = gr.Markdown()
            with gr.Row():
                mc_vision_cb = gr.Checkbox(label="Vision-capable")
                mc_embed_cb = gr.Checkbox(label="Embedding-capable")
            mc_add_btn = gr.Button("Add to catalog", variant="primary")

            mc_verify_btn.click(do_verify_model, inputs=[mc_id_in],
                                outputs=[mc_verify_out, mc_vision_cb, mc_embed_cb])
            mc_add_btn.click(do_add_to_catalog, inputs=[mc_id_in, mc_label_in, mc_vision_cb, mc_embed_cb],
                             outputs=[mc_verify_out, mc_table_out])


if __name__ == "__main__":
    demo.launch()
