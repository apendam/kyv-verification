# batch_pipeline

A standalone batch runner for the KYV front-image gate sequence: give it a
list of images (a CSV or Excel manifest) with the claim attached to each
one (VRN, make, vehicle type), and it runs every image through the same
front image -> VRN -> maker -> duplicate check sequence used elsewhere in
this repo, one by one, and writes the results to an Excel file — one row
per image.

**This is a self-contained copy**, split out from `openrouter_checks/` in
this same repo so it's easy to hand to someone without the rest of that
package (no Gradio webapp, no model-catalog tooling — just the checks and
the batch runner). The core check logic (`kyv_batch/`) is a duplicate of
`openrouter_checks/openrouter_checks/`, not a shared import — a fix or
improvement made in one place does **not** automatically apply to the
other. If you're maintaining both, port changes across by hand, or treat
`openrouter_checks/` as the source of truth and re-copy from there.

## What it checks, per image

1. **Front image check** — reads the vehicle type from the image
   independently (never told what was claimed), and rejects outright if it
   doesn't match the claimed `vehicle_type` (or isn't a bus/truck at all).
   Also flags (manual review) if the image looks altered/AI-generated.
2. **VRN check** — reads the plate; a confusable-character-aware fuzzy
   match against the claimed VRN decides match / manual review / reject.
3. **Maker check** — only runs after a VRN match; a mismatch is manual
   review, never a reject on its own.
4. **Duplicate check** — pHash first (free, instant); falls back to a
   local SigLIP vector embedding only if pHash comes back clean. Both
   signals ignore the plate area (blacked out before comparison) and both
   report *every* matching reference within threshold, not just the
   closest one.

## Setup

```bash
cd batch_pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../truck_extract_match   # pure-logic VRN/maker matching, from this repo's root

cp .env.example .env
# edit .env, set OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

Build a manifest (CSV or `.xlsx`) — see `sample_manifest.csv`:

| column | required? | meaning |
|---|---|---|
| `image_path` | yes | Path to the image. Relative paths resolve against the manifest's own directory. |
| `vrn` | yes | Claimed vehicle registration number. |
| `make` | yes | Claimed manufacturer/make. |
| `vehicle_type` | yes | Claimed type: `bus` or `truck`. |
| `upload_id` | no | Auto-generated from the VRN if left blank. |
| `vision_model` | no | Per-row model override — see below. |

Then run:

```bash
python scripts/batch_check.py --manifest sample_manifest.csv --output results.xlsx
```

**Duplicate check needs a seeded corpus first** — without one, every image
reports `duplicate_signal_used = none` (nothing to compare against):

```bash
python scripts/seed_reference.py --image samples/known_good_front.jpg \
    --image-type front --upload-id ref_001 --vrn MH12AB1234
```

### Choosing a model

Pick any vision-capable model from <https://openrouter.ai/models>:

- `--model <id>` sets the default for every row in the batch.
- A row's own `vision_model` column overrides `--model` for just that row
  — so one batch can mix models if you want to compare them.
- Every output row records which model actually ran it, in its own
  `vision_model` column.

## Output columns

One row per image, in this order — the claim, then the script's findings
in the same sequence the checks run, then an explicit VRN comparison, then
cost/latency/token totals:

`image_path, upload_id, claimed_vrn, claimed_make, claimed_vehicle_type,
vision_model, detected_vehicle_type, vehicle_type_match,
is_altered_or_ai_generated, front_image_reasoning, vrn_plate_readable,
vrn_detected_text, vrn_fuzzy_outcome, vrn_edit_distance, vrn_exact_match,
maker_readable, maker_detected_text, maker_outcome, duplicate_signal_used,
is_duplicate, duplicate_match_upload_ids, duplicate_match_scores,
duplicate_match_vrns, final_decision, final_reason,
latency_ms_model_checks, latency_ms_total, prompt_tokens,
completion_tokens, cost_usd`

Notes:
- `vrn_exact_match` is a literal claimed-vs-detected string compare —
  separate from `vrn_fuzzy_outcome`, which is the confusable-character-aware
  match the actual pass/reject decision is based on.
- `duplicate_match_*` columns are **comma-separated** — a query image can be
  a near-duplicate of more than one reference image, and every one within
  threshold is listed, not just the closest.
- If a gate rejects or sends to manual review early (e.g. a vehicle-type
  mismatch), every column for a check that never ran is blank.
- `latency_ms_model_checks` is the OpenRouter model calls only (front + VRN
  + maker); `latency_ms_total` adds the local duplicate-check's own compute
  time (real wall-clock time even though it's free — SigLIP especially).

Re-running the same `upload_id` is a no-op (nothing charged) unless you
pass `--force`.

## What this does not cover

Same caveats as `openrouter_checks/` — see that package's own README for
the full list (structured-output support varies by model, no image
downscaling, no concurrency/batch parallelism, duplicate-check thresholds
are uncalibrated starting points). This folder is a leaner packaging of
the same checks, not a different implementation of them.
