# openrouter_checks

OpenRouter-backed implementation of the **KYV Gate Sequence** flowchart
(front image → vehicle type → VRN → maker → duplicate → approve), plus a
standalone reference-image repository for the duplicate check — built fresh
rather than reusing the SigLIP+pgvector system already in
`vehicle_front_image_validator/`, so there's no database server to stand up.
Model choice is a flag everywhere, not a constant, so you can swap between
providers per run.

The duplicate check runs over two **local, free** signals, cheapest first —
not an OpenRouter embedding, which didn't cluster near a clean threshold
even for two photos of the same truck differing only in an edited plate:

1. **pHash** (perceptual hash) — instant, no model to load, catches a
   near-pixel-identical reupload (same photo, maybe recompressed/lightly
   edited).
2. **SigLIP** (local vector embedding, same weights `vehicle_front_image_
   validator/` already uses for its own duplicate check) — only runs when
   pHash comes back clean, since it needs a model loaded via
   `transformers`/`torch`. Catches what pHash misses: a re-crop, a
   lighting/angle change.

See `duplicate.py` / `imaging.py` / `siglip.py`.

Two scripts, plus a Gradio front end:

- **`scripts/check_image.py`** — runs one upload through the full gate
  sequence, logging every model call's tokens/cost/verdict to SQLite.
- **`scripts/seed_reference.py`** — adds a known-good image (front / fastag /
  side) to the duplicate-check repository, tagged with a unique upload ID.
- **`openrouter_checks/webapp.py`** — a Gradio UI over both of the above,
  matching `vehicle_front_image_validator`'s own dark-theme test UI. Three
  tabs: **Check Image** (upload + run the gate sequence, see each step's
  verdict and the running token/cost total), **Seed Reference** (upload a
  known-good image into the duplicate-check corpus), **Model Catalog**
  (`models.json`, plus a "verify against OpenRouter" lookup for a typed-in
  model id before you use it). Run with:

  ```bash
  python -m openrouter_checks.webapp
  ```

Both scripts and the webapp write to the same SQLite file
(`kyv_checks.sqlite3` by default) — one file, no hardware deployment.

## Model catalog (`models.json`)

The webapp's "Vision model" dropdowns (Check Image, Seed Reference — both
use one to read/locate the plate) are populated from `models.json` (repo
root of this package) — a plain list you maintain by hand:

```json
{
  "models": [
    {"id": "anthropic/claude-sonnet-5", "label": "Claude Sonnet 5", "vision": true, "embedding": false}
  ]
}
```

Edit it directly to add/remove entries, or use the **Model Catalog** tab's
"Verify against OpenRouter" box, which checks a typed-in id against
OpenRouter's live catalog and shows its real pricing and input/output
modalities before you tick vision/embedding and click "Add to catalog". The
`embedding` flag/column is there for your own reference (OpenRouter also
serves a separate `GET /embeddings/models` catalog for it) — nothing in this
app currently consumes an embedding model, since the duplicate check is a
local perceptual hash instead (see above).

## Setup

Needs **Python 3.11+** (`truck_extract_match` requires it, and it also keeps
you on a `gradio` release recent enough to resolve cleanly against today's
`fastapi`/`pydantic` — see the comment in `requirements.txt` if you're stuck
on an older Python and the webapp fails to start).

```bash
cd openrouter_checks
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../truck_extract_match   # pure-logic VRN/maker matching, reused not re-derived

cp .env.example .env
# edit .env, set OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

```bash
# Seed the duplicate-check corpus with known-good images:
python scripts/seed_reference.py --image samples/truck1_front.jpg \
    --image-type front --upload-id ref_001 --vrn MH12AB1234
python scripts/seed_reference.py --image samples/truck1_fastag.jpg \
    --image-type fastag --upload-id ref_001 --vrn MH12AB1234

# Check a new upload against the full gate sequence:
python scripts/check_image.py --image samples/new_upload.jpg \
    --vrn MH12AB1234 --make "TATA MOTORS LTD" --upload-id upload_042

# Same image, different model, nothing else changes:
python scripts/check_image.py --image samples/new_upload.jpg \
    --vrn MH12AB1234 --make "TATA MOTORS LTD" --upload-id upload_043 \
    --model google/gemini-2.5-flash

# What has this cost so far?
python scripts/show_costs.py
python scripts/show_costs.py --by check_name
```

Pick any vision-capable model from <https://openrouter.ai/models> for
`--model` (or `seed_reference.py`'s `--vision-model`, used there to locate
the plate for masking, same as the Check Image path's own VRN check).
Defaults live in `openrouter_checks/config.py`.

## What each script actually does

`check_image.py` calls `gate_sequence.run_gate_sequence()`, which walks the
flowchart node by node:

1. **Front image check** — one vision call asks both "is this a bus/truck?"
   and "is this altered/AI-generated?" at once (mirroring the flowchart's
   "one CV+Claude pass" framing) → manual review on either bad answer.
2. **VRN check** — reads the plate, then matches it against the claimed VRN
   using `truck_extract_match`'s confusion-aware edit distance (0/O, 1/I,
   5/S, …). Unreadable → manual review. A small residual distance (probably
   a smudge) → manual review. A large one (a different plate) → **reject**.
3. **Maker check** — only runs after a confirmed VRN match. Unreadable or a
   match both proceed toward approval; a mismatch → manual review (it can
   never reject on its own).
4. **Duplicate check** — only runs on the path that would otherwise approve.
   Blacks out the plate (using the bounding box the VRN check already read —
   no extra model call) before either signal below ever sees the image, so
   reusing the same photo under a different claimed VRN by only editing the
   plate can't dodge this check:
   - **pHash** compares Hamming distance against every reference image of
     the same type. A duplicate needs the closest match's distance ≤
     `DUPLICATE_HAMMING_MAX` (default `10`, out of 64 bits — lower means
     more similar) **and** to have been filed under a *different* claimed
     VRN — an honest re-upload under the same VRN is never flagged.
   - If pHash comes back clean, **SigLIP** (a local vector embedding, only
     loaded at this point) runs the same same-type/different-VRN check
     using cosine similarity against `DUPLICATE_SIGLIP_SIMILARITY_MIN`
     (default `0.97`) instead of a Hamming distance — this is what catches
     a duplicate pHash misses (re-crop, lighting/angle change). Reference
     images seeded before this signal existed have no vector stored yet
     and are skipped until re-seeded; pHash still covers them.

Every step is logged to the `checks` table regardless of outcome, including a
`technical_failure` flag (API error, malformed response, exhausted retries) —
that flag is what turns into "Manual Review" at every stage, same as the
dashed gates in the flowchart artifact. A run that hits `402` (out of
OpenRouter credit) stops immediately rather than logging a wave of technical
failures — that's an account problem, not a per-image one.

Re-running the same `--upload-id` is a no-op (nothing charged) unless you
pass `--force` — see `db.already_checked`.

## What this does *not* cover yet

- **Side/corner image and FASTag checks** — only the front-image gate
  sequence is implemented. The same pattern (schema + prompt + a
  `run_*_sequence` function) extends to those; ask if you want them built out
  next.
- **Untested against a live key** — this was written and reviewed against
  OpenRouter's documented request/response shapes, but not run end-to-end
  against a real account in this environment. Smoke-test on one image before
  trusting it on a batch — if a response shape doesn't match what's expected,
  `OpenRouterBadResponse` will say so rather than silently misparsing.
- **Structured-output support varies by model.** Not every model behind
  OpenRouter honours `response_format: json_schema` — if you point `--model`
  at one that doesn't, expect `OpenRouterBadResponse` rather than a
  best-effort text parse. Check a model's `supported_parameters` via
  `GET /api/v1/models` before relying on it here.
- **No image downscaling yet.** Large phone photos are sent as-is; most
  vision APIs price by image tokens/tiling, so this is the most direct lever
  if cost turns out to matter — not implemented here on purpose, since the
  right max dimension differs per check (OCR-sensitive checks like the plate
  read want to stay higher-res than the tamper/vehicle-type check does).
- **No concurrency/batch runner.** Each script processes one image per
  invocation; wrap it in a shell loop or extend it if you need to run a
  folder overnight — worth adding real rate-limiting before you do, rather
  than firing requests as fast as the loop allows.
- **`DUPLICATE_HAMMING_MAX` (default `10`) and `DUPLICATE_SIGLIP_SIMILARITY_MIN`
  (default `0.97`) are starting points, not calibrated values.** Test both
  against your own labeled pairs (known duplicates vs. known-different
  vehicles) and adjust in `openrouter_checks/config.py`.
- **Linear-scan Hamming-distance search.** Fine up to tens of thousands of
  reference images; past that, an indexed nearest-neighbor structure would
  be the next step.
