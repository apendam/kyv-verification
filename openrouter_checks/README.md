# openrouter_checks

OpenRouter-backed implementation of the **KYV Gate Sequence** flowchart
(front image → vehicle type → VRN → maker → duplicate → approve), plus a
standalone, OpenRouter-embeddings-based reference-image repository for the
duplicate check — built fresh rather than reusing the SigLIP+pgvector system
already in `vehicle_front_image_validator/`, so there's no database server to
stand up. Model choice is a flag everywhere, not a constant, so you can swap
between providers per run.

Two scripts:

- **`scripts/check_image.py`** — runs one upload through the full gate
  sequence, logging every model call's tokens/cost/verdict to SQLite.
- **`scripts/seed_reference.py`** — adds a known-good image (front / fastag /
  side) to the duplicate-check repository, tagged with a unique upload ID.

Both write to the same SQLite file (`kyv_checks.sqlite3` by default) — one
file, no hardware deployment.

## Setup

```bash
cd openrouter_checks
python3 -m venv .venv && source .venv/bin/activate
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
`--model`; any embedding model for `--embed-model`. Defaults live in
`openrouter_checks/config.py`.

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
   Embeds the image, compares by cosine similarity against every reference
   image of the same type, and flags a duplicate only when the closest match
   is ≥ `DUPLICATE_SIMILARITY_MIN` (default `0.97`) **and** was filed under a
   *different* claimed VRN — an honest re-upload under the same VRN is never
   flagged.

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
- **The embedding model for duplicate detection** (`nvidia/llama-nemotron-embed-vl-1b-v2`
  by default) was confirmed in OpenRouter's docs as image-capable, but not
  confirmed live-available against the full model catalog — verify it
  resolves before seeding a real corpus with it; swap via `--embed-model` if not.
- **Linear-scan similarity search.** Fine up to tens of thousands of
  reference images; past that, this is exactly the job the existing
  pgvector setup is built for.
