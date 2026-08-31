# Vehicle Image Validator

Image validation for an internal document/image-validation SaaS platform, designed to
be deployed independently in production. Given an uploaded image claimed to be one of a
truck/bus's **front**, **side/axle**, or **FASTag sticker** photos, decide whether it
genuinely is one — before any downstream document processing runs on it.

Each check calls whichever real model actually does that job best — Claude is used
only for the judgment calls no dedicated CV/OCR model can do (screenshot detection,
AI-generated detection). Everything else runs on real, purpose-built models, copied in
from the sibling `truck_verification_pipeline`/`truck_front_extractor` repos so this
module has no runtime dependency on them (only `truck_extract_match`'s pure-logic
matching code is installed as an actual dependency — see Run below).

| Check | Real model(s) | Status |
|---|---|---|
| **Q1 — Front image gate** | YOLOv8 (vehicle) + SigLIP 2 (pose) + Claude (screenshot/photo-of-photo/AI-generated) | ✅ |
| **Q2 — VRN + plate colour** | AWS Rekognition (text detection) + HSV (colour) — no Claude at all | ✅ |
| **Q3 — Make + model** | SigLIP 2 + Rekognition (make, two independent votes) + Claude (model) | ✅ |
| **Side/Axle image** | Claude/Gemini (axle count) + SigLIP (identity binding) + pgvector (duplicate check) | ✅ |
| **FASTag image** | pyzbar (QR/barcode decode) + Rekognition/Claude/Gemini (printed-digit OCR) | ✅ |
| **Test/inference interface** | Gradio UI — one upload at a time, backend-selectable per stage | ✅ |

## Origin — where each model came from

- **YOLOv8 vehicle detector** (`backends/vehicle.py`) — copied from
  `truck_front_extractor/src/tfe/backends/real.py`'s `_Vehicle`. No custom-trained
  weights exist in this environment, so it falls back to the generic COCO `yolov8n.pt`
  (auto-downloads). **Known limitation, confirmed in testing here**: this generic model
  misses some genuine Indian trucks (it's not fine-tuned on Indian truck body styles) —
  same caveat that project's own README documents. The CV gate is still authoritative
  (matches that project's architecture) — swap in a truck-finetuned detector in
  production. See `tests/test_front_image.py::test_generic_coco_yolo_misses_this_truck_known_limitation`.
- **SigLIP 2** (`backends/siglip.py`) — copied from the same file's `_SigLIP`/
  `_SigLIPPose`/`_SigLIPMake`. Weights (`google/siglip2-base-patch16-512`) were already
  cached locally from prior work. Used for: Q1's pose classification (front/front34/
  side/rear) and Q3's zero-shot make classification.
  **Known limitation, confirmed in testing here**: zero-shot make misclassified 2 of 3
  real Tata trucks as "Force Motors" — the same project's README already flags this
  ("zero-shot make is coarse; needs Qwen or a fine-tuned classifier"). This is why Q3
  backstops it with Rekognition (below).
- **AWS Rekognition** (`backends/rekognition.py`) — copied from
  `truck_verification_pipeline/step14_rekognition_detector.py`: real text detection +
  the full Indian-plate parsing logic (two-line handling, brand/slogan rejection, O/I
  fuzzy-correction). Replaces Claude entirely for Q2's plate reading, and its bonus
  `make` field (painted brand text, already extracted during the same call) backstops
  Q3's SigLIP make classifier.
- **HSV plate-colour classifier** (`backends/plate_colour.py`) — copied verbatim (pure
  numpy, no model weights) from the same `truck_front_extractor` file's `_HSVColour`.
- **Qwen2.5-VL** (`backends/qwen.py`) — copied from that file's `_QwenVLM`, real code,
  **not run live in this environment**: ~16GB of weights and documented as
  minutes/image on CPU (no GPU here). Wired and ready for a GPU box in production; Q3's
  model reading uses Claude until then.
- **`truck_extract_match`** (installed editable, see Run below) — VRN confusion-aware
  matching (`plate/format.py`) and make brand↔Parivahan-legal-entity canonicalisation
  (`make/aliases.py`), reused unchanged for the actual match decisions in Q2/Q3.

## What it checks

**Q1 — front image**:
- **vehicle_type / view / is_front / front_complete / confidence** — real CV pipeline
  (`backends/gate.py`, copied from `truck_front_extractor/src/tfe/gate/gate.py`): YOLO
  detects the vehicle, SigLIP classifies pose, a bbox-edge heuristic scores
  completeness, fused into one gate confidence. **Authoritative** — see the known YOLO
  limitation above.
- **is_screenshot** — not a screenshot / photo-of-a-screen / UI capture? (Claude)
- **is_photo_of_photo** — not a re-photographed printed photo/poster (glare, paper
  texture, print edges)? Genuine gallery photos of the real vehicle are fine either way. (Claude)
- **ai_generated** — synthetic/CGI/heavily AI-edited image? (Claude)

Decision (`PASS` / `REJECT` / `MANUAL_REVIEW`) is derived from ordered hard checks — see
`front_image.py:validate_front_image`. `checked=False` (→ `MANUAL_REVIEW`) if *either*
the CV gate or Claude fails — a screenshot verdict with no real vehicle classification
(or vice versa) isn't a usable result.

**Q2 — VRN + plate colour** (real Rekognition + HSV, `truck_extract_match`'s matcher —
no VLM call anywhere in this module):
- **plate** — the registration number, read by Rekognition's text detection + Indian-VRN
  regex parsing (handles two-line plates, strips slogans/brand text, corrects O/I
  misreads) — see `backends/rekognition.py`.
- **plate_colour** — white (private) / yellow (commercial) / green (EV) / black
  (self-drive rental) / red (temporary/govt), from a real HSV pixel classifier on the
  Rekognition-extracted plate crop.
- Extracted plate vs `claimed_vrn` via confusion-aware edit distance — a smudged
  `0`/`O`, `1`/`I`/`L`, etc. that's visually consistent with the claim still counts as a
  match (`inferred=true`), tolerating up to `VRN_MAX_CONFUSABLE_EDITS` (default 1)
  genuine edits.

Status (`MATCH` / `MISMATCH` / `UNREADABLE`, `truck_extract_match`'s own vocabulary) maps
to this platform's `PASS` / `REJECT` / `MANUAL_REVIEW` — see `vrn_check.py:validate_vrn`.
`checked=False` only on an AWS Rekognition credential failure — a genuinely-empty read
(no plate detected) is still `checked=True` (→ `UNREADABLE`/`MANUAL_REVIEW`), since that's
a real, meaningful result, not a tech failure.

**Where the "claimed" VRN comes from**: `claimed_vrn` is a required parameter on
`validate_vrn`/`validate_upload` — whatever value the caller sends alongside the image
when invoking this module. In production that's the real platform's upload interface
(once built); until then, pass it manually (`--vrn` on the CLI) for testing. Either
way, from this module's point of view it's the same plain string — where it came from
isn't this module's concern, and no DB lookup happens inside `vfiv` itself. Same for
`claimed_make`/`claimed_model` in Q3.

**Q3 — make + model** (real SigLIP + Rekognition for make, Claude for model):
- **make comparison is always enforced**, checked against **two independent real-model
  sources** — matched if EITHER agrees with the claimed make (`make_match_via` reports
  `"siglip"` / `"rekognition"` / `"both"`): SigLIP's zero-shot classifier (always runs)
  and Rekognition's painted-brand-text read (best-effort — `None` if no legible wordmark
  was detected). This backstop exists because SigLIP alone got 2 of 3 real Tata trucks
  wrong in testing — a single misclassifying model shouldn't false-reject a genuinely
  correct truck on a mandatory, always-enforced check. Make no longer depends on Claude
  at all, so a Claude outage doesn't block this decision.
- **model comparison is conditional** — `claimed_model` is optional. It's only enforced
  (able to REJECT) when the model *read's own confidence* is `>= MODEL_CONF_MIN` (default
  90). Below that threshold — or if `claimed_model` wasn't supplied, or Claude is
  unavailable — the model read is still reported (`extracted_model_raw`,
  `model_confidence`) but doesn't affect `decision`; `model_checked` tells you whether it
  was actually enforced. Qwen2.5-VL (`backends/qwen.py`) is the real option for exact
  model reading but isn't run live here — see Origin above.

See `make_model_check.py:validate_make_model` / `decide_make_model`.

**Combined — the single entry point** (`combined.py:validate_upload`):
orchestrates all three checks behind one call — what an external platform registers
per image type. Q1 gates Q2+Q3 entirely (a Q1 REJECT/MANUAL_REVIEW wins outright and
**neither Q2 nor Q3 runs**, saving the Rekognition/SigLIP calls on an image that isn't
even a genuine front-of-truck photo). Once Q1 PASSes, Q2 and Q3 run independently of
each other — neither gates the other, so a single call always returns the complete
audit trail rather than stopping at the first mismatch. The overall decision takes the
worst of the two: `REJECT` > `MANUAL_REVIEW` > `PASS`.

This used to be "Q1+Q2 as one shared Claude call" (back when both were Claude-only);
Q2 no longer calls Claude at all, so there was no shared round-trip left to combine —
the short-circuit, and now the full three-check orchestration, is what replaces that.

**"One prompt per image type," reframed**: a prompt can only instruct one model call —
it can't itself invoke YOLO, SigLIP, AWS Rekognition, or an HSV pixel classifier. So if
the platform's provision is a single registrable *entry point* per image type (not
literally one string sent to one model), `validate_upload` **is** that entry point, and
what the platform stores/shows as "the prompt" is the human-readable description of
what it checks and which model does each piece — see **`VALIDATION_SPEC.md`** — decoupled
from how it actually runs.

## Test/inference interface

A Gradio app (`vfiv/webapp.py`) for comparing backends per stage — **this is a testing
tool, not the production path**; `validate_upload` always uses today's defaults
regardless of what you pick here.

```bash
python -m vfiv.webapp   # → http://127.0.0.1:7860
```

- **Individual test only** — upload one image, enter truck number / make / model
  (optional), pick a backend per stage, run. No bulk/CSV mode anywhere in this
  interface — one upload at a time, matching how it's actually used.
- **Top-level layout** — grouped by which physical photo the checks run against, not
  by check name, since that's the natural mental model for someone testing an upload
  ("does this front photo pass?" not "does Q1 pass?"):
  - **Front Image** — Q1 (front-image gate) / Q2 (VRN + colour) / Q3 (make + model) /
    end-to-end, all sub-tabs testing the same uploaded front photo. Q1 takes an
    optional VRN, which (if filled in) also checks this upload against the `"front"`
    reference corpus; **End-to-end** always runs that same duplicate check as part of
    Q1, since it already requires a truck number for Q2/Q3 regardless.
  - **Side/Axle Image** — broken into check-based buckets: **Axle count** (isolated
    `check_axle_count`), **Identity binding** (isolated `check_side_identity`, routed
    into vrn_visible/corner_view/pure_side_profile), **Duplicate check** (an explicit,
    standalone check against the `"side"` corpus), and **End-to-end** (the combined
    `check_side_image_upload`, worst of axle/identity/duplicate). The first three are
    independently unit-testable — see `side_image/side_image_check.py`.
  - **FASTag** — **QR / Barcode read** and **Printed digits (OCR)** always show the
    raw decode/read; give them a claimed value and they'll ALSO return a narrower,
    single-source verdict (`decide_qr_only` / `decide_printed_digits_only`) — leave
    the claim blank for the old raw-read-only behaviour. Neither runs the
    cross-source tamper check (`decide_fastag` needs all three sources together to
    judge consistency) — an isolated bucket fabricating that would be misleading, so
    it stays exclusive to **End-to-end** (`check_fastag_upload`), which (like Q1)
    also takes an optional VRN to check against the `"fastag"` reference corpus.
  - **Reference Images** — a separate top-level tab for *seeding* the corpus (see
    "Duplicate detection" below) — it stores an embedding, full stop. It never runs a
    duplicate search or produces a decision; the actual check happens later, when a
    real front/side/FASTag upload is tested through the tabs above.
- **Backend choices per stage** (`experiments/q1_select.py` / `q2_select.py` / `q3_select.py`):
  - **Q1**: `real_cv` (production default) | `claude` (the original all-Claude Q1,
    reconstructed) | `gemini`
  - **Q2**: `rekognition` (production default) | `claude` (original all-Claude Q2) | `gemini`
  - **Q3 make**: `siglip_rekognition` (production default) | `claude` | `gemini` |
    `gcv_logo` (Google Cloud Vision Logo Detection) | `clarifai` (Clarifai logo-recognition)
  - **Q3 model**: `claude` (production default) | `gemini`

**New experimentation backends** (`backends/gemini.py`, `google_vision.py`,
`clarifai_backend.py`):

| Backend | Needs | Status here | Caveat |
|---|---|---|---|
| Gemini 2.5 | `GEMINI_API_KEY`, **or** `GOOGLE_APPLICATION_CREDENTIALS` + `GEMINI_VERTEX_PROJECT` (Vertex AI) | ✅ configured (Vertex AI) | Same general-VLM class as Claude — not "specialized," but already your `truck_verification_workflow`'s VLM backbone (chosen for cost). Model is configurable per stage — see Run above (`VFIV_GEMINI_MODEL_Q1`/`Q2`/`Q3_MAKE`/`Q3_MODEL`, default `gemini-2.5-flash`). |
| Google Cloud Vision Logo Detection | `GOOGLE_APPLICATION_CREDENTIALS` | ⚠️ not verified | Reuses the same credentials file as Gemini's Vertex AI mode, but Cloud Vision is a **separate GCP API** that needs its own enablement/IAM permission on that project — untested whether it's actually authorized. Also: its training set is generic/commercial-brand-heavy; unconfirmed whether it recognises Indian truck manufacturer badges at all. |
| Clarifai logo model | `CLARIFAI_API_KEY` | ⚠️ not configured | Unrelated vendor/credential to the Google backends above. Wired via a raw REST call (no SDK dependency) against a configurable model reference (`CLARIFAI_USER_ID`/`APP_ID`/`MODEL_ID`) since the exact model you have access to may differ. Same accuracy caveat as GCV. |

Each degrades gracefully ("not configured") when its credentials are missing, exactly
like `backends/qwen.py`.

The reconstructed legacy all-Claude Q1/Q2/Q3 prompts live in
`experiments/legacy_prompts.py` — verbatim copies of what each stage looked like before
its real-model rewrite (see Origin above), kept here purely so the old design stays
A/B-testable rather than lost.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # needs Python <3.13 (torch 2.2.2 has no 3.13+ wheels)
pip install -r requirements.txt && pip install -e .
pip install -e ~/truck_extract_match                 # local sibling repo, not on PyPI

# FASTag checks also need the zbar shared library at the OS level (not a pip
# package) -- `apt-get install libzbar0` (Debian/Ubuntu) or `brew install zbar`
# (macOS); see "FASTag check" below for the Apple Silicon caveat.

export ANTHROPIC_API_KEY=...      # Q1 screenshot/AI-gen judgment + Q3 model reading
export AWS_ACCESS_KEY_ID=...      # Q2 VRN (Rekognition) + Q3's Rekognition make backstop
export AWS_SECRET_ACCESS_KEY=...  # (already set in this environment)

# Optional — only needed for the test/inference interface's alternate backends.
# Gemini is configured in this environment via Vertex AI (a GCP service account, not
# a simple API key) — backends/gemini.py tries GEMINI_API_KEY first, then falls back
# to Vertex AI automatically:
export GEMINI_API_KEY=...                       # simplest: Gemini Developer API / AI Studio key
# — or, Vertex AI (what's actually configured here) —
export GOOGLE_APPLICATION_CREDENTIALS=...        # service-account JSON path (also used by GCV below)
export GEMINI_VERTEX_PROJECT=...                 # GCP project id (read from the JSON's project_id)
# export GEMINI_VERTEX_LOCATION=...              # defaults to us-central1

# Optional per-stage model override (defaults to VFIV_GEMINI_MODEL, itself defaulting
# to gemini-2.5-flash) — e.g. cheap Flash for Q1/Q2, stronger Pro for Q3:
# export VFIV_GEMINI_MODEL_Q1=gemini-2.5-flash
# export VFIV_GEMINI_MODEL_Q2=gemini-2.5-flash
# export VFIV_GEMINI_MODEL_Q3_MAKE=gemini-2.5-pro
# export VFIV_GEMINI_MODEL_Q3_MODEL=gemini-2.5-pro

export CLARIFAI_API_KEY=...                      # Q3 make alternate: Clarifai logo model — not configured here
# GOOGLE_APPLICATION_CREDENTIALS above also backs Q3's "gcv_logo" backend (GCV Logo
# Detection) — but that's a SEPARATE GCP API from Vertex AI's Gemini access, so having
# working Gemini credentials doesn't guarantee Cloud Vision is enabled/authorized for
# the same service account; test it before trusting it.

python -m vfiv.cli --image samples/truck1.png --type front
python -m vfiv.cli --image samples/truck2.jpg --type vrn --vrn UP42T4069
python -m vfiv.cli --image samples/truck1.png --type make_model --make "ASHOK LEYLAND LTD"
python -m vfiv.cli --image samples/truck2.jpg --type make_model --make "TATA MOTORS LTD" --model-name "LPT 1613"
python -m vfiv.cli --image samples/truck2.jpg --type combined --vrn UP42T4069 --make "TATA MOTORS LTD"
```

```python
from vfiv import validate_front_image, validate_vrn, validate_make_model, validate_upload

front = validate_front_image("path/to/upload.jpg")
front.decision, front.reason   # "PASS" | "REJECT" | "MANUAL_REVIEW"

vrn = validate_vrn("path/to/upload.jpg", claimed_vrn="UP42T4069")
vrn.decision, vrn.status, vrn.plate_colour, vrn.inferred

make_model = validate_make_model("path/to/upload.jpg", claimed_make="TATA MOTORS LTD",
                                 claimed_model="LPT 1613")  # claimed_model optional
make_model.decision, make_model.make_status, make_model.make_match_via, make_model.model_checked

# the single entry point — all three checks, one call
combined = validate_upload("path/to/upload.jpg", claimed_vrn="UP42T4069",
                           claimed_make="TATA MOTORS LTD")  # claimed_model optional too
combined.decision, combined.front, combined.vrn, combined.make_model  # Q2/Q3 skipped if Q1 doesn't PASS
```

## Tests

```bash
pytest -q   # offline (no credentials) tests + live tests against samples/ — 54 total
```

Live tests are gated by whichever credential each check actually needs (not just
`ANTHROPIC_API_KEY` anymore) — `AWS_ACCESS_KEY_ID` for Q2/Q3's Rekognition calls, both
for the full `combined` path. `test_make_check_works_without_any_api_key` specifically
proves Q3's make decision no longer depends on Claude being available at all.

## Layout

```
src/vfiv/
  config.py                    # model names + thresholds (gate, VRN edit budget, model-conf gate, device)
  schemas.py                   # FrontImageResult, VrnCheckResult, MakeModelCheckResult, CombinedResult (pydantic)
  backends/                    # real models, copied from the sibling repos (see Origin)
    device.py                  # torch cuda -> mps -> cpu resolution
    image_io.py                 # path/PIL/ndarray -> RGB ndarray, for the CV backends
    vehicle.py                  # YOLOv8 vehicle detector (Q1)
    siglip.py                   # SigLIP 2: pose classifier (Q1) + zero-shot make classifier (Q3)
    gate.py                      # Q1 fusion: vehicle + pose + completeness heuristic -> one gate result
    plate_colour.py              # HSV plate-colour classifier (Q2), no model weights
    rekognition.py                # AWS Rekognition VRN + brand-text detector (Q2, backstops Q3)
    qwen.py                        # Qwen2.5-VL wrapper (Q3 model, code-only — not run live here)
    gemini.py                      # Gemini 2.5 wrapper (experimentation only, no key configured)
    google_vision.py                # GCV Logo Detection (experimentation only, no key configured)
    clarifai_backend.py              # Clarifai logo model (experimentation only, no key configured)
    vector_store.py                  # pgvector nearest-neighbor store (duplicate detection, see below)
    fastag_reader.py                  # barcode/QR decode + Rekognition text read (FASTag check, see below)
  base.py                       # shared Claude image+prompt -> JSON plumbing (Q1's judgment calls, Q3's model read)
  combined.py                   # single entry point: Q1 gates Q2+Q3, worst-of severity ordering
  duplicate_check.py             # cross-upload near-duplicate check, NOT wired into combined.py (see below)
  front_image/                   # checks that run against the FRONT photo
    front_image.py                 # Q1: real CV gate + narrowed Claude prompt -> decide_front_image()
    vrn_check.py                    # Q2: Rekognition + HSV -> decide_vrn() (no Claude)
    make_model_check.py              # Q3: SigLIP+Rekognition (make) + Claude (model) -> decide_make_model()
  side_image/                    # checks that run against the SIDE/AXLE photo
    side_image_check.py             # axle count + identity binding + duplicate, NOT wired into combined.py
  fastag_image/                  # checks that run against the FASTAG sticker photo
    fastag_check.py                 # QR/barcode/OCR cross-check, NOT wired into combined.py
  experiments/                  # test/inference interface's backend-selection layer (not production)
    legacy_prompts.py            # reconstructed original all-Claude Q1/Q2/Q3 prompts
    q1_select.py                  # Q1: real_cv | claude | gemini
    q2_select.py                  # Q2: rekognition | claude | gemini
    q3_select.py                  # Q3 make: siglip_rekognition|claude|gemini|gcv_logo|clarifai; model: claude|gemini
    schemas.py                     # ExperimentResult (pydantic)
    runner.py                      # run_test_case(): Q1 -> Q2 + Q3, backend-selectable
  webapp.py                     # Gradio test/inference interface — `python -m vfiv.webapp`
  cli.py                        # `python -m vfiv.cli --image ... --type front|vrn|make_model|combined [--vrn|--make/--model-name]`
```

`VALIDATION_SPEC.md` — the human-readable spec for what `validate_upload` checks and
which model performs each piece, meant for the platform's admin UI / documentation, not
for execution.

## Duplicate detection (fraud lead, not wired into `validate_upload`)

A separate check, `check_duplicate()` (`duplicate_check.py`), for a fraud
pattern outside what Q1/Q2/Q3 look for: a ground agent without the actual vehicle in
front of them re-submitting an **old accepted photo under a new claimed VRN**. Q1–Q3
only ever look at *one* upload in isolation; this looks *across* all past uploads.

- **Embedding**: `SigLipModel.embed_image()` (`backends/siglip.py`) — the same SigLIP 2
  weights already loaded for Q1/Q3, just stopped one step earlier (the raw image
  embedding, before it's compared against any text prompt). No second model.
- **Storage/search**: Postgres + [pgvector](https://github.com/pgvector/pgvector)
  (`backends/vector_store.py`) — each upload's embedding + its claimed VRN + an
  `image_type` is stored; a new upload is compared only against prior uploads of the
  **same** `image_type` via exact cosine nearest-neighbor search (an HNSW index keeps
  this fast at scale). Needs `VFIV_PGVECTOR_DSN` set to a reachable Postgres instance
  with the `pgvector` extension available; degrades to a clear `checked=False` error
  (never a crash) when it isn't configured. The table/index are created automatically
  on first use — no manual migration step.
- **`image_type`** (`config.IMAGE_TYPES` = `["front", "side", "fastag"]`, any string is
  technically accepted) — front/side/FASTag photos are visually unrelated, so each is
  its own corpus; a side photo is never compared against a front photo. `check_duplicate`
  defaults to `"front"`; `side_image_check.py` passes `"side"` explicitly.
- **Decision** (`decide_duplicate`): a hit only counts as a fraud lead when the closest
  match (within the same `image_type`) is near-identical (`cosine similarity >=
  DUPLICATE_SIMILARITY_MIN`, default `0.97`) **and** was filed under a **different**
  claimed VRN — a near-duplicate under the *same* VRN is just an honest re-upload and is
  never flagged. A hit always resolves to `MANUAL_REVIEW`, never an auto-`REJECT`: this
  is a signal for a human, not a verdict.

```bash
python -m vfiv.cli --image samples/truck2.jpg --type duplicate --vrn UP42T4069 \
    --upload-id my_upload_1 --image-type front
```

```python
from vfiv import check_duplicate
result = check_duplicate(image, upload_id="upload_123", claimed_vrn="UP42T4069", image_type="front")
result.is_duplicate_suspect, result.best_match_id, result.best_match_similarity, result.best_match_vrn
result.duplicate_matches  # every suspect on file, not just the closest one -- see below
```

**`duplicate_matches`** — a `DuplicateMatchInfo` list (`upload_id` / `claimed_vrn` /
`similarity`, highest similarity first) of **every** prior upload that cleared
`DUPLICATE_SIMILARITY_MIN` under a different VRN, not just the single closest one
(`best_match_id` etc. still exist too, mirroring `duplicate_matches[0]`, kept for
anyone already reading those fields). This is what a manual reviewer actually needs —
several near-identical prior uploads on file is stronger tampering evidence than one,
and each `upload_id` is what they'd go pull up to eyeball side by side. Folded straight
into the result wherever the duplicate check itself gets folded in — Q1
(`FrontImageResult.duplicate_matches`), FASTag (`FastagCheckResult.duplicate_matches`),
and Side/Axle (`SideImageCheckResult.duplicate_matches`) — so it shows up automatically
in the webapp's JSON output alongside the plain-English `reason`, no separate lookup
needed.

**What `upload_id` is for**: never parsed or interpreted — it's a stable
identifier for the row being stored (in production, the real upload/
submission id from your own system), kept purely so a future match can be
traced back to *which* prior upload it's a near-duplicate of (`best_match_id`
in the result IS this value), and so storage is idempotent (re-running the
same `upload_id` upserts that row instead of accumulating duplicates of
itself). A real claimed VRN is still required to run the check at all (it's
what tells an honest re-upload apart from a swapped-plate fraud lead) — but
the webapp never requires `upload_id` itself: leave it blank and it
auto-generates a random one (`webapp-<uuid4 hex>`) so the check still runs
and stores; only fill it in if you specifically want this stored under an id
of your choosing.

**Q1 (front-image gate)** — `run_q1_only()` (`experiments/runner.py`, also
what the webapp's **Q1** tab calls) accepts the same optional `claimed_vrn` +
`upload_id` pair: give a VRN to also run `check_duplicate(..., image_type="front")`
and fold its verdict into Q1's own decision (worst-of, same REJECT >
MANUAL_REVIEW > PASS ordering as everywhere else); a suspected duplicate
surfaces as `duplicate_is_suspect=True` on the result. Wired at the
test/webapp layer (rather than inside `front_image/front_image.py`'s
production `validate_front_image`) so it works with whichever Q1 backend
you're comparing (`real_cv` | `claude` | `gemini`).

**End-to-end (front image)** — `run_test_case()` (`experiments/runner.py`, what
the webapp's front-image **End-to-end** tab calls) threads its own `claimed_vrn`
into the same `run_q1_only()` above, so the `"front"` duplicate check is folded
into Q1 there too — it runs unconditionally (no separate opt-in field) since a
truck number is already mandatory in that tab for Q2/Q3 to run at all. A
duplicate verdict escalates Q1's decision and, like any other non-PASS Q1
result, short-circuits Q2/Q3 for that request.

**FASTag (end-to-end)** — `check_fastag_upload()` takes the same optional
`claimed_vrn` + `upload_id` pair, folding in `check_duplicate(...,
image_type="fastag")` the same way Q1 does. The isolated QR/Barcode and OCR
buckets don't take a VRN — they never produce a decision at all (see above),
so there's nothing to fold a duplicate verdict into.

**Seeding the reference library** — the webapp's **Reference Images** tab
(`python -m vfiv.webapp`) is for *seeding only*: upload an image + VRN + type
(front/side/fastag) and it vectorizes and stores the embedding via
`store_reference_image()` — no duplicate search, no PASS/REJECT/MANUAL_REVIEW
decision. This is the path for importing a legacy dump of photos that
predates this vector-DB setup. The actual duplicate *check* against whatever
is stored happens later, when a real upload is tested — through Q1/FASTag's
optional VRN field, or the Side/Axle **Duplicate check** bucket. A "Refresh
library stats" button shows how many images are stored per type.

**Running pgvector locally** (e.g. on your own machine, via Homebrew on macOS):

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb vfiv
export VFIV_PGVECTOR_DSN="postgresql://$(whoami)@localhost:5432/vfiv"
```

No further setup needed — `store_embedding`/`find_similar` call `ensure_schema()`
automatically on first use, creating the `vector` extension, table, and indexes.

**Not wired into `combined.py`/`validate_upload`** — call it alongside the combined
check (or from a batch job over recent uploads) rather than folding it into
`CombinedResult.decision`, for two reasons: (1) `DUPLICATE_SIMILARITY_MIN` is an
uncalibrated starting point — a threshold this consequential (it can send a genuine
upload to manual review) should be tuned against real labeled pairs from your own
data before it affects a production decision, and (2) it depends on a live pgvector
instance, which nothing else in this module requires. `PGVECTOR_EMBED_DIM` (default
768) must match whatever `SIGLIP_MODEL` actually outputs — verify with
`SigLipModel().embed_image(img).shape` before running `vector_store.ensure_schema()`
for the first time.

## FASTag check (not wired into `validate_upload`)

`check_fastag_upload()` (`fastag_image/fastag_check.py`) validates a close-up photo of
the FASTag sticker itself — three independent, real reads of the tag's identity,
cross-checked against each other as well as against the claimed value:

- **QR code** (`backends/fastag_reader.py`, via `pyzbar`) — decodes the UPI-recharge
  payload (`<fastag_id>@<bank_code>`) directly. Most damage-tolerant of the three
  (built-in error correction).
- **1D barcode** — decoded directly, checksum-backed.
- **Printed digits** below the barcode — read via OCR, the only genuinely fuzzy
  source of the three (confusable-character tolerance reused from
  `truck_extract_match.plate.format.confusable_distance`). **Backend-selectable**:
  `backend="rekognition"` (default) | `"claude"` | `"gemini"` — the barcode/QR
  decode is a deterministic algorithm either way, not a model call, so only this
  OCR step swaps.

Forging all three consistently is a much higher bar than editing the visible digits
alone, so **a disagreement between sources that were each legibly read is itself a
REJECT**, checked before comparing any of them to the claimed value.

**Single-source checks** (`decide_qr_only` / `check_qr_only`, `decide_printed_digits_only`
/ `check_printed_digits_only`) — narrower than the full cross-source check above, each
validating exactly ONE source against its own claimed value, with no cross-source
tamper check (that stays exclusive to `decide_fastag`):

- **QR only** — exact match (the QR's own error correction already makes it damage-
  tolerant, so no fuzzy tolerance is added here) between the QR-parsed `fastag_id` and
  a claimed **Tag ID**; if a claimed bank code is also given, a QR-matched-but-bank-
  mismatched result downgrades to `MANUAL_REVIEW` rather than `PASS` — same posture as
  the full check's bank-code handling.
- **Printed digits only** — fuzzy match (same confusable-character tolerance as
  everywhere else) between the OCR'd printed digits and a claimed **barcode number** —
  useful when that number was captured independently (e.g. a dedicated handheld
  barcode scan) rather than decoded from this same photo, to cross-validate the two
  independently-obtained values against each other.

Both are opt-in in the webapp: leave the claimed-value field blank on the **QR /
Barcode read** or **Printed digits (OCR)** tab and you get the old raw-read-only
display; fill it in and you get a real `PASS`/`MISMATCH`/`UNREADABLE` verdict on that
one source alone, folded into the same result.

```bash
python -m vfiv.cli --image samples/fastag.jpg --type fastag --fastag-id 607469-009-0874936
python -m vfiv.cli --image samples/fastag.jpg --type fastag --fastag-id 607469-009-0874936 --backend gemini
```

Needs the system zbar shared library for `pyzbar` — a real OS-level dependency,
not just a pip package: `apt-get install libzbar0` (Debian/Ubuntu) or `brew install
zbar` (macOS) — on top of whichever OCR backend's credentials you're using (AWS by
default). Missing it surfaces as `"Unable to find zbar shared library"` (confirmed
during manual testing) — a real barcode/QR-decode failure, not a bug in this
codebase. On Apple Silicon Macs, Homebrew installs to `/opt/homebrew` rather than
`/usr/local`, and `pyzbar` doesn't always find it there automatically; if
`brew install zbar` alone doesn't fix it, also run:
```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```
before launching (Intel Macs: `/usr/local/lib` instead). Either way, restart
whatever's running `vfiv` afterward — the library loads at process start, so an
already-running webapp/CLI won't pick up a newly-installed library.

## Side/axle-image check (not wired into `validate_upload`)

`check_side_image_upload()` (`side_image/side_image_check.py`) validates the side
image used for axle-count verification, with two goals:

**Axle count** — no dedicated axle/wheel detector is wired (would need a custom-
trained model and a labeled dataset); this is a narrowed VLM judgment call instead,
gated by its own reported confidence. **Backend-selectable**: `backend="claude"`
(default) | `"gemini"` — same prompt either way, only which model reads it changes.
Known, real limitations a single 2D photo can't fully resolve: lift/tag axles
raised off the ground, dual/twin wheels vs. tandem/tridem bogies (a genuine 2-4
axle group closely spaced together, vs. one axle's dual wheels) being confused for
each other, and the model substituting brand/model prior knowledge for what's
actually visible — all three confirmed on real uploads during manual testing: a
genuine 2-axle truck's dual rear wheels misread as a third axle at 95% confidence,
and on a retry, the model explicitly justified a second rear axle by reasoning
"this looks like a Tata LPT 1918, which typically runs 6x2" — the vehicle's own
RC-derived class said 2 axles, but the model overrode that with a generic
assumption about the truck family rather than what was in the photo. The prompt
(`AXLE_PROMPT`, covering the full realistic 2-7 axle range, not just the 2-axle
case) asks the model to walk the wheelbase front-to-rear, cite the specific visual
feature (not "typical for this model") behind each axle it counts, and flag
suspected lift axles, rather than silently guessing from a wheel count or a class
assumption.

**Identity-to-claimed-vehicle** — does this side photo belong to the SAME truck as
the claimed VRN/make? Routed by `classify_side_image_type()`
(`side_image_check.py`, `SIDE_IMAGE_TYPE_PROMPT`) into three buckets of
**decreasing** reliability. This used to be a SigLIP zero-shot embedding
comparison (`backends/siglip.py`'s `SideImageTypeClassifier`, now removed
entirely) — it kept misrouting a genuine corner-view upload where the side of
the cargo box dominates the frame, and rewording its text prompts around
**windshield visibility** (the real discriminator: a true side profile is shot
perpendicular to the truck's length, so the forward-facing windshield is edge-on
or invisible; any shot where it's actually visible implies a forward-facing
viewing angle, which by definition makes it a corner/three-quarter shot) still
wasn't enough — rewording the TEXT side of a zero-shot comparison can't change
how the IMAGE itself embeds, so a stubbornly "side-profile-shaped" photo stayed
misrouted regardless of wording. Replaced with a VLM call (Claude/Gemini,
`config.SIDE_IMAGE_TYPE_BACKEND`) that reasons explicitly about windshield/plate
visibility and cites its evidence — the same "reasoning over embedding-
similarity" fix already applied to axle counting above. Degrades to
`MANUAL_REVIEW` (not a crash) if the VLM call itself is unavailable (missing
credentials, etc.), same posture as every other VLM-backed check here.

| Bucket | Strategy | Reliability |
|---|---|---|
| `vrn_visible` | Re-runs Q2's own VRN detector/matcher on this image, unchanged | Strong — exact identity |
| `corner_view` | A SigLIP embedding similarity + a colour-histogram comparison, both against this truck's own on-file front photo | Uncalibrated — see caveat below |
| `pure_side_profile` | Colour-histogram comparison only, against the same front reference photo | Weak by design — never a confident PASS alone |

**No make/model check anywhere in this module any more** — it used to run in both
`corner_view` and `pure_side_profile`, and was dropped from both: `MakeClassifier`
is a coarse, brand-only zero-shot read (compares the image against 8 fixed
brand-name text prompts, never actually reads painted logos/text — see the known
zero-shot-make limitation flagged above under "SigLIP 2") that can misfire between
visually-similar cab shapes across manufacturers — confirmed on a real upload
during manual testing: a genuine Tata read as "Eicher" **twice**, on separate
uploads of the same photo (see `diagnose_side_image.py`). Its old REJECT-on-
mismatch behaviour meant a genuine truck could get auto-rejected on what amounts
to a coin-flip brand read.

`corner_view` relies on two direct 1:1 comparisons against `front_reference_image`
instead — a SigLIP embedding similarity and a colour histogram (see below);
`pure_side_profile` relies on the colour-histogram comparison alone (no embedding
check there — SigLIP's embedding is angle-sensitive, so a side profile vs. a
front-on photo would likely score low even for the same truck; colour is roughly
angle-invariant and is the one signal that transfers). Without a
`front_reference_image` to compare against, both buckets are `MANUAL_REVIEW`
("unverifiable") — no fallback to the make classifier.

The `corner_view` embedding-similarity check is a **direct 1:1 comparison**
(`front_reference_image` vs. this crop), not a vector-DB search — and it is
explicitly **uncalibrated**: a general SigLIP embedding is trained for semantic
similarity (what make/model is this), not individual-vehicle re-identification, so
it may not reliably separate "same truck, different angle" from "different truck,
same make/model/colour." Validate `config.SIDE_IMAGE_SIMILARITY_MIN` against real
labeled pairs before trusting it — now that it's one of `corner_view`'s only two
signals, this matters more than it used to.

**Colour-histogram check** (`_color_histogram_similarity`, `config.SIDE_IMAGE_COLOR_HIST_MIN`,
default `0.8`) — a real, deterministic paint-colour comparison, no ML judgment
call involved: an RGB histogram correlation between the side photo's vehicle crop
and the front reference's vehicle crop. Both crops come from the same vehicle
detector already used elsewhere in this module (`get_vehicle_detector().best_truck()`)
— **never the raw uncropped photo**, since background/road/sky colour would
otherwise swamp the actual paint-colour signal. In `corner_view`, folded in
alongside the embedding-similarity check (either one failing is `MANUAL_REVIEW`,
never `REJECT` on its own); in `pure_side_profile`, it's the only signal —
capped at `MANUAL_REVIEW` even on a perfect match (individual-vehicle identity
still isn't solved — two different trucks of the same colour would pass this
too), and (unlike the make classifier it replaced) also `MANUAL_REVIEW` rather
than `REJECT` on a mismatch, same uncalibrated-threshold posture as `corner_view`.
Also **uncalibrated** in general: lighting/exposure can differ meaningfully
between two photos of the same truck shot at different times, so validate
`config.SIDE_IMAGE_COLOR_HIST_MIN` against real same-truck/different-truck pairs
before trusting it, same caveat as the embedding-similarity threshold.

The `pure_side_profile` bucket remains the genuinely open problem flagged in
design discussion, not solved here — a bare side profile has nothing but colour
to go on (no plate, no front grille), and colour alone can never prove individual-
vehicle identity.

**Axle-count source consistency** (`decide_axle_source_consistency`,
`check_axle_count`'s optional `axle_source`/`vehicle_mapper` params) — a separate,
non-photo data-consistency check, independent of everything above: does the
claimed axle count actually agree with the vehicle's own RC data? `axle_source`
is `"auto"` (pulled straight from the RC — trusted as-is, no further check) or
`"manual"` (a field agent typed it in — cross-checked against `vehicle_mapper`,
the vehicle's RC-derived class code, via a fixed lookup table:

| Vehicle mapper | Axle count | Vehicle mapper | Axle count |
|---|---|---|---|
| `VC4` (Car) | n/a | `VC10` | 2 |
| `VC20` | 2 | `VC11` | 3 |
| `VC9` | 2 | `VC12` | 4 |
| `VC7` | 2 | `VC13` | 5 |
| `VC8` | 3 | `VC14` | 6 |
| `VC5` | 2 | `VC15` | 7 |

A mismatch on the `"manual"` path is `REJECT` (still RC-derived, just one hop
via the class lookup rather than a direct field) — this can catch a fabricated
axle count even before the photo is looked at, folded in worst-of alongside the
existing photo-vs-claim check. Both `axle_source` and `vehicle_mapper` are
opt-in — omitting `axle_source` skips this check entirely, same pattern as the
duplicate check elsewhere.

Duplicate detection reuses `check_duplicate()` unchanged, always run (the
webapp auto-generates an `upload_id` when none is given) — scoped to the
`"side"` `image_type`, so it's never compared against front or FASTag
embeddings (see the `image_type` scoping above).

```bash
python -m vfiv.cli --image samples/side.jpg --type side --vrn UP42T4069 --make "TATA MOTORS LTD" --axle-count 3
python -m vfiv.cli --image samples/side.jpg --type side --vrn UP42T4069 --make "TATA MOTORS LTD" --axle-count 3 --backend gemini
```

## Next

- **Side image** validator: new real-model + prompt combination for side-view
  completeness + VRN/plate visibility checks.
- **Fastag image** validator: new prompt/schema for a genuine Fastag sticker/tag photo
  (tag ID legibility, placement, tamper/fraud red flags).
- **Truck-finetuned vehicle detector**: replace the generic COCO `yolov8n.pt` in Q1 to
  fix the documented false-reject limitation.
- **Verify the new experimentation backends once credentials exist**: Gemini, GCV Logo
  Detection, and Clarifai are all wired but unverified against real samples — test them
  via the interface (`python -m vfiv.webapp`) before trusting any of them, especially
  GCV/Clarifai's accuracy on Indian truck manufacturer badges specifically (unconfirmed
  — their training data leans generic/commercial-brand-heavy).
- **Fine-tuned or Qwen-backed make classifier**: reduce reliance on the Rekognition
  backstop for Q3's make check.
- **Qwen2.5-VL on a GPU box**: swap Q3's model reading from Claude to the already-wired
  `backends/qwen.py` once real GPU infra is available.
