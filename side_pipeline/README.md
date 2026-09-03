# side_pipeline

A standalone batch runner for the KYV side/axle image check: give it a list
of side/axle photos (a CSV or Excel manifest) with the claim attached to
each one (vehicle type, axle count, VRN), and it runs every image through
the agreed decision flow, one by one, and writes the results to an Excel
file — one row per image.

**Self-contained**, same pattern as `batch_pipeline/` and `fastag_pipeline/`
in this repo — no shared imports with the rest of this repository.

## The flow

```
SIDE / AXLE IMAGE UPLOAD (+ claimed vehicle_type, claimed axle_count, claimed_vrn)
        |
        v
1. Vehicle type = Bus/Truck AND not altered/AI-generated?  (one combined vision call)
        |-- not bus/truck ------------------------------> REJECT
        |-- altered/AI-generated ------------------------> REJECT
        `-- OK --+
                 v
2. Full side of vehicle visible? (cabin AND axle both in frame)
        |-- NO -------------------------------------------> REJECT
        `-- YES --+
                  v
3. Axle count = claimed?
        |-- NOT EQUAL -------------------------------------> REJECT (axle count mismatch)
        `-- EQUAL --+
                     v
4. VRN visible AND readable? (never guess; plate OR painted/stencilled text)
        |-- conflicting renderings (plate vs. painted disagree) --> MANUAL REVIEW
        |-- not visible / not readable ---------------------------> MANUAL REVIEW
        `-- readable (non-conflicting) --+
                                          v
                              5. VRN CHECK (same confusable-edit-distance
                                 match the front-image flow uses)
                                   |-- match --+
                                   |-- mismatch_similar --------> MANUAL REVIEW
                                   `-- mismatch_other ----------> REJECT
                                                |
                                                v
                                    6. DUPLICATE CHECK (shared gate --
                                       reuses the front-image flow's pHash ->
                                       SigLIP cascade, image_type="side")
                                       |-- duplicate --> MANUAL REVIEW
                                       `-- clean -------> APPROVE
```

Design notes:
- **No front-image lookup, no visual cross-check.** An earlier design had a
  fallback that looked up whether the front image was already approved and,
  if so, ran a cabin-only visual comparison against it when the VRN
  couldn't be read. It was dropped: front and side images arrive together
  and are reviewed front-first in practice, so "front image approved?"
  would almost always be false at check time — the fallback would rarely
  fire, just adding a vision call and two gates for a path that dead-ends
  at Manual Review anyway. An unreadable VRN now goes straight to Manual
  Review.
- **VRN can be a plate or painted/stencilled text** — many commercial
  vehicles carry the registration number painted directly on the body
  (door, cabin side, mudguard) in addition to, or instead of, a plate on
  this side. The read step checks both, and if it finds two disagreeing
  renderings, that goes straight to Manual Review with both values in the
  reasoning — never picked arbitrarily.
- **No Pure Side vs. Corner branch.** Both were originally classified and
  ran different logic (a colour-histogram check for Pure Side, a direct
  front-portion match for Corner). Once the framing gate started requiring
  the *whole* side of the vehicle — cabin and axle both — visible regardless
  of framing style, there was nothing left for the two to diverge on, so
  they're unified into one flow. `view_type` is still captured in the
  manifest/output purely as a label, in case you want to track which
  framing style was actually used, but it drives no check.
- **Axle visibility and axle count are separate gates** — a reviewer sees
  "axle not visible" and "axle count wrong" as distinct reasons, not
  collapsed into one.
- **Duplicate check only runs on the path that would otherwise approve**,
  same rule as the front-image flow — masks out the VRN region (wherever it
  was found) before hashing/embedding, so swapping only the VRN can't dodge
  it.

## Setup

```bash
cd side_pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../truck_extract_match   # pure-logic VRN matching, from this repo's root

cp .env.example .env
# edit .env, set OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

Build a manifest (CSV or `.xlsx`) — see `sample_manifest.csv`:

| column | required? | meaning |
|---|---|---|
| `image_path` | yes | Path to the image. Relative paths resolve against the manifest's own directory. |
| `claimed_vehicle_type` | yes | `bus` or `truck` |
| `claimed_axle_count` | yes | Claimed axle count |
| `claimed_vrn` | yes | Claimed vehicle registration number |
| `front_image_upload_id` | no | Audit link to the corresponding front-image record — not used by any check |
| `view_type` | no | `pure_side` \| `corner` — captured only |
| `upload_id` | no | Auto-generated if left blank |
| `vision_model` | no | Per-row model override — see below |

Then run:

```bash
python scripts/batch_check.py --manifest sample_manifest.csv --output results.xlsx
```

**Duplicate check needs a seeded corpus first** — without one, every image
reports `duplicate_signal_used = none`:

```bash
python scripts/seed_reference.py --image samples/known_good_side.jpg \
    --upload-id ref_001 --vrn MH12AB1234
```

### Choosing a model

Four vision calls per image (type/tamper, framing, axle count, VRN read).
Pick any vision-capable model from <https://openrouter.ai/models>:

- `--model <id>` sets the default for every row in the batch.
- A row's own `vision_model` column overrides `--model` for just that row.
- Every output row records which model actually ran it, in its own
  `vision_model` column.

## Output columns

One row per image, in flow order — the claim, then the script's findings in
the same sequence the checks run, then cost/latency/token totals:

`image_path, upload_id, claimed_vehicle_type, claimed_axle_count, claimed_vrn,
front_image_upload_id, view_type, vision_model, detected_vehicle_type,
vehicle_type_match, is_altered_or_ai_generated, type_tamper_check_reasoning,
full_side_visible, detected_axle_count, axle_count_match, vrn_visible,
vrn_readable, vrn_conflicting_renderings, vrn_value_read, vrn_fuzzy_outcome,
duplicate_signal_used, is_duplicate, duplicate_match_upload_ids,
final_decision, final_reason, latency_ms_vision_checks, latency_ms_total,
prompt_tokens, completion_tokens, cost_usd`

Any column belonging to a check that never ran (e.g. `vrn_fuzzy_outcome`
when the VRN wasn't readable) is blank.

Re-running the same `upload_id` is a no-op (nothing charged) unless you
pass `--force`.

## What this does not cover

Same caveats as `batch_pipeline/`/`fastag_pipeline/` — structured-output
support varies by model, no image downscaling, no concurrency, and the
duplicate-check thresholds (`DUPLICATE_HAMMING_MAX`,
`DUPLICATE_SIGLIP_SIMILARITY_MIN`) are uncalibrated starting points.
