# fastag_pipeline

A standalone batch runner for the KYV FASTag check: give it a list of FASTag
photos (a CSV or Excel manifest) with the claim attached to each one
(barcode number, Tag ID, bank code), and it runs every image through the
agreed decision flow, one by one, and writes the results to an Excel file —
one row per image.

**Self-contained**, same pattern as `batch_pipeline/` (the front-image
equivalent) in this repo — no shared imports with the rest of this
repository, so it's easy to hand to someone on its own.

## The flow

```
FASTag image upload (+ claimed barcode, Tag ID, bank code)
        |
        v
1. Front check (one vision call): fully framed AND not altered/AI-generated
        |-- not fully framed --------------------> MANUAL REVIEW
        |-- altered/AI-generated ----------------> MANUAL REVIEW
        `-- both OK --+
                      v
2. Barcode & QR readability (zxingcpp decode attempt, no vision call)
   - Barcode readable = decodes cleanly, every digit resolved
   - QR readable = decodes AND payload matches the expected FASTag schema
     (Tag ID + bank code fields present)
        |-- BOTH unreadable ---------------------> MANUAL REVIEW
        `-- at least ONE readable --+
                                    v
              +---------------------+---------------------+
   barcode unreadable, QR readable          barcode readable (any QR state)
              |                                            |
              v                                            v
    5. QR PARSING (see below)                 3. BARCODE CHECK
                                                  Decoded barcode == claimed value?
                                                     |-- NO --> REJECT
                                                     `-- YES --+
                                                               v
                                             QR readable?
                                                |-- NO --+
                                                |         v
                                                |    4. Barcode stripe-validity
                                                |       check (zxingcpp `.valid`)
                                                |         |-- valid ------> APPROVE
                                                |         `-- invalid ----> MANUAL REVIEW
                                                `-- YES --+
                                                          v
                                            5. QR PARSING
                                               Extract Tag ID + bank code,
                                               compare both to claim
                                                  |-- both match --> APPROVE
                                                  `-- either mismatches --> REJECT
```

Notes on the design:
- **No maker check, no duplicate check** — unlike the front-image flow, this
  check is entirely about Tag-ID/barcode/bank-code matching; there's no
  manufacturer marking on a FASTag and no duplicate-photo signal defined for
  it (yet).
- **Barcode and QR are decoded deterministically**, not read by the vision
  model — a barcode/QR is built to be machine-decoded, so `zxingcpp` (a
  Python binding for the ZXing-C++ library) does this locally, for free, with
  no guessing. The vision call is reserved for the one genuinely subjective
  judgment: is this fully framed, and does it look tampered with?
- **"Readable" is decode success, full stop** — a barcode is readable if the
  decoder recovers a value at all (a partially obscured barcode fails to
  decode rather than half-guessing); a QR is readable if it decodes *and*
  matches the expected FASTag payload schema (see the note below).
- **A barcode mismatch rejects outright**, before the QR is ever consulted.
  This severity was flagged as worth revisiting once there's real
  prompt/decode data to look at — see `fastag_sequence.py`'s comment at that
  branch.
- **The barcode "stripe-validity" fallback** (barcode matched, QR
  unreadable) uses `zxingcpp`'s `.valid` flag on the barcode decode — most
  barcode symbologies (Code128, EAN-13, ...) encode a checksum digit as part
  of the stripe pattern itself, so a decode only reports `valid` if that
  checksum reconciles. That's the self-consistency proof this step needs,
  with no extra logic required.

### ⚠️ Unconfirmed: the FASTag QR payload format

`kyv_fastag/barcode_qr.py`'s `parse_fastag_qr_payload` assumes a placeholder
format (`TAGID:<id>;BANK:<code>`) — this has **not** been verified against a
real NETC/NPCI FASTag QR code. Every QR will read as unreadable until this
regex matches the real payload shape. Replace it once you have either a real
decoded QR string from an actual tag, or NETC's documented format.

## Setup

```bash
cd fastag_pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, set OPENROUTER_API_KEY=sk-or-v1-...
```

No sibling-package install needed here (unlike `batch_pipeline/`) — this
check doesn't use `truck_extract_match`'s VRN/maker matching at all.

## Usage

Build a manifest (CSV or `.xlsx`) — see `sample_manifest.csv`:

| column | required? | meaning |
|---|---|---|
| `image_path` | yes | Path to the image. Relative paths resolve against the manifest's own directory. |
| `claimed_vrn` | yes | Claimed vehicle registration number this FASTag is filed against — audit metadata only, not a gate in this flow. |
| `claimed_barcode` | yes | Claimed barcode number printed on the tag. |
| `claimed_tag_id` | yes | Claimed Tag ID (from the QR). |
| `claimed_bank_code` | yes | Claimed issuing bank/code (from the QR). |
| `upload_id` | no | Auto-generated if left blank. |
| `vision_model` | no | Per-row model override — see below. |

Then run:

```bash
python scripts/batch_check.py --manifest sample_manifest.csv --output results.xlsx
```

### Choosing a model

Only step 1 (framing + tamper) makes a model call. Pick any vision-capable
model from <https://openrouter.ai/models>:

- `--model <id>` sets the default for every row in the batch.
- A row's own `vision_model` column overrides `--model` for just that row.
- Every output row records which model actually ran it, in its own
  `vision_model` column.

## Output columns

One row per image, in this order — the claim, then the script's findings in
the same sequence the checks run, then the branch taken, then the decision,
then cost/latency/token totals:

`image_path, upload_id, claimed_vrn, claimed_barcode, claimed_tag_id, claimed_bank_code,
vision_model, fastag_fully_framed, is_altered_or_ai_generated,
front_image_reasoning, barcode_readable, barcode_value_read, qr_readable,
tag_id_value_read, bank_code_value_read, path_taken, barcode_match,
barcode_checksum_valid, qr_tag_id_match, qr_bank_code_match, final_decision,
final_reason, latency_ms_vision_check, latency_ms_total, prompt_tokens,
completion_tokens, cost_usd`

Notes:
- `path_taken` is one of `both_unreadable` / `barcode_only` / `qr_only` /
  `both_readable` — which of the four branches this image took, useful for
  auditing at a glance which columns are meaningfully blank vs. never
  reached.
- `barcode_checksum_valid` is only populated on the `barcode_only` path (no
  QR to cross-check against).
- `qr_tag_id_match` / `qr_bank_code_match` are only populated when the QR
  was actually parsed (`qr_only` or `both_readable` paths).
- `latency_ms_vision_check` is step 1's model call only; `latency_ms_total`
  adds the local barcode/QR decode's own wall-clock time (real, even though
  it's free).

Re-running the same `upload_id` is a no-op (nothing charged) unless you pass
`--force`.
