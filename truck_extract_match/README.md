# truck_extract_match

Shared **extract-and-match** module for the truck/bus verification workflow — the first
build unit. Covers **Q2 (plate number / VRN)** and **Q3 (make)** with one skeleton:

```
detect region  ->  read text/logo  ->  domain-normalise  ->  match claimed value  ->  status
```

`status ∈ {MATCH, MISMATCH, UNREADABLE}` maps straight onto the flow diagram:
`MATCH` → continue, `MISMATCH` → auto-reject, `UNREADABLE` → caller routes (reject / manual).

## Why one module for both

| | Q2 — VRN | Q3 — Make |
|---|---|---|
| detect | plate box (fast-alpr / YOLO) | grille/front region or whole image |
| read | OCR the crop (PaddleOCR/PARSeq) or VLM | word (OCR/VLM) + optional logo classifier |
| normalise | Indian plate grammar + smudge inference | brand ↔ Parivahan legal-entity aliases |
| match | confusion-aware edit distance to claimed VRN | brand-set intersection vs claimed maker |

Reading **only the detected crop** is what excludes painted body text ("Horn OK Please",
phone numbers) — it is never fed to the reader.

## What's real vs what you wire

- **Pure logic (implemented, tested, no models):** `plate/format.py` (state whitelist,
  confusion table, confusable edit distance, grammar correction, smudge inference) and
  `make/aliases.py` (brand→maker alias map, canonicalisation, matching).
- **Model backends (you wire to real models):** `adapters.py` — `FastAlprDetector`,
  `PaddleOCRReader`, `VLMReader` (inject your Qwen2.5-VL / Gemini call), `WholeImageDetector`.
  Nothing fabricates a read; missing deps raise a clear error.

## Usage

```python
from truck_extract_match import PlateVerifier, MakeVerifier
from truck_extract_match.adapters import FastAlprDetector, PaddleOCRReader, VLMReader, WholeImageDetector

# Q2 — plate: real detect + OCR
plate = PlateVerifier(FastAlprDetector(), PaddleOCRReader(lang="en"))
r = plate.verify(image, claimed_vrn="MH12AB1234")
if r.status.name == "MISMATCH":
    ...  # auto-reject

# Q3 — make: VLM reads word + logo (inject YOUR real Qwen2.5-VL / Gemini call)
def qwen_generate(img, prompt) -> str:
    ...  # call your served model, return its text
make = MakeVerifier(WholeImageDetector(), VLMReader(qwen_generate, mode="make"))
m = make.verify(image, claimed_make="VE COMMERCIAL VEHICLES LTD")
```

## Smudge inference (Q2)

Because the pipeline always has the claimed VRN, the read is compared with a
**confusion-aware** edit distance: `0↔O`, `8↔B`, `5↔S`, `2↔Z`, `6↔G`, `1↔I/L`, etc. cost
**0**. A smudged char that is visually consistent with the claimed plate is accepted and
flagged `inferred`. `max_confusable_edits` (default 1) also tolerates one genuine error.
Tune it per your precision target.

## Brand ↔ legal-entity (Q3)

Parivahan stores the maker as a legal entity, not the painted brand. Both sides are
canonicalised to brand keys and matched on set intersection:
`EICHER` ↔ `VE COMMERCIAL VEHICLES LTD`, `BharatBenz` ↔ `DAIMLER INDIA COMMERCIAL VEHICLES`,
`TATA` ↔ `TATA MOTORS LTD`. VECV (Eicher **and** Volvo) maps to a set, so either brand
satisfies the claim. Curate `_MAKER_TO_BRANDS` once from your VAHAN distinct-maker list.

## Test / demo

```bash
cd truck_extract_match
PYTHONPATH=. python -m pytest tests/ -q     # pure-logic tests, no models needed
PYTHONPATH=. python demo.py                 # logic walk-through with a stub reader
```

`rapidfuzz` is used if installed; otherwise a pure-Python fuzzy fallback runs.

## Integrating

Drop `truck_extract_match/` into `truck_front_extractor` (already does front-gate +
attribute extraction) or `truck_verification_pipeline`, reuse the served Qwen2.5-VL via
`VLMReader`, and call `PlateVerifier`/`MakeVerifier` for the Q2/Q3 nodes. `FieldVerification.status`
drives the branch; `.evidence`/`.notes` feed the manual-review UI.
```
