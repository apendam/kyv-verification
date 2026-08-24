# Validation Spec: Truck/Bus Front Image

This is the human-readable specification for the **"truck/bus front image"** validation
type — what an admin configuring the verification platform would register against that
image type, and what a reviewer would read to understand why an upload passed, was
rejected, or was sent for manual review.

**This document is not executed by a model.** It describes what the entry point
(`vfiv.validate_upload`) checks and which real model performs each piece. A single text
prompt can only instruct one model call — it cannot itself run a vehicle detector, an
OCR engine, or a pixel-colour classifier — so the checks below run as code, calling
whichever model actually does that job, not as one prompt interpreted by one model.

## Inputs

| Input | Required? | Description |
|---|---|---|
| `image` | required | The uploaded file (path, PIL.Image, or ndarray). |
| `claimed_vrn` | required | The registration number this upload is supposed to belong to. |
| `claimed_make` | required | The manufacturer this upload is supposed to belong to. |
| `claimed_model` | optional | The specific model/variant, if known — only enforced if legibly read. |

Where these claimed values come from (a real backend lookup, a form field, a test
harness) is outside this validator's concern — it only compares against whatever it's
given.

## What gets checked, and by what

### 1. Is this a genuine front photo of a truck/bus?
| Sub-check | Model | Notes |
|---|---|---|
| Is the vehicle a truck/bus? | **YOLOv8** (vehicle detector) | Falls back to a generic COCO-trained model in this environment (no custom truck weights) — known to occasionally miss real Indian trucks; swap in a truck-finetuned detector to fix. |
| Is the viewpoint front-on? | **SigLIP 2** (zero-shot pose classifier) | Classifies front / three-quarter-front / side / rear. |
| Is the whole front visible (not cropped)? | Geometric heuristic | Based on how much of the frame the vehicle bbox covers and whether it's clipped at an edge. |
| Is this a screenshot / photo of a screen? | **Claude** | Judgment call — no CV model does this reliably. |
| Is this a re-photographed printed photo? | **Claude** | Distinct from the screenshot check — physical print, not digital screen. |
| Is this AI-generated / synthetic? | **Claude** | Conservative — only flags clear synthetic artifacts. |

**If this fails, nothing else runs** — an image that isn't a genuine front-of-truck
photo makes every downstream check meaningless (and wastes real API calls).

### 2. Does the plate match the claimed VRN?
| Sub-check | Model | Notes |
|---|---|---|
| Read the plate number | **AWS Rekognition** (text detection) | Real Indian-VRN parsing: two-line plates, brand/slogan rejection, O/I misread correction. |
| Match against the claim | Pure logic (`truck_extract_match`) | Confusion-aware — a smudged 0/O, 1/I/L, 5/S, etc. that's visually consistent with the claim still counts as a match. |
| Read the plate colour | **HSV pixel classifier** | White (private) / yellow (commercial) / green (EV) / black (rental) / red (temporary/govt). |

### 3. Does the vehicle match the claimed make (and model)?
| Sub-check | Model | Notes |
|---|---|---|
| Read the manufacturer brand | **SigLIP 2** (zero-shot) **and** **AWS Rekognition** (painted brand-text) | Two independent votes — matches if EITHER agrees with the claim. (SigLIP alone missed 2 of 3 real Tata trucks in testing; Rekognition's real brand-text read catches what SigLIP misses.) |
| Read the model designation | **Claude** | Only enforced (able to reject) if read with ≥90% confidence — a low-confidence guess isn't trusted enough to hold against the vehicle. Real alternative (Qwen2.5-VL) is wired but not run live (needs a GPU box). |

Checks 2 and 3 both run once check 1 passes — neither gates the other, so a single call
always returns the complete picture for a manual-review UI, rather than stopping at the
first mismatch.

## Overall decision

- **PASS** — the front-image gate passes, and both the VRN and make(/model) checks match.
- **REJECT** — the front-image gate fails a hard rule, OR the VRN mismatches, OR the make
  (or a confidently-read model) mismatches.
- **MANUAL_REVIEW** — any check couldn't produce a confident result (unreadable plate,
  a possibly-AI-generated image below the reject threshold, a technical failure in one
  of the model calls) and needs a human decision.

When more than one downstream check disagrees, the worst outcome wins:
`REJECT` > `MANUAL_REVIEW` > `PASS`.

## Where this is implemented

`vfiv.validate_upload(image, claimed_vrn, claimed_make, claimed_model=None)` —
see `src/vfiv/validators/combined.py` for the orchestration and `README.md` for the
full technical detail on every model and its known limitations.
