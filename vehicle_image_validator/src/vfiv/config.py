import os

VLM_MODEL = os.environ.get("VFIV_MODEL", "claude-sonnet-4-6")

# Front-image gate (Q1) thresholds.
FRONT_CONF_MIN = 60.0
FRONT_AI_REJECT_CONF = 85.0

# VRN check (Q2) — max confusable (smudge-tolerant) edit distance to the claimed VRN.
VRN_MAX_CONFUSABLE_EDITS = 1

# Make/model check (Q3). Make comparison always runs. Model comparison only runs (and
# can REJECT) when the model read's own confidence is >= MODEL_CONF_MIN — below that,
# the read is reported but not trusted enough to hold against the vehicle.
MODEL_CONF_MIN = 90.0
MODEL_MATCH_MIN = 0.75  # fuzzy token-set-ratio threshold for the model string match

# --- Real-model backends (see backends/) -------------------------------------

DEVICE = os.environ.get("VFIV_DEVICE", "cuda")  # resolves to mps/cpu when unavailable

# YOLOv8 vehicle detector (Q1). No custom-trained weights here -> falls back to the
# downloadable COCO yolov8n.pt (same dev fallback truck_front_extractor documents;
# weak on Indian trucks — swap in a truck-finetuned detector in production).
YOLO_VEHICLE_WEIGHTS = os.environ.get("VFIV_YOLO_VEHICLE_WEIGHTS", "yolov8n.pt")

# SigLIP 2 — shared zero-shot classifier for pose (Q1) and make (Q3). Already cached
# locally under ~/.cache/huggingface from prior work in this environment.
SIGLIP_MODEL = os.environ.get("VFIV_SIGLIP_MODEL", "google/siglip2-base-patch16-512")

# Q1 gate thresholds (mirrors truck_front_extractor's GateThresholds defaults).
GATE_TRUCK_MIN = 0.5        # YOLO vehicle objectness*classprob to consider a truck present
GATE_FRONTAL_MIN = 0.6      # SigLIP pose head P(front)+P(front34)
GATE_COVERAGE_MIN = 0.25    # truck bbox area / frame area for "complete"
GATE_ACCEPT_MIN = 0.7       # fused gate confidence required to pass

# AWS Rekognition (Q2 VRN). Region falls back to the AWS_DEFAULT_REGION/profile default
# when unset; credentials come from the standard AWS env vars / profile chain.
AWS_REKOGNITION_REGION = os.environ.get("VFIV_AWS_REGION")  # None -> boto3 default

# Qwen2.5-VL (Q3 exact model, opt-in). NOT run live in dev — ~16GB weights, minutes/image
# on CPU. Real code is wired (backends/qwen.py); default path stays Claude until a GPU
# box is available.
QWEN_MODEL = os.environ.get("VFIV_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# --- Experimentation backends (test/inference interface only — see experiments/) ----
# None of these have credentials configured in this environment; each degrades
# gracefully ("not configured") rather than crashing, same as backends/qwen.py.

GEMINI_MODEL = os.environ.get("VFIV_GEMINI_MODEL", "gemini-2.5-flash")  # global default/fallback
# Per-stage override — lets you use e.g. Flash for the cheap Q1/Q2 reads and Pro for
# Q3's make/model, or any other mix, without touching code. Each falls back to the
# global VFIV_GEMINI_MODEL above when unset.
GEMINI_MODEL_Q1 = os.environ.get("VFIV_GEMINI_MODEL_Q1", GEMINI_MODEL)
GEMINI_MODEL_Q2 = os.environ.get("VFIV_GEMINI_MODEL_Q2", GEMINI_MODEL)
GEMINI_MODEL_Q3_MAKE = os.environ.get("VFIV_GEMINI_MODEL_Q3_MAKE", GEMINI_MODEL)
GEMINI_MODEL_Q3_MODEL = os.environ.get("VFIV_GEMINI_MODEL_Q3_MODEL", GEMINI_MODEL)

# Two auth modes, tried in this order (see backends/gemini.py):
#   1. GEMINI_API_KEY            — Gemini Developer API / AI Studio (simple API key)
#   2. GOOGLE_APPLICATION_CREDENTIALS + GEMINI_VERTEX_PROJECT
#                                 — Vertex AI (service-account JSON), same credentials
#                                   env var already used for GCV Logo Detection below.
#      GEMINI_VERTEX_LOCATION defaults to us-central1 if unset.
GEMINI_VERTEX_PROJECT = os.environ.get("GEMINI_VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
GEMINI_VERTEX_LOCATION = os.environ.get("GEMINI_VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Google Cloud Vision Logo Detection (Q3 make, alternate). Needs
# GOOGLE_APPLICATION_CREDENTIALS (service-account JSON path).

# Clarifai logo-recognition model (Q3 make, alternate). Needs CLARIFAI_API_KEY (a
# Personal Access Token). Model reference defaults to Clarifai's general public logo
# model — adjust to whatever logo-recognition model you actually have access to.
CLARIFAI_USER_ID = os.environ.get("VFIV_CLARIFAI_USER_ID", "clarifai")
CLARIFAI_APP_ID = os.environ.get("VFIV_CLARIFAI_APP_ID", "main")
CLARIFAI_MODEL_ID = os.environ.get("VFIV_CLARIFAI_MODEL_ID", "logo")

# --- Duplicate / photo-reuse detection (cross-upload, not one of Q1/Q2/Q3) ----------
# NOT wired into validate_upload/CombinedResult yet — see duplicate_check.py.
# Reuses the same SigLIP 2 weights already loaded above (SIGLIP_MODEL): one extra
# forward-pass step that stops at the raw image embedding instead of comparing it
# against a text prompt. No second model to load.

PGVECTOR_DSN = os.environ.get("VFIV_PGVECTOR_DSN")  # postgresql://user:pass@host/db ; unset -> not configured
PGVECTOR_TABLE = os.environ.get("VFIV_PGVECTOR_TABLE", "upload_embeddings")
# Must match SIGLIP_MODEL's actual image-embedding width — verify with
# `SigLipModel().embed_image(some_image).shape` before running ensure_schema() for
# the first time; 768 is the base-variant's typical projection dim, not a guarantee.
PGVECTOR_EMBED_DIM = int(os.environ.get("VFIV_PGVECTOR_EMBED_DIM", "768"))

# Cosine-similarity floor for "this upload's photo is a near-duplicate of a prior
# one" (1.0 = identical direction). UNCALIBRATED — a starting point only. Tune this
# against real labeled pairs from your own uploads (genuine near-duplicates vs.
# genuinely different trucks) before trusting it — see README's "Duplicate
# detection" section for why the raw number isn't meaningful on its own.
DUPLICATE_SIMILARITY_MIN = float(os.environ.get("VFIV_DUPLICATE_SIMILARITY_MIN", "0.97"))

# The image types the reference-image library distinguishes — front/side/fastag are
# unrelated corpora and are never compared against each other (see image_type in
# backends/vector_store.py). A convenience list for UI dropdowns/CLI choices, not
# an enforced constraint — any string is technically accepted as an image_type.
IMAGE_TYPES = ["front", "side", "fastag"]

# --- FASTag validator (fastag_image/fastag_check.py) ----------------------------
# Printed-digit OCR fuzz budget -- same idea as VRN_MAX_CONFUSABLE_EDITS, but only
# ever the last-resort match arm: the QR/barcode decodes are exact and checked first.
FASTAG_OCR_MAX_CONFUSABLE_EDITS = 1

# Which model reads the printed digits -- "rekognition" (default) | "claude" | "gemini".
# The barcode/QR decode itself is a deterministic algorithm either way, not a model
# call, so this only affects the OCR fallback arm. See backends/fastag_reader.py.
FASTAG_OCR_BACKEND = os.environ.get("VFIV_FASTAG_OCR_BACKEND", "rekognition")

# --- Side/axle-image validator (side_image/side_image_check.py) ----------------
# No dedicated axle/wheel detector is wired (would need a custom-trained model and
# a labeled dataset) -- axle count is a narrowed VLM judgment call instead, gated by
# its own reported confidence. Which model reads it is selectable -- "claude"
# (default) | "gemini" -- same prompt either way; see side_image_check.py.
AXLE_COUNT_CONF_MIN = float(os.environ.get("VFIV_AXLE_COUNT_CONF_MIN", "70.0"))
AXLE_COUNT_BACKEND = os.environ.get("VFIV_AXLE_COUNT_BACKEND", "claude")

# Which model classifies a side/axle photo into vrn_visible | corner_view |
# pure_side_profile (see side_image_check.py's module docstring). A VLM judgment
# call, not a zero-shot embedding comparison -- the earlier SigLIP-based
# classifier kept misrouting photos where the windshield/plate visibility cue
# that actually distinguishes the buckets was a small part of the frame, and
# rewording its text prompts couldn't fix that (changing the TEXT side of a
# zero-shot comparison can't change how the IMAGE itself embeds).
SIDE_IMAGE_TYPE_BACKEND = os.environ.get("VFIV_SIDE_IMAGE_TYPE_BACKEND", "claude")

# Cosine-similarity floor for "this corner-shot's vehicle crop looks like the same
# truck as the claimed vehicle's on-file front photo". UNCALIBRATED -- a general
# SigLIP embedding is trained for semantic similarity (what make/model is this),
# not individual-vehicle re-identification, so it may not reliably separate "same
# truck, different angle" from "different truck, same make/model/colour". Validate
# against real same-truck vs. same-model-different-truck pairs before trusting
# this threshold in production — see side_image_check.py's module docstring.
SIDE_IMAGE_SIMILARITY_MIN = float(os.environ.get("VFIV_SIDE_IMAGE_SIMILARITY_MIN", "0.97"))

# Colour-histogram correlation floor for the same corner-view/front-photo pair
# (see SIDE_IMAGE_SIMILARITY_MIN above) -- a real, deterministic paint-colour
# comparison (unlike the embedding check, no ML judgment call involved), but
# UNCALIBRATED for the same reason: lighting/exposure can differ meaningfully
# between two photos of the same truck shot at different times. Validate against
# real same-truck vs. different-truck pairs before trusting this threshold.
SIDE_IMAGE_COLOR_HIST_MIN = float(os.environ.get("VFIV_SIDE_IMAGE_COLOR_HIST_MIN", "0.8"))

# Floor for backends.gate.completeness_score (the same YOLO-bbox-vs-frame-edge
# heuristic Q1 uses, see GATE_ACCEPT_MIN above) applied to a side/axle photo --
# "is the truck cut off at a frame edge?" Deliberately much more lenient than
# Q1's own GATE_ACCEPT_MIN=0.7: a long truck/trailer shot from a normal standoff
# distance can legitimately run off the left/right edge more than a compact
# front-on shot would, so this only ever downgrades to MANUAL_REVIEW (see
# side_image_check.py's check_side_completeness), never a solo REJECT like Q1's
# own use of the same score. UNCALIBRATED -- a starting point only.
SIDE_IMAGE_COMPLETENESS_MIN = float(os.environ.get("VFIV_SIDE_IMAGE_COMPLETENESS_MIN", "0.5"))

# --- FASTag / side-image completeness ("is the whole thing actually in frame?") ----
# Which model checks whether the FULL FASTag sticker (QR + barcode + printed
# digits) is visible and not cut off/obscured -- "rekognition" isn't an option
# here (Rekognition has no notion of "FASTag sticker", only generic text/object
# detection), so this is Claude/Gemini only, same posture as
# SIDE_IMAGE_TYPE_BACKEND above. See fastag_check.py::check_fastag_completeness.
FASTAG_COMPLETENESS_BACKEND = os.environ.get("VFIV_FASTAG_COMPLETENESS_BACKEND", "claude")

# Confidence floor (0-100, the VLM's own self-reported confidence -- same idea as
# AXLE_COUNT_CONF_MIN) for trusting the sticker_complete verdict either way. Below
# this, the read is too uncertain to call it either complete or incomplete, so it
# falls back to MANUAL_REVIEW regardless of what sticker_complete says. UNCALIBRATED
# -- a starting point only; tune both this and SIDE_IMAGE_COMPLETENESS_MIN above
# against real photos of genuinely complete vs. cropped uploads before relying on
# either to move a real decision, and adjust per-call (both check_fastag_upload and
# check_side_image_upload accept these as overridable arguments, not just env vars).
FASTAG_COMPLETENESS_CONF_MIN = float(os.environ.get("VFIV_FASTAG_COMPLETENESS_CONF_MIN", "70.0"))
