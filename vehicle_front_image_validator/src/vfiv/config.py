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
