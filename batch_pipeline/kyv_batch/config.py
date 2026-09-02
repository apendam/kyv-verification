"""Defaults + thresholds, all overridable per call/CLI flag — nothing here is a
hardcoded model choice baked into the request logic itself.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- credentials --------------------------------------------------------------
# Never hardcode the key. Loaded from the environment; scripts/*.py also load a
# local .env (via python-dotenv) before reading this, so `OPENROUTER_API_KEY=...`
# in a gitignored .env file works without exporting it in your shell.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- models ---------------------------------------------------------------
# "<provider>/<model-slug>" per OpenRouter's model-id format. Override with
# --model on either script, or this env var, without touching code. Pick any
# vision-capable model from https://openrouter.ai/models.
DEFAULT_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "anthropic/claude-sonnet-5")

# Local vector-embedding model for the duplicate check's second signal (see
# siglip.py / duplicate.py) — not an OpenRouter model id, a HuggingFace one,
# downloaded and run locally the first time it's needed. Same weights
# vehicle_front_image_validator/ already uses for its own duplicate check.
SIGLIP_MODEL = os.environ.get("KYV_SIGLIP_MODEL", "google/siglip2-base-patch16-512")
SIGLIP_DEVICE = os.environ.get("KYV_SIGLIP_DEVICE", "cuda")  # resolves to mps/cpu when unavailable

# --- storage ----------------------------------------------------------------
DEFAULT_DB_PATH = Path(os.environ.get("KYV_DB_PATH", "kyv_checks.sqlite3"))

# --- thresholds (tune against your own labeled data before trusting these) ---
# VRN confusion-aware matching, reused unchanged from truck_extract_match.
VRN_MAX_CONFUSABLE_EDITS = 1
# A mismatch whose confusable-edit-distance is still this small is treated as
# "probably a smudge/misread, not a different plate" -> manual review rather
# than an outright reject. Above this, it's read as a genuinely different VRN.
VRN_SIMILAR_CHAR_MAX_DISTANCE = 3

# Duplicate-photo threshold: only flag a hit when the perceptual-hash (pHash)
# Hamming distance is at or below this AND the claimed VRN differs from the
# matched upload's (an honest re-upload under the *same* VRN is never
# flagged) — same rule the existing pgvector-based duplicate_check.py in
# vehicle_front_image_validator/ uses, applied to a different similarity
# metric. Lower = more similar; 0 means the two 64-bit hashes are identical.
# 10 is a commonly-cited "near duplicate" cutoff for this hash size, but
# treat it as a starting point, not a calibrated value — tune against your
# own labeled pairs.
DUPLICATE_HAMMING_MAX = 10

# Second duplicate-detection signal, only run when pHash doesn't already flag
# a duplicate (SigLIP catches near-duplicates pHash misses — a re-crop, a
# lighting/angle change — but is much more expensive to run, so it's a
# fallback, not a first pass). Same 0.97 cosine-similarity starting point
# vehicle_front_image_validator/'s own SigLIP-based duplicate check uses —
# equally uncalibrated here; tune against your own labeled pairs.
DUPLICATE_SIGLIP_SIMILARITY_MIN = float(os.environ.get("KYV_DUPLICATE_SIGLIP_SIMILARITY_MIN", "0.97"))

# --- HTTP behaviour ----------------------------------------------------------
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_S = 2.0  # exponential: base * 2**attempt, capped below
RETRY_BACKOFF_MAX_S = 30.0
