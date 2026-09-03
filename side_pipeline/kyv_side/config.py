"""Defaults + thresholds, all overridable per call/CLI flag -- nothing here is
a hardcoded model choice baked into the request logic itself.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- credentials --------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- models ---------------------------------------------------------------
DEFAULT_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "anthropic/claude-sonnet-5")

# Local vector-embedding model for the duplicate check's second signal (see
# siglip.py / duplicate.py) -- a HuggingFace id, downloaded and run locally.
SIGLIP_MODEL = os.environ.get("KYV_SIGLIP_MODEL", "google/siglip2-base-patch16-512")
SIGLIP_DEVICE = os.environ.get("KYV_SIGLIP_DEVICE", "cuda")  # resolves to mps/cpu when unavailable

# --- storage ----------------------------------------------------------------
DEFAULT_DB_PATH = Path(os.environ.get("KYV_SIDE_DB_PATH", "kyv_side_checks.sqlite3"))

# --- thresholds (tune against your own labeled data before trusting these) ---
VRN_MAX_CONFUSABLE_EDITS = 1
VRN_SIMILAR_CHAR_MAX_DISTANCE = 3

# Same pHash -> SigLIP cascade as the front-image flow, applied to image_type
# "side" -- see duplicate.py for the full explanation. Both thresholds are
# starting points, not calibrated values.
DUPLICATE_HAMMING_MAX = 10
DUPLICATE_SIGLIP_SIMILARITY_MIN = float(os.environ.get("KYV_DUPLICATE_SIGLIP_SIMILARITY_MIN", "0.97"))

# --- HTTP behaviour ----------------------------------------------------------
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_S = 2.0
RETRY_BACKOFF_MAX_S = 30.0
