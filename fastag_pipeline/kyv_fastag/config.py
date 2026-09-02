"""Defaults + thresholds, all overridable per call/CLI flag -- nothing here is
a hardcoded model choice baked into the request logic itself.
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
# --model on the script, or this env var, without touching code. This is the
# ONLY vision-model call in the whole flow (framing + tamper check) -- barcode
# and QR reading are deterministic local decodes, not a model call.
DEFAULT_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "anthropic/claude-sonnet-5")

# --- storage ----------------------------------------------------------------
DEFAULT_DB_PATH = Path(os.environ.get("KYV_FASTAG_DB_PATH", "kyv_fastag_checks.sqlite3"))

# --- HTTP behaviour ----------------------------------------------------------
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_S = 2.0  # exponential: base * 2**attempt, capped below
RETRY_BACKOFF_MAX_S = 30.0
