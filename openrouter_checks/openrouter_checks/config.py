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
# --model / --embed-model on either script, or these env vars, without touching
# code. Pick any vision-capable model from https://openrouter.ai/models —
# these three are just sane, cheap-ish starting points.
DEFAULT_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "anthropic/claude-sonnet-5")
# nvidia/llama-nemotron-embed-vl-1b-v2 (the original default here) currently
# 404s with "No endpoints found" -- no provider serves it right now. Of the
# embedding models OpenRouter currently lists, google/gemini-embedding-2 is
# the one confirmed to accept image input (most embedding models are
# text-only and will reject the image content this app sends for the
# duplicate check) — see models.json / the Model Catalog tab for others.
DEFAULT_EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL", "google/gemini-embedding-2")

# --- storage ----------------------------------------------------------------
DEFAULT_DB_PATH = Path(os.environ.get("KYV_DB_PATH", "kyv_checks.sqlite3"))

# --- thresholds (tune against your own labeled data before trusting these) ---
# VRN confusion-aware matching, reused unchanged from truck_extract_match.
VRN_MAX_CONFUSABLE_EDITS = 1
# A mismatch whose confusable-edit-distance is still this small is treated as
# "probably a smudge/misread, not a different plate" -> manual review rather
# than an outright reject. Above this, it's read as a genuinely different VRN.
VRN_SIMILAR_CHAR_MAX_DISTANCE = 3

# Duplicate-photo threshold: only flag a hit when cosine similarity is at or
# above this AND the claimed VRN differs from the matched upload's (an honest
# re-upload under the *same* VRN is never flagged) — same rule the existing
# pgvector-based duplicate_check.py in vehicle_front_image_validator/ uses.
DUPLICATE_SIMILARITY_MIN = 0.97

# --- HTTP behaviour ----------------------------------------------------------
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_S = 2.0  # exponential: base * 2**attempt, capped below
RETRY_BACKOFF_MAX_S = 30.0
