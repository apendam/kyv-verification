"""Filename-based heuristic: checks the ORIGINAL uploaded filename against
known AI-generator export naming conventions and known real-camera/screenshot
naming conventions.

Weak and easily evaded -- a rename before upload defeats it entirely, and any
image relayed through WhatsApp gets rewritten to WhatsApp's own
IMG-YYYYMMDD-WAxxxx pattern regardless of its origin, erasing this signal
either way (WhatsApp forwarding is extremely common in exactly this
trucking/logistics workflow). Kept as one more MANUAL_REVIEW-only trigger,
never a reject gate, for the narrow case of a careless direct upload -- not
something to rely on.

This checks whatever filename is passed in, which must be the client-provided
original name, not a server-renamed storage path -- if your upload handler
saves incoming files under a generated name (e.g. the upload_id), thread the
original filename through separately rather than relying on the storage path.
"""
from __future__ import annotations

import re

# Default filenames used by popular AI image generators/editors, when
# unedited by the user. Matched against a normalized filename (lowercased,
# every non-alphanumeric character stripped -- see _normalize) so upload-host
# sanitization that swaps spaces/underscores for hyphens (confirmed in
# practice: postimg.cc rewrote "ChatGPT Image ....png" to
# "Chat-GPT-Image-....png" on upload, which a plain "chatgpt" substring check
# missed) doesn't defeat the match. Only ChatGPT and Gemini self-identify with
# a literal brand string by default; the others listed here (ComfyUI,
# DALL-E) are well-documented, fixed defaults -- Midjourney's own default
# ("<username>_<prompt>_<uuid>.png") deliberately isn't included here since it
# carries no fixed brand keyword to match on.
AI_GENERATOR_FILENAME_MARKERS = (
    "chatgpt", "dalle", "geminigeneratedimage", "comfyui", "firefly", "leonardo",
)


def _normalize(s: str) -> str:
    """Lowercase and strip every non-alphanumeric character, so separator
    choice (space/underscore/hyphen/none) can't affect a marker match."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

# Known real-camera / screenshot naming conventions, also unedited by the
# user. A match here is NOT evidence of anything by itself -- it's the common
# case. Listed for documentation/audit-trail purposes: anything that doesn't
# match the AI list above already proceeds through the normal flow, whether
# or not it matches one of these.
CAMERA_FILENAME_PATTERNS = (
    re.compile(r"^IMG_\d{4}\.(heic|jpe?g|png)$", re.IGNORECASE),     # iPhone photo AND screenshot (same IMG_ counter)
    re.compile(r"^IMG_\d{8}_\d{6}\.jpe?g$", re.IGNORECASE),          # Android (stock/AOSP) photo
    re.compile(r"^PXL_\d{8}_\d{6,9}.*\.jpe?g$", re.IGNORECASE),      # Google Pixel photo
    re.compile(r"^Screenshot_\d{8}-\d{6}.*\.png$", re.IGNORECASE),   # Android screenshot (Samsung appends an app name)
    re.compile(r"^IMG-\d{8}-WA\d+\.jpe?g$", re.IGNORECASE),          # WhatsApp-relayed image
)


def matches_ai_generator_filename(filename: str) -> bool:
    """True if `filename` (just the name, not a full path) looks like an
    unedited default export from a known AI image generator/editor."""
    normalized = _normalize(filename)
    return any(marker in normalized for marker in AI_GENERATOR_FILENAME_MARKERS)


def matches_known_camera_filename(filename: str) -> bool:
    """True if `filename` matches a known real-camera, screenshot, or
    WhatsApp-relay naming convention. Purely informational for the audit
    trail -- NOT matching this does not imply anything suspicious, since
    plenty of legitimate upload paths rename files."""
    return any(pattern.match(filename) for pattern in CAMERA_FILENAME_PATTERNS)
