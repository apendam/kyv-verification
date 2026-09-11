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


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (Levenshtein + adjacent-transposition
    as one edit, e.g. "chatgtp" -> "chatgpt" is distance 1, not 2) -- catches
    the common fat-finger typo/rename case, not just exact matches."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[la][lb]


def _fuzzy_tolerance(marker: str) -> int:
    """Edit-distance budget for a marker, scaled to its length. Short/generic
    markers (e.g. "dalle", 5 chars) get zero slack -- fuzzy matching those
    would trade a handful of real catches for a lot of noise, since plenty of
    unrelated words sit one edit away. Longer, more distinctive markers get
    1-2 edits of slack to survive a typo'd rename."""
    n = len(marker)
    if n < 6:
        return 0
    if n < 10:
        return 1
    return 2


def _fuzzy_contains(haystack: str, needle: str, max_dist: int) -> bool:
    if max_dist == 0:
        return needle in haystack
    n = len(needle)
    for length in range(max(1, n - max_dist), n + max_dist + 1):
        for start in range(0, len(haystack) - length + 1):
            if _edit_distance(haystack[start:start + length], needle) <= max_dist:
                return True
    return False

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
    re.compile(r"^IMG-\d{8}-WA\d+\.jpe?g$", re.IGNORECASE),          # WhatsApp mobile -- saved from a chat
    re.compile(r"^WhatsApp Image \d{4}-\d{2}-\d{2} at \d{2}\.\d{2}\.\d{2}(?: \(\d+\))?\.jpe?g$", re.IGNORECASE),  # WhatsApp Desktop/Web -- downloaded from a chat
)


def matches_ai_generator_filename(filename: str) -> bool:
    """True if `filename` (just the name, not a full path) looks like a
    default export from a known AI image generator/editor, allowing for a
    typo'd or lightly-altered rename (see _fuzzy_tolerance) -- not just an
    exact, unedited default."""
    normalized = _normalize(filename)
    return any(
        _fuzzy_contains(normalized, marker, _fuzzy_tolerance(marker))
        for marker in AI_GENERATOR_FILENAME_MARKERS
    )


def matches_known_camera_filename(filename: str) -> bool:
    """True if `filename` matches a known real-camera, screenshot, or
    WhatsApp-relay naming convention. Purely informational for the audit
    trail -- NOT matching this does not imply anything suspicious, since
    plenty of legitimate upload paths rename files."""
    return any(pattern.match(filename) for pattern in CAMERA_FILENAME_PATTERNS)
