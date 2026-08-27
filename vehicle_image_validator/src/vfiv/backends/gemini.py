"""Gemini 2.5 (Pro/Flash) VLM wrapper — an alternate/experimental backend for the
judgment-call and reading tasks Claude currently handles. NOT wired into production
(the production validator modules' default paths are unchanged) — this exists for the test/inference
interface's backend-selection (``experiments/``), so Claude vs Gemini vs the real CV
models can be compared side by side.

Supports TWO auth modes (tried in this order — see ``config.py``):
  1. ``GEMINI_API_KEY``      — Gemini Developer API / AI Studio (a short "AIzaSy..."
                               key), the simple case.
  2. ``GOOGLE_APPLICATION_CREDENTIALS`` (service-account JSON path) + a GCP project
                               (``GEMINI_VERTEX_PROJECT````GOOGLE_CLOUD_PROJECT``) —
                               Vertex AI's Gemini, using Application Default
                               Credentials (the same credentials env var already used
                               for Google Cloud Vision's Logo Detection).
If neither is configured, this degrades gracefully ("not configured"), the same way
``backends/qwen.py`` does. Mirrors ``base.py``'s ``call_vlm_json`` contract
exactly (``{"checked": True/False, ...}``) so it's a drop-in alternate for any
Claude-based ``classify_*`` function.
"""
from __future__ import annotations

import json
import os

from vfiv import config


def _make_client():
    """Returns a genai.Client using whichever auth mode is configured, or raises
    RuntimeError with a clear reason if neither is."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and config.GEMINI_VERTEX_PROJECT:
        return genai.Client(vertexai=True, project=config.GEMINI_VERTEX_PROJECT,
                            location=config.GEMINI_VERTEX_LOCATION)
    if creds_path and not config.GEMINI_VERTEX_PROJECT:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is set but no GCP project — "
                          "set GEMINI_VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT")
    raise RuntimeError("no GEMINI_API_KEY, and no GOOGLE_APPLICATION_CREDENTIALS "
                      "(for Vertex AI) configured")


def call_gemini_json(image, prompt: str, model: str | None = None) -> dict:
    """image: file path or PIL.Image. Returns {"checked": True, **parsed_json} on
    success, or {"checked": False, "error": "..."} if the SDK/credentials are missing
    or the call fails — same contract as ``base.py:call_vlm_json``."""
    try:
        client = _make_client()
    except Exception as e:
        return {"checked": False, "error": str(e)}

    try:
        from PIL import Image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB")

        resp = client.models.generate_content(
            model=model or config.GEMINI_MODEL,
            contents=[img, prompt],
        )
        txt = (resp.text or "").strip()
        parsed = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        return {"checked": True, **parsed}
    except Exception as e:
        return {"checked": False, "error": str(e)}
