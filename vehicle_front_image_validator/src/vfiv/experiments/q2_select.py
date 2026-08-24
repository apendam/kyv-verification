"""Selectable Q2 backends — for the test/inference interface only; production
(``validators/vrn_check.py``) is unaffected and always uses ``"rekognition"``.

  "rekognition" — production default: AWS Rekognition + HSV colour classifier
                  (``vrn_check.classify_plate``, unchanged).
  "claude"      — the ORIGINAL all-Claude Q2 (before the Rekognition rewrite): one
                  Claude call reads the plate + colour together
                  (``legacy_prompts.Q2_LEGACY_PROMPT``).
  "gemini"      — the same single-call design as "claude", via Gemini instead.

All three produce the same dict shape ``vrn_check.decide_vrn`` expects, so decisioning
is identical regardless of backend — only the read step differs.
"""
from vfiv import config
from vfiv.backends.gemini import call_gemini_json
from vfiv.experiments.legacy_prompts import Q2_LEGACY_PROMPT
from vfiv.validators.base import call_vlm_json
from vfiv.validators.vrn_check import classify_plate as _classify_plate_rekognition

Q2_BACKENDS = ["rekognition", "claude", "gemini"]


def _parse_legacy(r: dict) -> dict:
    return {
        "checked": True,
        "plate": str(r.get("plate", "") or "").strip(),
        "plate_confidence": float(r.get("plate_confidence", 0)),
        "plate_colour": str(r.get("plate_colour", "unknown")).lower(),
        "colour_confidence": float(r.get("colour_confidence", 0)),
        "reason": r.get("reason", ""),
    }


def classify_q2(image, backend: str = "rekognition", gemini_model: str | None = None) -> dict:
    """``gemini_model`` overrides ``config.GEMINI_MODEL_Q2`` for this call only —
    ignored unless ``backend == "gemini"``."""
    if backend == "rekognition":
        return _classify_plate_rekognition(image)
    if backend == "claude":
        r = call_vlm_json(image, Q2_LEGACY_PROMPT, config.VLM_MODEL)
        return _parse_legacy(r) if r.get("checked") else r
    if backend == "gemini":
        r = call_gemini_json(image, Q2_LEGACY_PROMPT, model=gemini_model or config.GEMINI_MODEL_Q2)
        return _parse_legacy(r) if r.get("checked") else r
    raise ValueError(f"unknown Q2 backend: {backend!r} (expected one of {Q2_BACKENDS})")
