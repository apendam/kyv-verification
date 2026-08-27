"""Selectable Q1 backends — for the test/inference interface only; production
(``front_image/front_image.py``) is unaffected and always uses ``"real_cv"``.

  "real_cv" — production default: YOLO+SigLIP real CV gate + narrowed Claude judgment
              prompt (``front_image.classify_front_image``, unchanged).
  "claude"  — the ORIGINAL all-Claude Q1 (before the real-model rewrite): one Claude
              call does vehicle/view/complete AND the judgment calls together
              (``legacy_prompts.Q1_LEGACY_PROMPT``).
  "gemini"  — the same single-call design as "claude", via Gemini instead.

All three produce the same dict shape ``front_image.decide_front_image`` expects, so
decisioning is identical regardless of backend — only the read step differs.
"""
from vfiv import config
from vfiv.backends.gemini import call_gemini_json
from vfiv.experiments.legacy_prompts import Q1_LEGACY_PROMPT
from vfiv.base import call_vlm_json
from vfiv.front_image.front_image import classify_front_image as _classify_front_image_real_cv

Q1_BACKENDS = ["real_cv", "claude", "gemini"]


def _parse_legacy(r: dict) -> dict:
    return {
        "checked": True,
        "vehicle_type": str(r.get("vehicle_type", "other")).lower(),
        "view": str(r.get("view", "other")).lower(),
        "is_front": bool(r.get("is_front")),
        "front_complete": bool(r.get("front_complete")),
        "confidence": float(r.get("confidence", 0)),
        "is_screenshot": bool(r.get("is_screenshot")),
        "is_photo_of_photo": bool(r.get("is_photo_of_photo")),
        "ai_generated": bool(r.get("ai_generated")),
        "ai_confidence": float(r.get("ai_confidence", 0)),
        "reason": r.get("reason", ""),
    }


def classify_q1(image, backend: str = "real_cv", gemini_model: str | None = None) -> dict:
    """``gemini_model`` overrides ``config.GEMINI_MODEL_Q1`` for this call only (e.g.
    from the interface's model picker) — ignored unless ``backend == "gemini"``."""
    if backend == "real_cv":
        return _classify_front_image_real_cv(image)
    if backend == "claude":
        r = call_vlm_json(image, Q1_LEGACY_PROMPT, config.VLM_MODEL)
        return _parse_legacy(r) if r.get("checked") else r
    if backend == "gemini":
        r = call_gemini_json(image, Q1_LEGACY_PROMPT, model=gemini_model or config.GEMINI_MODEL_Q1)
        return _parse_legacy(r) if r.get("checked") else r
    raise ValueError(f"unknown Q1 backend: {backend!r} (expected one of {Q1_BACKENDS})")
