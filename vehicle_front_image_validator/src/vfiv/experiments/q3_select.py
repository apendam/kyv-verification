"""Selectable Q3 backends — for the test/inference interface only; production
(``validators/make_model_check.py``) is unaffected and always uses
``"siglip_rekognition"`` for make + ``"claude"`` for model.

Make backends (each collects one or more VOTES: (source_name, extracted_make, confidence)):
  "siglip_rekognition" — production default's two real-model votes
                         (``make_model_check.classify_make`` — SigLIP always, Rekognition
                         best-effort).
  "claude"    — the ORIGINAL all-Claude Q3 make+model read (``legacy_prompts.Q3_LEGACY_PROMPT``),
                make portion only.
  "gemini"    — the same single-call design as "claude", via Gemini instead.
  "gcv_logo"  — Google Cloud Vision Logo Detection (``backends/google_vision.py``).
  "clarifai"  — Clarifai logo-recognition model (``backends/clarifai_backend.py``).

Model backends:
  "claude" (production default, ``make_model_check.classify_model``) | "gemini"

Decisioning (``decide_make_multi``) reuses ``truck_extract_match.make.aliases.match_make``
per vote and ORs across however many votes were actually collected — generalizes
``make_model_check.decide_make_model``'s hardcoded two-source logic to N sources, so any
combination of the backends above can be compared.
"""
from truck_extract_match.core import VerificationStatus, decide_status
from truck_extract_match.make.aliases import match_make

from vfiv import config
from vfiv.backends import clarifai_backend, google_vision
from vfiv.backends.gemini import call_gemini_json
from vfiv.experiments.legacy_prompts import Q3_LEGACY_PROMPT
from vfiv.validators.base import call_vlm_json
from vfiv.validators.make_model_check import classify_make as _classify_make_siglip_rekognition
from vfiv.validators.make_model_check import classify_model as _classify_model_claude

Q3_MAKE_BACKENDS = ["siglip_rekognition", "claude", "gemini", "gcv_logo", "clarifai"]
Q3_MODEL_BACKENDS = ["claude", "gemini"]


def collect_make_votes(
    image, backend: str = "siglip_rekognition", gemini_model: str | None = None,
) -> tuple[list[tuple[str, str, float]], list[dict[str, str]]]:
    """Returns (votes, errors). ``votes`` is [(source_name, extracted_make,
    confidence_0_100), ...] — one vote for most backends, two for
    "siglip_rekognition" (its own two internal sources). ``errors`` is
    [{"source": ..., "error": ...}, ...] for any backend call that genuinely failed
    (``checked=False`` — bad/unenabled credentials, network error, etc.) — kept
    SEPARATE from "checked fine, found nothing" (an empty ``make`` with no error),
    so a real backend failure (e.g. an unenabled GCP API) doesn't silently look
    identical to "no logo/brand text visible in the image".

    ``gemini_model`` overrides ``config.GEMINI_MODEL_Q3_MAKE`` for this call only —
    ignored unless ``backend == "gemini"``."""
    if backend == "siglip_rekognition":
        r = _classify_make_siglip_rekognition(image)
        votes = [("siglip", r["make_siglip"], r["make_siglip_confidence"])]
        if r.get("make_rekognition"):
            votes.append(("rekognition", r["make_rekognition"], 100.0))
        return votes, []
    if backend == "claude":
        r = call_vlm_json(image, Q3_LEGACY_PROMPT, config.VLM_MODEL)
        if not r.get("checked"):
            return [], [{"source": "claude", "error": r.get("error", "unavailable")}]
        return [("claude", str(r.get("make", "") or "").strip(),
                float(r.get("make_confidence", 0)))], []
    if backend == "gemini":
        r = call_gemini_json(image, Q3_LEGACY_PROMPT, model=gemini_model or config.GEMINI_MODEL_Q3_MAKE)
        if not r.get("checked"):
            return [], [{"source": "gemini", "error": r.get("error", "unavailable")}]
        return [("gemini", str(r.get("make", "") or "").strip(),
                float(r.get("make_confidence", 0)))], []
    if backend == "gcv_logo":
        r = google_vision.classify_logo(image)
        if not r.get("checked"):
            return [], [{"source": "gcv_logo", "error": r.get("error", "unavailable")}]
        if not r.get("make"):
            return [], []
        return [("gcv_logo", r["make"], r["make_confidence"])], []
    if backend == "clarifai":
        r = clarifai_backend.classify_logo(image)
        if not r.get("checked"):
            return [], [{"source": "clarifai", "error": r.get("error", "unavailable")}]
        if not r.get("make"):
            return [], []
        return [("clarifai", r["make"], r["make_confidence"])], []
    raise ValueError(f"unknown Q3 make backend: {backend!r} (expected one of {Q3_MAKE_BACKENDS})")


def classify_model(image, backend: str = "claude", gemini_model: str | None = None) -> dict:
    """Returns {"checked", "model", "model_confidence", "reason"}. ``gemini_model``
    overrides ``config.GEMINI_MODEL_Q3_MODEL`` for this call only — ignored unless
    ``backend == "gemini"``."""
    if backend == "claude":
        return _classify_model_claude(image)
    if backend == "gemini":
        r = call_gemini_json(image, Q3_LEGACY_PROMPT, model=gemini_model or config.GEMINI_MODEL_Q3_MODEL)
        if not r.get("checked"):
            return r
        return {"checked": True, "model": str(r.get("model", "") or "").strip(),
                "model_confidence": float(r.get("model_confidence", 0)), "reason": r.get("reason", "")}
    raise ValueError(f"unknown Q3 model backend: {backend!r} (expected one of {Q3_MODEL_BACKENDS})")


def decide_make_multi(votes: list[tuple[str, str, float]], claimed_make: str,
                      errors: list[dict[str, str]] | None = None) -> dict:
    """Generalized N-source OR-match decision — mirrors
    ``make_model_check.decide_make_model``'s two-source logic without being hardcoded
    to siglip/rekognition specifically. ``errors`` (from ``collect_make_votes``) are
    passed through verbatim so a genuine backend failure (bad/unenabled credentials,
    network error) is visible in the result rather than looking identical to "checked
    fine, found no logo/brand text". Returns {"make_status": ..., "matched_via":
    [source, ...] | None, "votes": [{...}, ...], "errors": [{...}, ...]}."""
    errors = errors or []
    if not votes:
        return {"make_status": VerificationStatus.UNREADABLE.value, "matched_via": None,
                "votes": [], "errors": errors}

    vote_details = []
    matched_sources = []
    any_extracted = False
    for source, extracted, confidence in votes:
        if extracted:
            any_extracted = True
        mm = match_make(extracted, claimed_make)
        vote_details.append({"source": source, "extracted": extracted, "confidence": confidence,
                             "matched": mm.matched, "brands": sorted(mm.extracted_brands)})
        if mm.matched:
            matched_sources.append(source)

    matched = bool(matched_sources)
    status = decide_status(matched, "x" if any_extracted else None)
    return {"make_status": status.value, "matched_via": matched_sources or None,
            "votes": vote_details, "errors": errors}
