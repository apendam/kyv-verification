"""Model catalog: a JSON file you maintain by hand (``models.json``, repo
root of this package) that feeds the vision/embedding-model dropdowns in the
webapp, plus a live lookup against OpenRouter's own model list so a typed-in
model id can be confirmed to exist — and see its real pricing/modalities —
before you point a run at it.

``fetch_model_info`` calls ``GET {OPENROUTER_BASE_URL}/models``, the same
catalog endpoint OpenRouter's own model picker at openrouter.ai/models reads
from. It's a public, unauthenticated GET (no API key spent, no cost) that
returns every model OpenRouter proxies, each entry carrying pricing
(`pricing.prompt` / `.completion` / `.image` — USD per token, as strings) and
`architecture.input_modalities` (e.g. ``["text", "image"]``), which is how
"is this vision-capable" / "does this do embeddings" gets inferred when you
add a verified model to the catalog.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from . import config

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "models.json"


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("models", [])


def save_catalog(models: list[dict[str, Any]], path: str | Path = DEFAULT_CATALOG_PATH) -> None:
    Path(path).write_text(json.dumps({"models": models}, indent=2) + "\n")


def vision_models(path: str | Path = DEFAULT_CATALOG_PATH) -> list[str]:
    return [m["id"] for m in load_catalog(path) if m.get("vision")]


def embed_models(path: str | Path = DEFAULT_CATALOG_PATH) -> list[str]:
    return [m["id"] for m in load_catalog(path) if m.get("embedding")]


def list_all_models() -> list[dict[str, Any]]:
    """The full live catalog from OpenRouter's GET /models — every entry
    carries the exact `id` OpenRouter's chat-completions API expects
    (always ``provider/lowercase-slug``, e.g. ``qwen/qwen3-30b-a3b``), which
    is usually NOT the same string as the bold display name shown on
    openrouter.ai/models (e.g. "Qwen: Qwen3 30B A3B").
    """
    resp = requests.get(f"{config.OPENROUTER_BASE_URL}/models", timeout=config.REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_model_info(model_id: str, all_models: list[dict[str, Any]] | None = None
                      ) -> dict[str, Any] | None:
    """Looks `model_id` up by an EXACT match against OpenRouter's live model
    catalog (case-sensitive, same string the chat-completions API needs).
    Returns the matching entry (id, pricing, architecture,
    supported_parameters, ...) or None if OpenRouter doesn't know that
    literal id — that's the "verify" step: a typed-in model name is only
    added to models.json once this finds it. Pass `all_models` (from
    `list_all_models()`) to reuse an already-fetched catalog instead of
    hitting the network again.
    """
    for entry in all_models if all_models is not None else list_all_models():
        if entry.get("id") == model_id:
            return entry
    return None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_similar_models(model_id: str, all_models: list[dict[str, Any]] | None = None,
                         limit: int = 6) -> list[dict[str, Any]]:
    """A forgiving fallback for when `fetch_model_info` misses: matches
    against both `id` and the human-readable `name` with case/punctuation
    stripped out, so e.g. "Qwen3.6-35B-A3B" still surfaces
    "qwen/qwen3-30b-a3b" ("Qwen: Qwen3 30B A3B") as a likely candidate even
    though neither the case, the punctuation, nor the exact version number
    matches. Pure substring containment, not fuzzy edit-distance — good
    enough to catch "used the display name" / "wrong case" / "wrong
    separator", not meant to correct a genuinely different model number.
    """
    query = _normalize(model_id)
    if not query:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for entry in all_models if all_models is not None else list_all_models():
        norm_id = _normalize(entry.get("id", ""))
        norm_name = _normalize(entry.get("name", ""))
        if query in norm_id or query in norm_name:
            scored.append((0, entry))  # query is a substring of the real id/name
        elif norm_id in query or norm_name in query:
            scored.append((1, entry))  # real id/name is a substring of the query
    scored.sort(key=lambda pair: pair[0])
    return [entry for _, entry in scored[:limit]]


def add_to_catalog(model_id: str, *, label: str | None, vision: bool, embedding: bool,
                    path: str | Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    """Adds (or updates, if `model_id` is already present) one entry and
    rewrites models.json. Returns the full catalog after the change.
    """
    models = load_catalog(path)
    entry = {"id": model_id, "label": label or model_id, "vision": vision, "embedding": embedding}
    for i, m in enumerate(models):
        if m["id"] == model_id:
            models[i] = entry
            break
    else:
        models.append(entry)
    save_catalog(models, path)
    return models
