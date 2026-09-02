"""Model catalog: a JSON file you maintain by hand (``models.json``, repo
root of this package) that feeds the vision/embedding-model dropdowns in the
webapp, plus a live lookup against OpenRouter's own model lists so a typed-in
model id can be confirmed to exist — and see its real pricing/modalities —
before you point a run at it.

OpenRouter splits its catalog across TWO separate endpoints, and a model
only ever appears in one of them:

  - ``GET /models`` — chat-completions models (vision models included).
    Public, unauthenticated, no API key spent. Embedding models are NOT
    in this list at all, however image-capable they are described elsewhere.
  - ``GET /embeddings/models`` — embedding models only. Requires
    Authorization like any other call. Confirmed by hitting it directly:
    the chat-completions catalog returned zero hits for "embed" across
    its ~400 entries, while this endpoint is where e.g.
    ``google/gemini-embedding-2`` (one of the few embedding models that
    accepts image input, per OpenRouter's own model page) actually lives.

Verifying/fuzzy-matching a model id checks both lists together (see
`list_all_models`), so a miss is a real miss, not an artifact of asking
the wrong catalog.
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


def list_chat_models() -> list[dict[str, Any]]:
    """The chat-completions catalog from GET /models — every entry carries
    the exact `id` OpenRouter's API expects (always
    ``provider/lowercase-slug``, e.g. ``qwen/qwen3-30b-a3b``), which is
    usually NOT the same string as the bold display name shown on
    openrouter.ai/models (e.g. "Qwen: Qwen3 30B A3B"). Does NOT include
    embedding models — see `list_embedding_models`.
    """
    resp = requests.get(f"{config.OPENROUTER_BASE_URL}/models", timeout=config.REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json().get("data", [])


def list_embedding_models() -> list[dict[str, Any]]:
    """The separate embeddings catalog from GET /embeddings/models — unlike
    /models, this one needs Authorization. Raises if OPENROUTER_API_KEY
    isn't set; callers that want a best-effort combined list should catch
    that rather than let a missing key hide the chat-models half too.
    """
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set — required for GET /embeddings/models")
    resp = requests.get(
        f"{config.OPENROUTER_BASE_URL}/embeddings/models",
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
        params={"limit": 1000}, timeout=config.REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def list_all_models() -> list[dict[str, Any]]:
    """Chat-completions + embedding models combined — the right list to
    verify or fuzzy-match a typed-in id against, since a real model lives in
    exactly one of the two catalogs and a caller shouldn't need to know
    which. Embedding-catalog errors (e.g. no API key yet) are swallowed
    here rather than raised, so a chat-model id can still be verified even
    when the embeddings half is unavailable.
    """
    chat_models = list_chat_models()
    try:
        embed_models_live = list_embedding_models()
    except Exception:  # noqa: BLE001 - best-effort; chat_models alone still verifies vision ids
        embed_models_live = []
    return chat_models + embed_models_live


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
