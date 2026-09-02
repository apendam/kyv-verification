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


def fetch_model_info(model_id: str) -> dict[str, Any] | None:
    """Looks `model_id` up in OpenRouter's live model catalog. Returns the
    matching entry (id, pricing, architecture, supported_parameters, ...) or
    None if OpenRouter doesn't know that id — that's the "verify" step: a
    typed-in model name is only added to models.json once this finds it.
    """
    resp = requests.get(f"{config.OPENROUTER_BASE_URL}/models", timeout=config.REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    for entry in resp.json().get("data", []):
        if entry.get("id") == model_id:
            return entry
    return None


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
