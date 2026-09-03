"""Thin OpenRouter client: chat completions with image input + structured JSON
output, embeddings, retry/backoff, and per-call token/cost accounting.

Verified against OpenRouter's current docs before writing this (request/response
shapes below, not guessed):
  - vision input: OpenAI-compatible `content` array with `image_url` data URIs
  - every chat completion response includes `usage.cost` — the real dollar
    amount billed to your account for that call, no separate endpoint needed
  - `response_format: {"type": "json_schema", ...}` constrains output, but
    support varies by model — callers should treat a parse failure as a
    technical failure, not assume every model honours the schema
  - errors come back as {"error": {"code": <int == http status>, "message": ...}}
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from . import config


class OpenRouterError(Exception):
    """Base class for every OpenRouter-call failure. Callers that want the
    "technical failure -> manual review" gate from the flowchart should catch
    this (not subclasses individually) unless they need to special-case one.
    """


class OpenRouterRateLimited(OpenRouterError):
    """429 after exhausting retries."""


class OpenRouterInsufficientCredits(OpenRouterError):
    """402 — out of funds. Never silently swallow this into "manual review":
    every subsequent call will fail identically until the account is topped up,
    so callers should let this propagate and stop the run.
    """


class OpenRouterBadResponse(OpenRouterError):
    """Got a 200 but couldn't parse a usable result out of it (malformed JSON,
    empty content, schema the model ignored, ...).
    """


@dataclass
class ChatResult:
    data: dict[str, Any]        # parsed JSON matching the requested schema
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    raw_content: str            # the raw text OpenRouter returned, for auditing


@dataclass
class EmbedResult:
    vector: list[float]
    model: str
    prompt_tokens: int
    cost_usd: float
    latency_ms: int


@dataclass
class SessionTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd


def _image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        mime = "image/jpeg"  # OpenRouter accepts png/jpeg/webp/gif; jpeg is a safe default
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout_s: float = config.REQUEST_TIMEOUT_S,
                 max_retries: int = config.MAX_RETRIES,
                 extra_headers: dict[str, str] | None = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        if not self.api_key:
            raise OpenRouterError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY in your "
                "environment or a local .env file (see .env.example)."
            )
        self.base_url = (base_url or config.OPENROUTER_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.totals = SessionTotals()
        self._session = requests.Session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional but recommended by OpenRouter for their own leaderboards/analytics.
        headers.update(extra_headers or {})
        self._session.headers.update(headers)

    # -- internals ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_error = exc
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 402:
                # Out of credits — never worth retrying, and shouldn't be treated
                # as a per-image technical failure (it'll fail identically on
                # every subsequent call until the account is topped up).
                raise OpenRouterInsufficientCredits(
                    f"OpenRouter reports insufficient credits: {resp.text[:300]}"
                )

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                last_error = OpenRouterError(f"{resp.status_code}: {resp.text[:300]}")
                if attempt < self.max_retries:
                    if retry_after:
                        try:
                            time.sleep(min(float(retry_after), config.RETRY_BACKOFF_MAX_S))
                            continue
                        except ValueError:
                            pass
                    self._sleep_backoff(attempt)
                    continue
                if resp.status_code == 429:
                    raise OpenRouterRateLimited(str(last_error))
                raise OpenRouterError(str(last_error))

            # 4xx other than 402/429 — not retryable (bad request, bad model id, ...)
            raise OpenRouterError(f"{resp.status_code}: {resp.text[:500]}")

        raise OpenRouterError(f"Exhausted retries calling {path}: {last_error}")

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = min(config.RETRY_BACKOFF_BASE_S * (2 ** attempt), config.RETRY_BACKOFF_MAX_S)
        time.sleep(delay)

    # -- public API ----------------------------------------------------------

    def chat_json(self, *, model: str, system_prompt: str, user_text: str,
                  image_paths: list[str | Path], json_schema: dict[str, Any],
                  schema_name: str, max_tokens: int = 1024) -> ChatResult:
        """One vision + structured-JSON chat completion. Raises OpenRouterError
        (or a subclass) on any failure — callers implement the flowchart's
        "check failed technically? -> manual review" gate by catching that.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(p)}})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
            },
        }

        started = time.monotonic()
        body = self._post("/chat/completions", payload)
        latency_ms = int((time.monotonic() - started) * 1000)

        try:
            choice = body["choices"][0]
            raw_content = choice["message"]["content"]
            if isinstance(raw_content, list):  # some models return content parts
                raw_content = "".join(
                    part.get("text", "") for part in raw_content if isinstance(part, dict)
                )
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterBadResponse(f"Unexpected response shape: {body}") from exc

        data = _extract_json(raw_content)
        if data is None:
            raise OpenRouterBadResponse(
                f"Model did not return parseable JSON for schema '{schema_name}': "
                f"{raw_content[:300]!r}"
            )

        usage = body.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost_usd = float(usage.get("cost", 0.0))
        self.totals.add(prompt_tokens, completion_tokens, cost_usd)

        return ChatResult(
            data=data, model=body.get("model", model),
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost_usd, latency_ms=latency_ms, raw_content=raw_content,
        )

    def embed(self, *, model: str, image_path: str | Path | None = None,
              text: str | None = None) -> EmbedResult:
        """Embed an image and/or text. At least one of image_path/text is required —
        pass both for a joint embedding on models that support it.

        For image input, OpenRouter's `input` field is an array of `{"content":
        [...]}` wrapper objects (content-part dicts nested under a "content"
        key), NOT a bare array of content-part dicts — sending the flat form
        (each part as a top-level array item) 400s with a Zod "invalid_union"
        error, since it matches neither the plain-string nor array-of-strings
        branch of the schema.
        """
        if image_path is None and text is None:
            raise ValueError("embed() needs image_path and/or text")

        if image_path is not None and text is not None:
            content: Any = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_path)}},
            ]
            inp: Any = [{"content": content}]
        elif image_path is not None:
            content = [{"type": "image_url", "image_url": {"url": _image_to_data_uri(image_path)}}]
            inp = [{"content": content}]
        else:
            inp = text

        payload = {"model": model, "input": inp}

        started = time.monotonic()
        body = self._post("/embeddings", payload)
        latency_ms = int((time.monotonic() - started) * 1000)

        try:
            vector = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterBadResponse(f"Unexpected embeddings response shape: {body}") from exc

        usage = body.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        cost_usd = float(usage.get("cost", 0.0))
        self.totals.add(prompt_tokens, 0, cost_usd)

        return EmbedResult(
            vector=vector, model=body.get("model", model),
            prompt_tokens=prompt_tokens, cost_usd=cost_usd, latency_ms=latency_ms,
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse the model's response as JSON, tolerating a ```json ... ``` fence or
    leading/trailing prose some models add even when a schema was requested.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
