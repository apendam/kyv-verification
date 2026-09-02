"""Embedding-based near-duplicate detection, built fresh on OpenRouter's
`/embeddings` endpoint (not the SigLIP+pgvector system already in
vehicle_front_image_validator/ — a deliberate choice to avoid a Postgres
deployment). Embeddings are stored as blobs in the same SQLite file and
compared with plain cosine similarity in Python; fine up to tens of thousands
of reference images (see db.fetch_reference_embeddings for the scaling note).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from . import config, db
from .client import OpenRouterClient


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class DuplicateResult:
    is_duplicate: bool
    best_match_upload_id: str | None
    best_match_similarity: float
    best_match_claimed_vrn: str | None
    reason: str


def check_duplicate(conn: sqlite3.Connection, client: OpenRouterClient, *,
                     image_path: str, image_type: str, claimed_vrn: str,
                     exclude_upload_id: str | None = None,
                     embed_model: str = config.DEFAULT_EMBED_MODEL,
                     similarity_min: float = config.DUPLICATE_SIMILARITY_MIN,
                     ) -> tuple[DuplicateResult, "EmbedCallInfo"]:
    """Embeds `image_path`, compares it against every stored reference image of
    the same `image_type`, and flags a duplicate only when the closest match is
    at or above `similarity_min` AND was filed under a *different* claimed VRN —
    an honest re-upload under the same VRN is never flagged (same rule the
    existing pgvector-based checker uses).
    """
    result = client.embed(model=embed_model, image_path=image_path)
    call_info = EmbedCallInfo(result.model, result.prompt_tokens, result.cost_usd, result.latency_ms)

    candidates = db.fetch_reference_embeddings(conn, image_type, exclude_upload_id)
    best_upload_id: str | None = None
    best_sim = -1.0
    best_vrn: str | None = None
    for upload_id, vector, ref_vrn in candidates:
        sim = cosine_similarity(result.vector, vector)
        if sim > best_sim:
            best_upload_id, best_sim, best_vrn = upload_id, sim, ref_vrn

    if best_upload_id is None:
        return DuplicateResult(False, None, 0.0, None, "no reference images of this type yet"), call_info

    is_dup = best_sim >= similarity_min and best_vrn != claimed_vrn
    if best_sim >= similarity_min and best_vrn == claimed_vrn:
        reason = "near-duplicate but under the same claimed VRN — treated as an honest re-upload"
    elif is_dup:
        reason = f"near-duplicate (sim={best_sim:.4f}) filed under a different VRN"
    else:
        reason = f"closest match sim={best_sim:.4f}, below threshold {similarity_min}"

    return DuplicateResult(is_dup, best_upload_id, round(best_sim, 4), best_vrn, reason), call_info


@dataclass
class EmbedCallInfo:
    model: str
    prompt_tokens: int
    cost_usd: float
    latency_ms: int
