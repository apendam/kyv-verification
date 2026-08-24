"""Near-duplicate upload detection — is this upload's image a near-duplicate of a
PREVIOUSLY ACCEPTED upload filed under a DIFFERENT claimed VRN?

Catches the "same photo reused, only the VRN changed" fraud pattern: a ground
agent without the actual vehicle in front of them re-submits an old accepted
photo against a new claim. This looks *across* uploads, not within one — distinct
from (and not gating/gated by) Q1/Q2/Q3.

Embedding: SigLIP 2's image tower (``backends/siglip.py``'s ``embed_image`` —
same weights already loaded for Q1's pose check and Q3's make check, just one
step earlier in the forward pass — no second model to load). Storage/search:
Postgres + pgvector (``backends/vector_store.py``) — exact cosine nearest-
neighbor search, fast even over millions of rows via an HNSW index.

NOT wired into ``validators/combined.py``/``validate_upload`` yet. Call
``check_duplicate`` alongside it (or from a batch job over recent uploads) until
you've calibrated ``config.DUPLICATE_SIMILARITY_MIN`` against real labeled pairs
from your own data — see the README's "Duplicate detection" section. Treat a hit
as a MANUAL_REVIEW lead, never an auto-reject: this is a fraud *signal*, not a
verdict, same posture as Q3's make-check backstop.
"""
from __future__ import annotations

from truck_extract_match.plate.format import normalize_vrn

from vfiv import config
from vfiv.backends.siglip import get_siglip_model
from vfiv.backends.vector_store import (
    DuplicateMatch,
    DuplicateStoreError,
    find_similar,
    store_embedding,
)
from vfiv.schemas import DuplicateCheckResult


def decide_duplicate(
    matches: list[DuplicateMatch],
    claimed_vrn: str,
    similarity_min: float = config.DUPLICATE_SIMILARITY_MIN,
) -> DuplicateCheckResult:
    """Pure decision logic over already-fetched nearest neighbors — no model or DB
    call here, so this is fully unit-testable on its own.

    A prior upload only counts as a fraud lead if it's near-identical AND was
    filed under a DIFFERENT VRN than this upload's claim; a near-duplicate under
    the SAME VRN is just an honest re-submission and is never flagged.
    """
    claimed_norm = normalize_vrn(claimed_vrn)
    suspects = [m for m in matches
                if m.similarity >= similarity_min and normalize_vrn(m.claimed_vrn) != claimed_norm]

    if not suspects:
        best = matches[0] if matches else None
        return DuplicateCheckResult(
            decision="PASS",
            checked=True,
            claimed_vrn=claimed_vrn,
            is_duplicate_suspect=False,
            best_match_id=best.upload_id if best else None,
            best_match_similarity=best.similarity if best else None,
            best_match_vrn=best.claimed_vrn if best else None,
            reason=("closest prior upload isn't a near-duplicate under a different VRN"
                    if best else "no prior uploads on file"),
        )

    top = max(suspects, key=lambda m: m.similarity)
    return DuplicateCheckResult(
        decision="MANUAL_REVIEW",
        checked=True,
        claimed_vrn=claimed_vrn,
        is_duplicate_suspect=True,
        best_match_id=top.upload_id,
        best_match_similarity=top.similarity,
        best_match_vrn=top.claimed_vrn,
        reason=(f"near-duplicate (similarity={top.similarity:.4f}) of upload "
                f"{top.upload_id!r}, filed there under a different VRN "
                f"({top.claimed_vrn!r}) — possible reused/tampered photo"),
    )


def check_duplicate(
    image,
    upload_id: str,
    claimed_vrn: str,
    top_k: int = 5,
    similarity_min: float = config.DUPLICATE_SIMILARITY_MIN,
    store: bool = True,
) -> DuplicateCheckResult:
    """Embed ``image``, search prior uploads for near-duplicates, decide, then (by
    default) store this upload's own embedding so future uploads can be checked
    against it too. ``upload_id`` is a stable identifier for this upload (e.g. its
    DB row id) — used only for the audit trail, never interpreted here.
    """
    try:
        embedding = get_siglip_model().embed_image(image)
        matches = find_similar(embedding, top_k=top_k)
    except DuplicateStoreError as e:
        return DuplicateCheckResult(
            decision="MANUAL_REVIEW",
            checked=False,
            claimed_vrn=claimed_vrn,
            is_duplicate_suspect=False,
            reason=f"duplicate check unavailable ({e})",
            error=str(e),
        )

    result = decide_duplicate(matches, claimed_vrn, similarity_min)
    if store:
        store_embedding(upload_id, claimed_vrn, embedding)
    return result
