"""Postgres + pgvector nearest-neighbor store for cross-upload duplicate detection
(see ``duplicate_check.py``). Real code, degrading with a clear error —
same pattern as ``backends/rekognition.py``'s credential handling — until
``VFIV_PGVECTOR_DSN`` is set. Not wired into production by default.

One row per upload: its SigLIP embedding, the VRN claimed at upload time, an
``image_type`` (e.g. "front" | "side" | "fastag"), and an upload id for the audit
trail. ``image_type`` matters because these are visually unrelated corpora — a
front photo should never be compared against a side or FASTag photo — so every
search/store is scoped to one type. ``find_similar`` does exact cosine nearest-
neighbor search via pgvector's ``<=>`` operator, backed by an HNSW index — fast
even over millions of rows. See the README's "Duplicate detection" section for why
the raw similarity number needs calibrating against real examples before trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vfiv import config


class DuplicateStoreError(RuntimeError):
    """Raised when pgvector isn't configured, or the connection/query fails."""


@dataclass
class DuplicateMatch:
    upload_id: str
    claimed_vrn: str
    image_type: str
    similarity: float  # cosine similarity, 0..1 — 1.0 means identical direction


def _connect():
    if not config.PGVECTOR_DSN:
        raise DuplicateStoreError(
            "pgvector not configured — set VFIV_PGVECTOR_DSN "
            "(postgresql://user:pass@host:port/dbname) to enable duplicate detection."
        )
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as e:
        raise DuplicateStoreError(f"psycopg/pgvector not installed: {e}") from e
    try:
        conn = psycopg.connect(config.PGVECTOR_DSN)
        register_vector(conn)
        return conn
    except Exception as e:
        raise DuplicateStoreError(f"could not connect to pgvector store: {e}") from e


def ensure_schema(conn=None, dim: int = None) -> None:
    """Idempotent — creates the pgvector extension, table, and HNSW index if they
    don't already exist. Call once at startup/deploy time, not per-request.
    ``dim`` defaults to config.PGVECTOR_EMBED_DIM — verify it matches your actual
    SigLIP variant's embedding width first (see config.py's comment).
    """
    dim = config.PGVECTOR_EMBED_DIM if dim is None else dim
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {config.PGVECTOR_TABLE} (
                        upload_id   text PRIMARY KEY,
                        claimed_vrn text NOT NULL,
                        image_type  text NOT NULL DEFAULT 'front',
                        embedding   vector({dim}) NOT NULL,
                        created_at  timestamptz NOT NULL DEFAULT now()
                    );"""
            )
            # Idempotent upgrade path for a table created before image_type existed.
            cur.execute(
                f"""ALTER TABLE {config.PGVECTOR_TABLE}
                    ADD COLUMN IF NOT EXISTS image_type text NOT NULL DEFAULT 'front';"""
            )
            cur.execute(
                f"""CREATE INDEX IF NOT EXISTS {config.PGVECTOR_TABLE}_embedding_hnsw
                    ON {config.PGVECTOR_TABLE} USING hnsw (embedding vector_cosine_ops);"""
            )
            cur.execute(
                f"""CREATE INDEX IF NOT EXISTS {config.PGVECTOR_TABLE}_image_type
                    ON {config.PGVECTOR_TABLE} (image_type);"""
            )
        conn.commit()
    finally:
        if owns_conn:
            conn.close()


def store_embedding(
    upload_id: str, claimed_vrn: str, image_type: str, embedding: np.ndarray, conn=None,
) -> None:
    """Upsert this upload's embedding — re-running the same upload_id refreshes
    its stored claimed_vrn/image_type/embedding rather than erroring."""
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {config.PGVECTOR_TABLE} (upload_id, claimed_vrn, image_type, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (upload_id) DO UPDATE
                        SET claimed_vrn = EXCLUDED.claimed_vrn, image_type = EXCLUDED.image_type,
                            embedding = EXCLUDED.embedding""",
                (upload_id, claimed_vrn, image_type, embedding),
            )
        conn.commit()
    finally:
        if owns_conn:
            conn.close()


def find_similar(
    embedding: np.ndarray, image_type: str, top_k: int = 5, conn=None,
) -> list[DuplicateMatch]:
    """Nearest neighbors ALREADY ON FILE UNDER THE SAME image_type, ranked by
    cosine similarity (highest first). Empty list if nothing of that type has been
    stored yet — front/side/fastag are unrelated corpora and are never compared
    against each other."""
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT upload_id, claimed_vrn, image_type, 1 - (embedding <=> %s) AS similarity
                    FROM {config.PGVECTOR_TABLE}
                    WHERE image_type = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s""",
                (embedding, image_type, embedding, top_k),
            )
            rows = cur.fetchall()
        return [DuplicateMatch(upload_id=r[0], claimed_vrn=r[1], image_type=r[2], similarity=float(r[3]))
                for r in rows]
    finally:
        if owns_conn:
            conn.close()


def count_by_type(conn=None) -> dict[str, int]:
    """How many reference embeddings are stored per image_type — for a UI's
    "library size" display."""
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(f"SELECT image_type, count(*) FROM {config.PGVECTOR_TABLE} GROUP BY image_type;")
            rows = cur.fetchall()
        return {r[0]: int(r[1]) for r in rows}
    finally:
        if owns_conn:
            conn.close()
