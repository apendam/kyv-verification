"""SQLite storage: one file, no server. Three tables:

  checks           — one row per model call: the token/cost/verdict ledger.
  results          — one row per upload: the final gate-sequence decision.
  reference_images — the duplicate-check corpus (script 2 writes here; the
                      duplicate check in script 1 reads it), image_type="side".
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id TEXT NOT NULL,
    image_type TEXT NOT NULL,
    check_name TEXT NOT NULL,
    model TEXT NOT NULL,
    verdict TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    technical_failure INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    upload_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,         -- APPROVED | MANUAL_REVIEW | REJECT
    reason TEXT NOT NULL,
    claimed_vehicle_type TEXT,
    claimed_axle_count INTEGER,
    claimed_vrn TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_images (
    upload_id TEXT NOT NULL,
    image_type TEXT NOT NULL,       -- always "side" here, kept generic for parity with the front-image flow
    image_path TEXT NOT NULL,
    claimed_vrn TEXT,
    phash TEXT NOT NULL,            -- local perceptual hash (imagehash.phash), as hex
    siglip_embedding BLOB,          -- packed float32 SigLIP vector (see pack_embedding); NULL until seeded
    created_at REAL NOT NULL,
    PRIMARY KEY (upload_id, image_type)
);

CREATE INDEX IF NOT EXISTS idx_checks_upload ON checks(upload_id);
CREATE INDEX IF NOT EXISTS idx_ref_type ON reference_images(image_type);
"""


def connect(db_path: str | Path = config.DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def log_check(conn: sqlite3.Connection, *, upload_id: str, image_type: str,
              check_name: str, model: str, verdict: str, detail: dict[str, Any],
              prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0,
              latency_ms: int = 0, technical_failure: bool = False) -> None:
    conn.execute(
        "INSERT INTO checks (upload_id, image_type, check_name, model, verdict, "
        "detail_json, prompt_tokens, completion_tokens, cost_usd, latency_ms, "
        "technical_failure, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (upload_id, image_type, check_name, model, verdict, json.dumps(detail),
         prompt_tokens, completion_tokens, cost_usd, latency_ms,
         int(technical_failure), time.time()),
    )
    conn.commit()


def record_result(conn: sqlite3.Connection, *, upload_id: str, decision: str, reason: str,
                   claimed_vehicle_type: str | None = None, claimed_axle_count: int | None = None,
                   claimed_vrn: str | None = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (upload_id, decision, reason, claimed_vehicle_type, "
        "claimed_axle_count, claimed_vrn, created_at) VALUES (?,?,?,?,?,?,?)",
        (upload_id, decision, reason, claimed_vehicle_type, claimed_axle_count, claimed_vrn, time.time()),
    )
    conn.commit()


def fetch_checks_for_upload(conn: sqlite3.Connection, upload_id: str) -> list[dict[str, Any]]:
    """Every logged check-step row for one upload, oldest first -- what a
    batch report is built from (see scripts/batch_check.py). Later rows for
    the same check_name (a --force re-run) naturally come after earlier
    ones; a caller building a {check_name: row} dict by iterating this list
    keeps the most recent.
    """
    rows = conn.execute(
        "SELECT check_name, model, verdict, detail_json, prompt_tokens, completion_tokens, "
        "cost_usd, latency_ms, technical_failure FROM checks WHERE upload_id = ? ORDER BY id ASC",
        (upload_id,),
    ).fetchall()
    return [
        {"check_name": r[0], "model": r[1], "verdict": r[2], "detail": json.loads(r[3]),
         "prompt_tokens": r[4], "completion_tokens": r[5], "cost_usd": r[6],
         "latency_ms": r[7], "technical_failure": bool(r[8])}
        for r in rows
    ]


def already_checked(conn: sqlite3.Connection, upload_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM results WHERE upload_id = ?", (upload_id,)).fetchone()
    return row is not None


# -- reference-image repository (duplicate-check corpus) ---------------------

def pack_embedding(vector) -> bytes:
    import numpy as np
    return np.asarray(vector, dtype="float32").tobytes()


def unpack_embedding(blob: bytes):
    import numpy as np
    return np.frombuffer(blob, dtype="float32")


def insert_reference_image(conn: sqlite3.Connection, *, upload_id: str, image_type: str,
                            image_path: str, claimed_vrn: str | None, phash: str,
                            siglip_embedding: bytes | None = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO reference_images (upload_id, image_type, image_path, "
        "claimed_vrn, phash, siglip_embedding, created_at) VALUES (?,?,?,?,?,?,?)",
        (upload_id, image_type, image_path, claimed_vrn, phash, siglip_embedding, time.time()),
    )
    conn.commit()


def fetch_reference_phashes(conn: sqlite3.Connection, image_type: str,
                             exclude_upload_id: str | None = None
                             ) -> list[tuple[str, str, str | None]]:
    q = "SELECT upload_id, phash, claimed_vrn FROM reference_images WHERE image_type = ?"
    params: list[Any] = [image_type]
    if exclude_upload_id is not None:
        q += " AND upload_id != ?"
        params.append(exclude_upload_id)
    return conn.execute(q, params).fetchall()


def fetch_reference_siglip_embeddings(conn: sqlite3.Connection, image_type: str,
                                       exclude_upload_id: str | None = None
                                       ) -> list[tuple[str, bytes, str | None]]:
    q = ("SELECT upload_id, siglip_embedding, claimed_vrn FROM reference_images "
         "WHERE image_type = ? AND siglip_embedding IS NOT NULL")
    params: list[Any] = [image_type]
    if exclude_upload_id is not None:
        q += " AND upload_id != ?"
        params.append(exclude_upload_id)
    return conn.execute(q, params).fetchall()


def reference_stats(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT image_type, COUNT(*) FROM reference_images GROUP BY image_type"
    ).fetchall()
    return dict(rows)


def list_reference_images(conn: sqlite3.Connection, image_type: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT upload_id, image_path, claimed_vrn, phash, siglip_embedding, created_at "
        "FROM reference_images WHERE image_type = ? ORDER BY created_at DESC",
        (image_type,),
    ).fetchall()
    return [{"upload_id": r[0], "image_path": r[1], "claimed_vrn": r[2], "phash": r[3],
             "has_siglip_embedding": r[4] is not None, "created_at": r[5],
             "image_type": image_type} for r in rows]


def delete_reference_image(conn: sqlite3.Connection, upload_id: str, image_type: str) -> None:
    conn.execute(
        "DELETE FROM reference_images WHERE upload_id = ? AND image_type = ?",
        (upload_id, image_type),
    )
    conn.commit()
