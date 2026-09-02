"""SQLite storage: one file, no server, no hardware deployment — shared by both
scripts. Three tables:

  checks           — one row per model call: the token/cost/verdict ledger.
  results          — one row per upload: the final gate-sequence decision.
  reference_images — the duplicate-check corpus (script 2 writes here; the
                      duplicate check in script 1 reads it).
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
    verdict TEXT NOT NULL,          -- short outcome label, e.g. "match" / "mismatch_other"
    detail_json TEXT NOT NULL,      -- full structured response from the model
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
    claimed_vrn TEXT,
    claimed_make TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_images (
    upload_id TEXT NOT NULL,
    image_type TEXT NOT NULL,       -- "front" | "fastag" | "side"
    image_path TEXT NOT NULL,
    claimed_vrn TEXT,
    phash TEXT NOT NULL,            -- local perceptual hash (imagehash.phash), as hex
    created_at REAL NOT NULL,
    PRIMARY KEY (upload_id, image_type)
);

CREATE INDEX IF NOT EXISTS idx_checks_upload ON checks(upload_id);
CREATE INDEX IF NOT EXISTS idx_ref_type ON reference_images(image_type);
"""


def _migrate_reference_images(conn: sqlite3.Connection) -> None:
    """The duplicate check used to store an OpenRouter embedding per
    reference image; it now stores a local perceptual hash instead (see
    imaging.py / duplicate.py) -- a different signal entirely, so an
    embedding-schema row can't be reused under the new comparison. A table
    from before this change is dropped and recreated with the new schema
    rather than migrated in place; this is a duplicate-detection corpus you
    rebuild by re-seeding, not an irreplaceable record.
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(reference_images)").fetchall()]
    if cols and "phash" not in cols:
        conn.execute("DROP TABLE reference_images")
        conn.commit()


def connect(db_path: str | Path = config.DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    _migrate_reference_images(conn)
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


def record_result(conn: sqlite3.Connection, *, upload_id: str, decision: str,
                   reason: str, claimed_vrn: str | None = None,
                   claimed_make: str | None = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (upload_id, decision, reason, claimed_vrn, "
        "claimed_make, created_at) VALUES (?,?,?,?,?,?)",
        (upload_id, decision, reason, claimed_vrn, claimed_make, time.time()),
    )
    conn.commit()


def already_checked(conn: sqlite3.Connection, upload_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM results WHERE upload_id = ?", (upload_id,)).fetchone()
    return row is not None


# -- reference-image repository (duplicate-check corpus) ---------------------

def insert_reference_image(conn: sqlite3.Connection, *, upload_id: str, image_type: str,
                            image_path: str, claimed_vrn: str | None, phash: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO reference_images (upload_id, image_type, image_path, "
        "claimed_vrn, phash, created_at) VALUES (?,?,?,?,?,?)",
        (upload_id, image_type, image_path, claimed_vrn, phash, time.time()),
    )
    conn.commit()


def fetch_reference_phashes(conn: sqlite3.Connection, image_type: str,
                             exclude_upload_id: str | None = None
                             ) -> list[tuple[str, str, str | None]]:
    """Returns [(upload_id, phash_hex, claimed_vrn), ...] for one image_type.
    A linear scan is fine up to tens of thousands of rows; if this repo grows
    past that, swap this for an indexed nearest-neighbor structure rather
    than optimizing this scan.
    """
    q = "SELECT upload_id, phash, claimed_vrn FROM reference_images WHERE image_type = ?"
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
    """Every stored reference image of one type, newest first — for browsing/
    viewing individually (e.g. a gallery), not for the duplicate comparison
    (see fetch_reference_phashes for that).
    """
    rows = conn.execute(
        "SELECT upload_id, image_path, claimed_vrn, phash, created_at FROM reference_images "
        "WHERE image_type = ? ORDER BY created_at DESC",
        (image_type,),
    ).fetchall()
    return [{"upload_id": r[0], "image_path": r[1], "claimed_vrn": r[2], "phash": r[3],
             "created_at": r[4], "image_type": image_type} for r in rows]


def delete_reference_image(conn: sqlite3.Connection, upload_id: str, image_type: str) -> None:
    conn.execute(
        "DELETE FROM reference_images WHERE upload_id = ? AND image_type = ?",
        (upload_id, image_type),
    )
    conn.commit()
