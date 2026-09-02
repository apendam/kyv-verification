"""SQLite storage: one file, no server. Two tables:

  checks  -- one row per step: the token/cost/verdict ledger.
  results -- one row per upload: the final decision.

No reference_images table here -- unlike the front-image flow, this FASTag
check has no duplicate-detection step.
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
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    path_taken TEXT,
    claimed_vrn TEXT,
    claimed_barcode TEXT,
    claimed_tag_id TEXT,
    claimed_bank_code TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checks_upload ON checks(upload_id);
"""


def connect(db_path: str | Path = config.DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def log_check(conn: sqlite3.Connection, *, upload_id: str, check_name: str, model: str,
              verdict: str, detail: dict[str, Any], prompt_tokens: int = 0,
              completion_tokens: int = 0, cost_usd: float = 0.0, latency_ms: int = 0,
              technical_failure: bool = False) -> None:
    conn.execute(
        "INSERT INTO checks (upload_id, check_name, model, verdict, detail_json, "
        "prompt_tokens, completion_tokens, cost_usd, latency_ms, technical_failure, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (upload_id, check_name, model, verdict, json.dumps(detail), prompt_tokens,
         completion_tokens, cost_usd, latency_ms, int(technical_failure), time.time()),
    )
    conn.commit()


def record_result(conn: sqlite3.Connection, *, upload_id: str, decision: str, reason: str,
                   path_taken: str | None, claimed_vrn: str | None, claimed_barcode: str | None,
                   claimed_tag_id: str | None, claimed_bank_code: str | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (upload_id, decision, reason, path_taken, claimed_vrn, "
        "claimed_barcode, claimed_tag_id, claimed_bank_code, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (upload_id, decision, reason, path_taken, claimed_vrn, claimed_barcode, claimed_tag_id,
         claimed_bank_code, time.time()),
    )
    conn.commit()


def already_checked(conn: sqlite3.Connection, upload_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM results WHERE upload_id = ?", (upload_id,)).fetchone()
    return row is not None


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
