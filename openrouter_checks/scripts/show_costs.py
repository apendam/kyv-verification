#!/usr/bin/env python3
"""Token/cost report over everything logged so far — "how much did this cost
me" answered from the ledger, not re-computed from a price table.

Usage:
    python scripts/show_costs.py
    python scripts/show_costs.py --by model
    python scripts/show_costs.py --upload-id upload_001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openrouter_checks import config, db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(config.DEFAULT_DB_PATH), help="SQLite file path.")
    ap.add_argument("--by", choices=["model", "check_name"], default="model",
                     help="Group the breakdown by model or by check type (default: model).")
    ap.add_argument("--upload-id", help="Restrict to one upload_id.")
    args = ap.parse_args()

    conn = db.connect(args.db)
    where, params = "", []
    if args.upload_id:
        where, params = "WHERE upload_id = ?", [args.upload_id]

    print(f"=== by {args.by} ===")
    rows = conn.execute(
        f"SELECT {args.by}, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), "
        f"SUM(cost_usd), SUM(technical_failure) FROM checks {where} GROUP BY {args.by} "
        f"ORDER BY SUM(cost_usd) DESC",
        params,
    ).fetchall()
    for name, n_calls, p_tok, c_tok, cost, n_fail in rows:
        print(f"  {name:40s}  calls={n_calls:5d}  tokens={((p_tok or 0)+(c_tok or 0)):8d}  "
              f"cost=${cost or 0:.5f}  technical_failures={n_fail}")

    total = conn.execute(
        f"SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(cost_usd) "
        f"FROM checks {where}", params,
    ).fetchone()
    n_calls, p_tok, c_tok, cost = total
    print(f"\nTOTAL: {n_calls or 0} calls, {((p_tok or 0)+(c_tok or 0))} tokens, ${cost or 0:.5f}")

    decisions = conn.execute(
        f"SELECT decision, COUNT(*) FROM results {where} GROUP BY decision", params,
    ).fetchall()
    if decisions:
        print("\n=== decisions ===")
        for decision, n in decisions:
            print(f"  {decision:15s} {n}")


if __name__ == "__main__":
    main()
