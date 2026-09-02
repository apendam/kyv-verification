#!/usr/bin/env python3
"""Run one upload through the KYV front-image gate sequence via OpenRouter.

Usage:
    python scripts/check_image.py --image path/to/front.jpg --vehicle-type truck \\
        --vrn MH12AB1234 --make "TATA MOTORS LTD" --upload-id upload_001

    # try a different model for this run only, nothing else changes:
    python scripts/check_image.py --image front.jpg --vehicle-type bus \\
        --vrn MH12AB1234 --make "TATA MOTORS LTD" --upload-id upload_002 \\
        --model google/gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from openrouter_checks import config, db  # noqa: E402
from openrouter_checks.client import OpenRouterClient, OpenRouterInsufficientCredits  # noqa: E402
from openrouter_checks.gate_sequence import run_gate_sequence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to the front-image upload.")
    ap.add_argument("--vrn", required=True, help="Claimed vehicle registration number.")
    ap.add_argument("--make", required=True, help="Claimed manufacturer/make.")
    ap.add_argument("--vehicle-type", required=True, choices=["bus", "truck"],
                     help="Claimed vehicle type -- rejected outright if the image doesn't match.")
    ap.add_argument("--upload-id", required=True, help="Unique id for this upload.")
    ap.add_argument("--model", default=config.DEFAULT_VISION_MODEL,
                     help=f"OpenRouter vision model (default: {config.DEFAULT_VISION_MODEL}).")
    ap.add_argument("--db", default=str(config.DEFAULT_DB_PATH), help="SQLite file path.")
    ap.add_argument("--force", action="store_true",
                     help="Re-run even if this upload_id already has a recorded result "
                          "(by default, a repeat run is skipped to avoid re-paying for it).")
    args = ap.parse_args()

    if not Path(args.image).is_file():
        ap.error(f"image not found: {args.image}")

    conn = db.connect(args.db)

    if not args.force and db.already_checked(conn, args.upload_id):
        print(f"upload_id '{args.upload_id}' was already checked — pass --force to re-run "
              f"(no API calls made, nothing charged).", file=sys.stderr)
        row = conn.execute(
            "SELECT decision, reason FROM results WHERE upload_id = ?", (args.upload_id,)
        ).fetchone()
        print(json.dumps({"upload_id": args.upload_id, "decision": row[0], "reason": row[1],
                          "skipped": True}, indent=2))
        return

    client = OpenRouterClient()
    try:
        result = run_gate_sequence(
            conn, client, image_path=args.image, claimed_vrn=args.vrn, claimed_make=args.make,
            claimed_vehicle_type=args.vehicle_type, upload_id=args.upload_id, vision_model=args.model,
        )
    except OpenRouterInsufficientCredits as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        print("Top up your OpenRouter account before retrying — this isn't a per-image "
              "failure, every call will fail the same way until you do.", file=sys.stderr)
        sys.exit(2)

    print(json.dumps({
        "upload_id": result.upload_id,
        "decision": result.decision,
        "reason": result.reason,
        "steps": result.steps,
    }, indent=2))

    t = client.totals
    print(f"\n-- this run: {t.calls} call(s), {t.prompt_tokens + t.completion_tokens} tokens, "
          f"${t.cost_usd:.5f} --", file=sys.stderr)

    sys.exit(0 if result.decision != "REJECT" else 1)


if __name__ == "__main__":
    main()
