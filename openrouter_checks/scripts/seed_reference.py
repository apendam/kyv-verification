#!/usr/bin/env python3
"""Add a known-good image to the reference repository used by the duplicate
check — one row per (upload_id, image_type), storing an embedding, not the raw
pixels-in-a-database (the image file itself just needs to stay at the path you
give it, or be copied somewhere stable first).

Usage:
    python scripts/seed_reference.py --image path/to/front.jpg \\
        --image-type front --upload-id upload_001 --vrn MH12AB1234

    python scripts/seed_reference.py --image path/to/tag.jpg \\
        --image-type fastag --upload-id upload_001 --vrn MH12AB1234

    # bulk-seed from a CSV with columns: image_path,image_type,upload_id,vrn
    python scripts/seed_reference.py --csv reference_images.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from openrouter_checks import config, db  # noqa: E402
from openrouter_checks.client import OpenRouterClient, OpenRouterInsufficientCredits  # noqa: E402

VALID_TYPES = {"front", "fastag", "side"}


def seed_one(conn, client: OpenRouterClient, *, image_path: str, image_type: str,
             upload_id: str, vrn: str | None, embed_model: str) -> None:
    if image_type not in VALID_TYPES:
        raise ValueError(f"image_type must be one of {sorted(VALID_TYPES)}, got {image_type!r}")
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    result = client.embed(model=embed_model, image_path=image_path)
    db.insert_reference_image(
        conn, upload_id=upload_id, image_type=image_type, image_path=image_path,
        claimed_vrn=vrn, embedding=result.vector, embed_model=result.model,
    )
    print(f"seeded {upload_id} [{image_type}] <- {image_path} "
          f"(${result.cost_usd:.5f}, {result.prompt_tokens} tokens)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="Path to one image to seed.")
    ap.add_argument("--image-type", choices=sorted(VALID_TYPES), help="Image type for --image.")
    ap.add_argument("--upload-id", help="Unique id for --image.")
    ap.add_argument("--vrn", help="Claimed VRN for --image (optional, but needed for the "
                                  "same-VRN-is-not-a-duplicate rule to work).")
    ap.add_argument("--csv", help="Bulk-seed from a CSV: image_path,image_type,upload_id,vrn")
    ap.add_argument("--embed-model", default=config.DEFAULT_EMBED_MODEL,
                     help=f"OpenRouter embedding model (default: {config.DEFAULT_EMBED_MODEL}).")
    ap.add_argument("--db", default=str(config.DEFAULT_DB_PATH), help="SQLite file path.")
    ap.add_argument("--stats", action="store_true", help="Just print corpus counts and exit.")
    args = ap.parse_args()

    conn = db.connect(args.db)

    if args.stats:
        print(db.reference_stats(conn))
        return

    if not args.image and not args.csv:
        ap.error("pass --image (with --image-type/--upload-id) or --csv")

    client = OpenRouterClient()
    try:
        if args.csv:
            with open(args.csv, newline="") as f:
                for row in csv.DictReader(f):
                    seed_one(conn, client, image_path=row["image_path"],
                             image_type=row["image_type"], upload_id=row["upload_id"],
                             vrn=row.get("vrn") or None, embed_model=args.embed_model)
        else:
            if not (args.image_type and args.upload_id):
                ap.error("--image requires --image-type and --upload-id")
            seed_one(conn, client, image_path=args.image, image_type=args.image_type,
                     upload_id=args.upload_id, vrn=args.vrn, embed_model=args.embed_model)
    except OpenRouterInsufficientCredits as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(2)

    t = client.totals
    print(f"\n-- this run: {t.calls} call(s), {t.prompt_tokens} tokens, ${t.cost_usd:.5f} --",
          file=sys.stderr)
    print(f"corpus now: {db.reference_stats(conn)}", file=sys.stderr)


if __name__ == "__main__":
    main()
