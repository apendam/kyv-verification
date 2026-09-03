#!/usr/bin/env python3
"""Add a known-good side/axle image to the reference repository used by the
duplicate check — one row per upload_id, storing a local perceptual hash
(pHash) and, if available, a SigLIP vector -- not the raw pixels (the image
file itself just needs to stay at the path you give it).

Usage:
    python scripts/seed_reference.py --image path/to/side.jpg \\
        --upload-id upload_001 --vrn MH12AB1234

    # bulk-seed from a CSV with columns: image_path,upload_id,vrn
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

from kyv_side import config, db, duplicate, prompts, schemas  # noqa: E402
from kyv_side.client import OpenRouterClient, OpenRouterInsufficientCredits  # noqa: E402
from kyv_side.side_sequence import IMAGE_TYPE  # noqa: E402


def _locate_vrn_bbox(client: OpenRouterClient, image_path: str, vision_model: str):
    """One vision call to find the VRN (plate or painted text) so it can be
    blacked out before hashing (see duplicate.compute_phash) -- mirrors what
    a real check run gets for free from its own VRN read. A technical
    failure here just skips masking rather than failing the seed outright.
    """
    try:
        r = client.chat_json(
            model=vision_model, system_prompt=prompts.SIDE_VRN_READ_SYSTEM,
            user_text=prompts.side_vrn_read_user_text(), image_paths=[image_path],
            json_schema=schemas.SIDE_VRN_READ_SCHEMA, schema_name="vrn_locate",
        )
    except Exception:  # noqa: BLE001 - masking is a nice-to-have, not required to seed
        return None
    if not r.data.get("vrn_visible", False):
        return None
    return (r.data.get("bbox_x_min", 0.0), r.data.get("bbox_y_min", 0.0),
            r.data.get("bbox_x_max", 0.0), r.data.get("bbox_y_max", 0.0))


def _locate_vehicle_bbox(client: OpenRouterClient, image_path: str, vision_model: str):
    """One vision call to find the main vehicle so the reference image can be
    cropped to just the vehicle before hashing (see duplicate.compute_phash)
    -- mirrors what a real check run gets for free from its own framing
    check. A technical failure here just skips cropping rather than failing
    the seed outright.
    """
    try:
        r = client.chat_json(
            model=vision_model, system_prompt=prompts.FRAMING_SYSTEM,
            user_text=prompts.framing_user_text(), image_paths=[image_path],
            json_schema=schemas.FRAMING_SCHEMA, schema_name="vehicle_locate",
        )
    except Exception:  # noqa: BLE001 - cropping is a nice-to-have, not required to seed
        return None
    return (r.data.get("vehicle_bbox_x_min", 0.0), r.data.get("vehicle_bbox_y_min", 0.0),
            r.data.get("vehicle_bbox_x_max", 0.0), r.data.get("vehicle_bbox_y_max", 0.0))


def seed_one(conn, client: OpenRouterClient, *, image_path: str, upload_id: str,
             vrn: str | None, vision_model: str) -> None:
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    vrn_bbox = _locate_vrn_bbox(client, image_path, vision_model)
    vehicle_bbox = _locate_vehicle_bbox(client, image_path, vision_model)
    phash = str(duplicate.compute_phash(image_path, vrn_bbox, vehicle_bbox))

    siglip_embedding = None
    try:
        siglip_embedding = db.pack_embedding(
            duplicate.compute_siglip_embedding(image_path, vrn_bbox, vehicle_bbox))
    except Exception:  # noqa: BLE001 - torch/transformers missing, or model download failed
        print(f"warning: no vector embedding for {upload_id} (pHash alone still works)", file=sys.stderr)

    db.insert_reference_image(
        conn, upload_id=upload_id, image_type=IMAGE_TYPE, image_path=image_path,
        claimed_vrn=vrn, phash=phash, siglip_embedding=siglip_embedding,
    )
    print(f"seeded {upload_id} [{IMAGE_TYPE}] <- {image_path} (phash {phash}, "
          f"vector {'stored' if siglip_embedding else 'unavailable'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="Path to one image to seed.")
    ap.add_argument("--upload-id", help="Unique id for --image.")
    ap.add_argument("--vrn", help="Claimed VRN for --image (optional, but needed for the "
                                  "same-VRN-is-not-a-duplicate rule to work).")
    ap.add_argument("--csv", help="Bulk-seed from a CSV: image_path,upload_id,vrn")
    ap.add_argument("--vision-model", default=config.DEFAULT_VISION_MODEL,
                     help=f"OpenRouter vision model used to locate the VRN for masking "
                          f"(default: {config.DEFAULT_VISION_MODEL}).")
    ap.add_argument("--db", default=str(config.DEFAULT_DB_PATH), help="SQLite file path.")
    ap.add_argument("--stats", action="store_true", help="Just print corpus counts and exit.")
    args = ap.parse_args()

    conn = db.connect(args.db)

    if args.stats:
        print(db.reference_stats(conn))
        return

    if not args.image and not args.csv:
        ap.error("pass --image (with --upload-id) or --csv")

    client = OpenRouterClient()
    try:
        if args.csv:
            with open(args.csv, newline="") as f:
                for row in csv.DictReader(f):
                    seed_one(conn, client, image_path=row["image_path"], upload_id=row["upload_id"],
                             vrn=row.get("vrn") or None, vision_model=args.vision_model)
        else:
            if not args.upload_id:
                ap.error("--image requires --upload-id")
            seed_one(conn, client, image_path=args.image, upload_id=args.upload_id,
                     vrn=args.vrn, vision_model=args.vision_model)
    except OpenRouterInsufficientCredits as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(2)

    t = client.totals
    print(f"\n-- this run: {t.calls} call(s), {t.prompt_tokens} tokens, ${t.cost_usd:.5f} --",
          file=sys.stderr)
    print(f"corpus now: {db.reference_stats(conn)}", file=sys.stderr)


if __name__ == "__main__":
    main()
