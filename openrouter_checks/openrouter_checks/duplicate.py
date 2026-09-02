"""Near-duplicate detection via a local perceptual hash (pHash), not an
OpenRouter embedding. An embedding from a general-purpose multimodal model
turned out not to cluster near 1.0 cosine similarity even for two images of
the same truck differing only in an edited plate — pHash is a much better
fit for "is this the same photo, maybe re-cropped/recompressed/lightly
edited": it's deterministic, free (no API call), and Hamming distance
between two 64-bit hashes gives a clean, well-understood similarity metric
with none of the per-model threshold-calibration guesswork an embedding
needed. Hashes are stored as hex strings in the same SQLite file; comparing
against every stored reference hash is a linear scan, fine up to tens of
thousands of reference images (see db.fetch_reference_phashes).
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

import imagehash
from PIL import Image

from . import config, db, imaging


def compute_phash(image_path: str, plate_bbox: tuple[float, float, float, float] | None = None
                   ) -> imagehash.ImageHash:
    """Hashes `image_path`, blacking out `plate_bbox` first if given (see
    imaging.mask_normalized_box) so the comparison is based on the rest of
    the vehicle rather than the plate — reusing the same photo under a
    different claimed VRN by only swapping the plate can't dodge this check,
    and the hash isn't influenced by whatever text happens to be on the
    plate. Reference images are expected to be hashed the same way for the
    comparison to be meaningful.
    """
    hash_path = image_path
    if plate_bbox is not None:
        hash_path = imaging.mask_normalized_box(image_path, plate_bbox)
    try:
        return imagehash.phash(Image.open(hash_path))
    finally:
        if hash_path != image_path:
            os.unlink(hash_path)  # a masked temp copy, never the original


@dataclass
class DuplicateResult:
    is_duplicate: bool
    best_match_upload_id: str | None
    best_match_hamming_distance: int | None
    best_match_claimed_vrn: str | None
    reason: str


def check_duplicate(conn: sqlite3.Connection, *, image_path: str, image_type: str, claimed_vrn: str,
                     exclude_upload_id: str | None = None,
                     hamming_max: int = config.DUPLICATE_HAMMING_MAX,
                     plate_bbox: tuple[float, float, float, float] | None = None,
                     ) -> DuplicateResult:
    """Hashes `image_path` (see compute_phash), compares it against every
    stored reference hash of the same `image_type`, and flags a duplicate
    only when the closest match's Hamming distance is at or below
    `hamming_max` AND was filed under a *different* claimed VRN — an honest
    re-upload under the same VRN is never flagged (same rule the existing
    pgvector-based checker uses). Lower Hamming distance = more similar;
    0 means the two 64-bit hashes are identical.
    """
    query_hash = compute_phash(image_path, plate_bbox)

    candidates = db.fetch_reference_phashes(conn, image_type, exclude_upload_id)
    best_upload_id: str | None = None
    best_distance: int | None = None
    best_vrn: str | None = None
    for upload_id, phash_hex, ref_vrn in candidates:
        # `imagehash`'s `-` operator returns numpy.int64, not a plain int --
        # cast immediately so nothing downstream (json.dumps in db.log_check,
        # in particular, which chokes on numpy scalar types) ever sees one.
        distance = int(query_hash - imagehash.hex_to_hash(phash_hex))
        if best_distance is None or distance < best_distance:
            best_upload_id, best_distance, best_vrn = upload_id, distance, ref_vrn

    if best_upload_id is None:
        return DuplicateResult(False, None, None, None, "no reference images of this type yet")

    is_dup = bool(best_distance <= hamming_max and best_vrn != claimed_vrn)
    if best_distance <= hamming_max and best_vrn == claimed_vrn:
        reason = "near-duplicate but under the same claimed VRN — treated as an honest re-upload"
    elif is_dup:
        reason = f"near-duplicate (hamming={best_distance}) filed under a different VRN"
    else:
        reason = f"closest match hamming={best_distance}, above threshold {hamming_max}"

    return DuplicateResult(is_dup, best_upload_id, best_distance, best_vrn, reason)
