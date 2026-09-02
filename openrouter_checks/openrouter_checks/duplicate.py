"""Near-duplicate detection over two signals, cascaded cheapest-first:

  1. **pHash** (local perceptual hash) -- free, instant, no model load. Catches
     near-pixel-identical reuploads (same photo, maybe recompressed/lightly
     edited). If this already flags a duplicate, that's the answer -- no need
     to run anything heavier.
  2. **SigLIP** (local vector embedding) -- only run when pHash does NOT flag
     a duplicate. Needs a ~400MB model loaded via `transformers`/`torch` (see
     siglip.py), much more robust to a re-crop, angle, or lighting change
     than pHash, which only catches near-identical images. This is the
     fallback that catches what pHash misses, not a first pass.

Neither makes a network call -- both are local, free, deterministic. Hashes
and vectors are stored in the same SQLite file as hex/blob columns; comparing
against every stored reference is a linear scan, fine up to tens of
thousands of reference images (see db.fetch_reference_phashes /
db.fetch_reference_siglip_embeddings).
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

import imagehash
import numpy as np
from PIL import Image

from . import config, db, imaging
from .siglip import get_siglip_model


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


def compute_siglip_embedding(image_path: str,
                              plate_bbox: tuple[float, float, float, float] | None = None
                              ) -> np.ndarray:
    """Same plate-masking as compute_phash, just feeding the local SigLIP
    model instead of the hash function -- the two signals compare the same
    (masked) pixels."""
    mask_path = image_path
    if plate_bbox is not None:
        mask_path = imaging.mask_normalized_box(image_path, plate_bbox)
    try:
        return get_siglip_model().embed_image(mask_path)
    finally:
        if mask_path != image_path:
            os.unlink(mask_path)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class DuplicateResult:
    is_duplicate: bool
    signal: str  # "phash" | "siglip" | "none" (no reference images to compare against)
    best_match_upload_id: str | None
    best_match_score: float | None  # hamming distance (phash, lower=more similar) or
                                     # cosine similarity (siglip, higher=more similar)
    best_match_claimed_vrn: str | None
    reason: str


def _phash_check(conn: sqlite3.Connection, *, image_path: str, image_type: str, claimed_vrn: str,
                  exclude_upload_id: str | None, hamming_max: int,
                  plate_bbox: tuple[float, float, float, float] | None) -> DuplicateResult:
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
        return DuplicateResult(False, "none", None, None, None,
                                "no reference images of this type yet")

    is_dup = bool(best_distance <= hamming_max and best_vrn != claimed_vrn)
    if best_distance <= hamming_max and best_vrn == claimed_vrn:
        reason = "near-duplicate (pHash) but under the same claimed VRN — treated as an honest re-upload"
    elif is_dup:
        reason = f"near-duplicate (pHash hamming={best_distance}) filed under a different VRN"
    else:
        reason = f"closest pHash match hamming={best_distance}, above threshold {hamming_max}"

    return DuplicateResult(is_dup, "phash", best_upload_id, float(best_distance), best_vrn, reason)


def _siglip_check(conn: sqlite3.Connection, *, image_path: str, image_type: str, claimed_vrn: str,
                   exclude_upload_id: str | None, similarity_min: float,
                   plate_bbox: tuple[float, float, float, float] | None) -> DuplicateResult:
    query_vec = compute_siglip_embedding(image_path, plate_bbox)
    candidates = db.fetch_reference_siglip_embeddings(conn, image_type, exclude_upload_id)

    best_upload_id: str | None = None
    best_similarity: float | None = None
    best_vrn: str | None = None
    for upload_id, blob, ref_vrn in candidates:
        similarity = _cosine_similarity(query_vec, db.unpack_embedding(blob))
        if best_similarity is None or similarity > best_similarity:
            best_upload_id, best_similarity, best_vrn = upload_id, similarity, ref_vrn

    if best_upload_id is None:
        return DuplicateResult(False, "none", None, None, None,
                                "no reference images with a vector embedding for this type yet")

    is_dup = bool(best_similarity >= similarity_min and best_vrn != claimed_vrn)
    if best_similarity >= similarity_min and best_vrn == claimed_vrn:
        reason = "near-duplicate (SigLIP) but under the same claimed VRN — treated as an honest re-upload"
    elif is_dup:
        reason = f"near-duplicate (SigLIP similarity={best_similarity:.4f}) filed under a different VRN"
    else:
        reason = f"closest SigLIP match similarity={best_similarity:.4f}, below threshold {similarity_min}"

    return DuplicateResult(is_dup, "siglip", best_upload_id, best_similarity, best_vrn, reason)


def check_duplicate(conn: sqlite3.Connection, *, image_path: str, image_type: str, claimed_vrn: str,
                     exclude_upload_id: str | None = None,
                     hamming_max: int = config.DUPLICATE_HAMMING_MAX,
                     siglip_similarity_min: float = config.DUPLICATE_SIGLIP_SIMILARITY_MIN,
                     plate_bbox: tuple[float, float, float, float] | None = None,
                     ) -> DuplicateResult:
    """pHash first; if it already flags a duplicate (or there's no reference
    corpus at all to compare against), that's the final answer -- no need to
    load SigLIP. Only when pHash runs a real comparison and comes back clean
    does the heavier SigLIP pass run, to catch what pHash's near-identical-
    pixels test misses (a re-crop, a lighting or angle change).
    """
    phash_result = _phash_check(
        conn, image_path=image_path, image_type=image_type, claimed_vrn=claimed_vrn,
        exclude_upload_id=exclude_upload_id, hamming_max=hamming_max, plate_bbox=plate_bbox,
    )
    if phash_result.is_duplicate or phash_result.signal == "none":
        return phash_result

    return _siglip_check(
        conn, image_path=image_path, image_type=image_type, claimed_vrn=claimed_vrn,
        exclude_upload_id=exclude_upload_id, similarity_min=siglip_similarity_min,
        plate_bbox=plate_bbox,
    )
