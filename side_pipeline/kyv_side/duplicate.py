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
from dataclasses import dataclass, field

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
class DuplicateMatch:
    upload_id: str
    score: float  # hamming distance (phash, lower=more similar) or cosine similarity
                  # (siglip, higher=more similar) -- same scale/meaning as best_match_score
    claimed_vrn: str | None


@dataclass
class DuplicateResult:
    is_duplicate: bool
    signal: str  # "phash" | "siglip" | "none" (no reference images to compare against)
    best_match_upload_id: str | None
    best_match_score: float | None  # hamming distance (phash, lower=more similar) or
                                     # cosine similarity (siglip, higher=more similar)
    best_match_claimed_vrn: str | None
    reason: str
    matches: list[DuplicateMatch] = field(default_factory=list)
    # Every reference within threshold AND filed under a different VRN, best
    # first -- there can be more than one (e.g. the same photo reused across
    # several fraudulent claims). `best_match_*` above is just matches[0] when
    # matches is non-empty; is_duplicate is True iff matches is non-empty.


def _phash_check(conn: sqlite3.Connection, *, image_path: str, image_type: str, claimed_vrn: str,
                  exclude_upload_id: str | None, hamming_max: int,
                  plate_bbox: tuple[float, float, float, float] | None) -> DuplicateResult:
    query_hash = compute_phash(image_path, plate_bbox)
    candidates = db.fetch_reference_phashes(conn, image_type, exclude_upload_id)
    if not candidates:
        return DuplicateResult(False, "none", None, None, None,
                                "no reference images of this type yet")

    # `imagehash`'s `-` operator returns numpy.int64, not a plain int -- cast
    # immediately so nothing downstream (json.dumps in db.log_check, in
    # particular, which chokes on numpy scalar types) ever sees one.
    scored = [(upload_id, int(query_hash - imagehash.hex_to_hash(phash_hex)), ref_vrn)
              for upload_id, phash_hex, ref_vrn in candidates]
    overall_best_upload_id, overall_best_distance, overall_best_vrn = min(scored, key=lambda t: t[1])

    matches = sorted(
        (DuplicateMatch(upload_id, float(distance), ref_vrn)
         for upload_id, distance, ref_vrn in scored
         if distance <= hamming_max and ref_vrn != claimed_vrn),
        key=lambda m: m.score,
    )
    is_dup = bool(matches)

    if is_dup:
        ids = ", ".join(m.upload_id for m in matches)
        reason = f"near-duplicate (pHash) of {len(matches)} reference(s) [{ids}] filed under a different VRN"
    elif overall_best_distance <= hamming_max and overall_best_vrn == claimed_vrn:
        reason = "near-duplicate (pHash) but under the same claimed VRN — treated as an honest re-upload"
    else:
        reason = f"closest pHash match hamming={overall_best_distance}, above threshold {hamming_max}"

    best = matches[0] if matches else None
    return DuplicateResult(is_dup, "phash",
                            best.upload_id if best else None, best.score if best else None,
                            best.claimed_vrn if best else None, reason, matches)


def _siglip_check(conn: sqlite3.Connection, *, image_path: str, image_type: str, claimed_vrn: str,
                   exclude_upload_id: str | None, similarity_min: float,
                   plate_bbox: tuple[float, float, float, float] | None) -> DuplicateResult:
    query_vec = compute_siglip_embedding(image_path, plate_bbox)
    candidates = db.fetch_reference_siglip_embeddings(conn, image_type, exclude_upload_id)
    if not candidates:
        return DuplicateResult(False, "none", None, None, None,
                                "no reference images with a vector embedding for this type yet")

    scored = [(upload_id, _cosine_similarity(query_vec, db.unpack_embedding(blob)), ref_vrn)
              for upload_id, blob, ref_vrn in candidates]
    overall_best_upload_id, overall_best_similarity, overall_best_vrn = max(scored, key=lambda t: t[1])

    matches = sorted(
        (DuplicateMatch(upload_id, similarity, ref_vrn)
         for upload_id, similarity, ref_vrn in scored
         if similarity >= similarity_min and ref_vrn != claimed_vrn),
        key=lambda m: -m.score,
    )
    is_dup = bool(matches)

    if is_dup:
        ids = ", ".join(m.upload_id for m in matches)
        reason = f"near-duplicate (SigLIP) of {len(matches)} reference(s) [{ids}] filed under a different VRN"
    elif overall_best_similarity >= similarity_min and overall_best_vrn == claimed_vrn:
        reason = "near-duplicate (SigLIP) but under the same claimed VRN — treated as an honest re-upload"
    else:
        reason = f"closest SigLIP match similarity={overall_best_similarity:.4f}, below threshold {similarity_min}"

    best = matches[0] if matches else None
    return DuplicateResult(is_dup, "siglip",
                            best.upload_id if best else None, best.score if best else None,
                            best.claimed_vrn if best else None, reason, matches)


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
