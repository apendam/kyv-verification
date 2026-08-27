from typing import Literal, Optional

from pydantic import BaseModel

Decision = Literal["PASS", "REJECT", "MANUAL_REVIEW"]


class FrontImageResult(BaseModel):
    decision: Decision
    reason: str
    checked: bool
    vehicle_type: Optional[str] = None
    view: Optional[str] = None
    is_front: Optional[bool] = None
    front_complete: Optional[bool] = None
    is_screenshot: Optional[bool] = None
    is_photo_of_photo: Optional[bool] = None
    ai_generated: Optional[bool] = None
    ai_confidence: Optional[float] = None
    confidence: Optional[float] = None
    duplicate_is_suspect: Optional[bool] = None
    error: Optional[str] = None


class VrnCheckResult(BaseModel):
    decision: Decision
    status: str  # truck_extract_match.core.VerificationStatus: MATCH | MISMATCH | UNREADABLE
    claimed_vrn: str
    reason: str
    checked: bool
    extracted_raw: Optional[str] = None
    extracted_norm: Optional[str] = None
    match_score: Optional[float] = None       # 0..1
    read_confidence: Optional[float] = None   # 0..100
    confusable_distance: Optional[int] = None
    grammar_valid: Optional[bool] = None
    inferred: Optional[bool] = None           # smudge-inferred match
    plate_colour: Optional[str] = None
    colour_confidence: Optional[float] = None  # 0..100
    error: Optional[str] = None


class MakeModelCheckResult(BaseModel):
    """Q3: make comparison always runs — checked against TWO independent real-model
    sources (SigLIP zero-shot classifier + AWS Rekognition's painted-brand-text read),
    matched if EITHER agrees with the claimed make (each has known blind spots — see
    ``validators/front_image/make_model_check.py``). Model comparison only runs (and can flip the
    decision) when ``model_checked`` is True."""
    decision: Decision
    reason: str
    checked: bool
    claimed_make: str
    make_status: Optional[str] = None  # MATCH | MISMATCH | UNREADABLE
    extracted_make_siglip: Optional[str] = None
    siglip_confidence: Optional[float] = None       # 0..100
    extracted_make_rekognition: Optional[str] = None  # None if Rekognition found no brand text
    make_match_via: Optional[str] = None             # "siglip" | "rekognition" | "both" | None
    claimed_make_brands: Optional[list[str]] = None
    claimed_model: Optional[str] = None
    model_checked: bool = False               # was the model comparison actually enforced?
    model_status: Optional[str] = None        # MATCH | MISMATCH | UNREADABLE, None if not checked
    extracted_model_raw: Optional[str] = None
    model_match_score: Optional[float] = None  # 0..1
    model_confidence: Optional[float] = None   # 0..100 (VLM read confidence)
    error: Optional[str] = None


class DuplicateCheckResult(BaseModel):
    """Cross-upload near-duplicate check — NOT one of Q1/Q2/Q3, and not folded into
    ``CombinedResult``'s decision. Flags a MANUAL_REVIEW lead when this image is a
    near-duplicate (by SigLIP embedding cosine similarity) of a PRIOR upload filed
    under a DIFFERENT claimed VRN — the "same photo, swapped plate" fraud pattern.
    A near-duplicate under the SAME VRN is an ordinary re-upload and is never
    flagged. See ``validators/duplicate_check.py``."""
    decision: Decision
    reason: str
    checked: bool
    claimed_vrn: str
    is_duplicate_suspect: bool
    best_match_id: Optional[str] = None
    best_match_similarity: Optional[float] = None  # cosine similarity, 0..1
    best_match_vrn: Optional[str] = None
    error: Optional[str] = None


class FastagCheckResult(BaseModel):
    """FASTag sticker validator — cross-checks THREE independent identity sources
    (QR decode, 1D-barcode decode, OCR'd printed digits) against EACH OTHER as well
    as against the claimed value; a disagreement between sources that were each
    legibly read is itself a REJECT-worthy tamper signal, checked before the
    match-against-claim step. See ``validators/fastag_image/fastag_check.py``."""
    decision: Decision
    reason: str
    checked: bool
    claimed_fastag_id: str
    claimed_bank_code: Optional[str] = None
    decoded_sources: Optional[dict[str, str]] = None  # e.g. {"qr": "...", "barcode:code128": "..."}
    extracted_printed_id: Optional[str] = None
    matched_via: Optional[str] = None  # "qr" | "barcode:<symbology>" | "ocr" | None
    error: Optional[str] = None


class AxleCountResult(BaseModel):
    """Just the axle-count piece of ``SideImageCheckResult`` — same VLM judgment
    call (``classify_axle_count``/``decide_axle_count``), exposed standalone so it
    can be tested in isolation from identity-binding/duplicate. See
    ``validators/side_image/side_image_check.py``."""
    decision: Decision
    status: Optional[str] = None  # MATCH | MISMATCH | UNREADABLE
    checked: bool
    claimed_axle_count: int
    axle_count: Optional[int] = None
    axle_confidence: Optional[float] = None  # 0..100
    lift_axle_suspected: Optional[bool] = None
    reason: str
    error: Optional[str] = None


class SideImageIdentityResult(BaseModel):
    """Just the identity-binding piece of ``SideImageCheckResult`` — routed by
    ``SideImageTypeClassifier`` into vrn_visible / corner_view / pure_side_profile
    (decreasing reliability, see the module docstring), exposed standalone so it
    can be tested independent of axle-count/duplicate. See
    ``validators/side_image/side_image_check.py``."""
    decision: Decision
    checked: bool
    claimed_vrn: str
    claimed_make: str
    identity_bucket: Optional[str] = None  # vrn_visible | corner_view | pure_side_profile
    make_read: Optional[str] = None
    make_matched: Optional[bool] = None
    front_similarity: Optional[float] = None  # 0..1, corner_view bucket only
    vrn_status: Optional[str] = None  # MATCH | MISMATCH | UNREADABLE, vrn_visible bucket only
    reason: str
    error: Optional[str] = None


class SideImageCheckResult(BaseModel):
    """Side/axle-image validator — axle count (Claude judgment call, no dedicated
    detector wired) + identity-to-claimed-vehicle, routed by
    ``SideImageTypeClassifier`` into three buckets of DECREASING reliability
    (vrn_visible > corner_view > pure_side_profile — the last is NEVER a confident
    PASS on its own, see ``validators/side_image/side_image_check.py``). Overall ``decision``
    is the worst of axle / identity / duplicate, same REJECT > MANUAL_REVIEW >
    PASS ordering as ``CombinedResult``."""
    decision: Decision
    reason: str
    checked: bool
    claimed_vrn: str
    claimed_make: str
    claimed_axle_count: int
    axle_count: Optional[int] = None
    axle_status: Optional[str] = None  # MATCH | MISMATCH | UNREADABLE
    identity_bucket: Optional[str] = None  # vrn_visible | corner_view | pure_side_profile
    identity_decision: Optional[str] = None
    duplicate_is_suspect: Optional[bool] = None
    error: Optional[str] = None


class CombinedResult(BaseModel):
    """Single entry point per upload: Q1 gates Q2+Q3 (a Q1 REJECT/MANUAL_REVIEW wins
    outright and neither Q2 nor Q3 runs). Once Q1 PASSes, Q2 and Q3 both run
    independently (neither gates the other) so a single call always returns the
    complete audit trail; the overall ``decision`` takes the worst of the two:
    REJECT > MANUAL_REVIEW > PASS. See ``validators/combined.py``."""
    decision: Decision
    reason: str
    checked: bool
    front: FrontImageResult
    vrn: Optional[VrnCheckResult] = None
    make_model: Optional[MakeModelCheckResult] = None
    error: Optional[str] = None
