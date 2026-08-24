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
    ``validators/make_model_check.py``). Model comparison only runs (and can flip the
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
