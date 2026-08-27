from typing import Any, Optional

from pydantic import BaseModel

from vfiv.schemas import Decision, FrontImageResult, VrnCheckResult


class ExperimentResult(BaseModel):
    """One test case run through the backend-selectable Q1/Q2/Q3 pipeline —
    the test/inference interface's result shape. Unlike production's CombinedResult,
    this exposes which backend was used at each stage and the raw per-vote detail for
    Q3's make check, so you can compare backends side by side."""
    q1_backend: str
    q1: FrontImageResult
    q2_backend: Optional[str] = None
    q2: Optional[VrnCheckResult] = None
    q3_make_backend: Optional[str] = None
    q3_model_backend: Optional[str] = None
    q3_make_status: Optional[str] = None          # MATCH | MISMATCH | UNREADABLE
    q3_make_votes: Optional[list[dict[str, Any]]] = None  # per-source: {source, extracted, confidence, matched, brands}
    q3_model_checked: bool = False
    q3_model_status: Optional[str] = None
    q3_model_extracted: Optional[str] = None
    overall_decision: Decision
    overall_reason: str
