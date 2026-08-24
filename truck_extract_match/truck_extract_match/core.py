"""Shared contracts for the extract-and-match pipeline (Q2 VRN + Q3 Make).

Both fields follow the same skeleton:

    detect region  ->  read text/logo  ->  domain-normalise  ->  match claimed  ->  status

The domain-specific value lives in ``plate/format.py`` (Indian plate inference) and
``make/aliases.py`` (brand -> registered-maker canonicalisation). The model-backed
stages (detector / OCR / VLM / logo classifier) are Protocols implemented in
``adapters.py`` and MUST be wired to real models — nothing here fabricates a read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class VerificationStatus(str, Enum):
    MATCH = "MATCH"            # extracted value agrees with claimed  -> continue flow
    MISMATCH = "MISMATCH"      # extracted value contradicts claimed  -> auto-reject
    UNREADABLE = "UNREADABLE"  # nothing usable extracted             -> caller decides (reject / manual)


@dataclass
class Region:
    """A detected region of interest (plate box, grille/nameplate box, or whole image)."""
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2); None = whole image
    score: float = 1.0                       # detector confidence
    label: str = ""                          # e.g. "plate", "grille", "logo"
    crop: Any = None                         # optional pre-cropped image (np.ndarray / PIL)


@dataclass
class ReadCandidate:
    """One hypothesis returned by a TextReader / VLM. Best candidate first."""
    text: str
    confidence: float = 0.0    # 0..1 reader confidence, 0 if the backend gives none
    source: str = ""           # "paddleocr" | "parseq" | "qwen2.5-vl" | "logo-clf" ...
    extra: dict = field(default_factory=dict)  # per-char probs, logo class, bbox, etc.


@dataclass
class FieldVerification:
    """Result of verifying one field (VRN or Make) against its claimed value."""
    field: str                         # "vrn" | "make"
    status: VerificationStatus
    claimed: str
    extracted_raw: str | None = None   # what the reader actually saw
    extracted_norm: str | None = None  # canonical / grammar-corrected form
    match_score: float = 0.0           # 0..1 agreement strength
    read_confidence: float = 0.0       # 0..1 from the reader/detector
    evidence: dict = field(default_factory=dict)  # bbox, distance, alias hit, logo, notes
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status is VerificationStatus.MATCH

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d


@runtime_checkable
class RegionDetector(Protocol):
    """Locates the region(s) to read. For VLM readers use ``WholeImageDetector``."""
    def detect(self, image: Any) -> list[Region]: ...


@runtime_checkable
class TextReader(Protocol):
    """Reads text from a region. Return best-first candidates; [] if nothing read."""
    def read(self, image: Any, region: Region) -> list[ReadCandidate]: ...


@runtime_checkable
class LogoClassifier(Protocol):
    """Optional: classifies the manufacturer logo. Return (brand_key, prob) or None."""
    def classify(self, image: Any, region: Region) -> tuple[str, float] | None: ...


# --- shared decision helpers -------------------------------------------------

def blend_score(match_score: float, read_confidence: float) -> float:
    """Combine agreement strength with reader confidence into one 0..1 score."""
    if read_confidence <= 0:
        return round(match_score, 4)
    return round(0.7 * match_score + 0.3 * read_confidence, 4)


def decide_status(matched: bool | None, extracted: str | None) -> VerificationStatus:
    if not extracted:
        return VerificationStatus.UNREADABLE
    if matched:
        return VerificationStatus.MATCH
    return VerificationStatus.MISMATCH
