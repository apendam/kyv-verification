"""VRN and maker matching — thin wrappers around the pure-logic matcher already
in the sibling `truck_extract_match` package (confusion-aware plate matching,
brand<->legal-entity canonicalisation). No model calls in this file at all;
reusing it here instead of re-deriving fuzzy-match rules for a second time.

Install it once: `pip install -e ../truck_extract_match` (see this package's
own requirements.txt).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config

try:
    from truck_extract_match.plate.format import match_vrn
    from truck_extract_match.make.aliases import match_make
except ImportError as exc:  # pragma: no cover - surfaced at import time, not silently
    raise ImportError(
        "truck_extract_match isn't installed. From this repo's root: "
        "pip install -e ./truck_extract_match"
    ) from exc


@dataclass
class VrnVerdict:
    outcome: str  # "match" | "mismatch_similar" | "mismatch_other"
    distance: int
    score: float
    read_norm: str
    claimed_norm: str
    inferred: bool


def classify_vrn(read_text: str, claimed_vrn: str,
                  max_confusable_edits: int = config.VRN_MAX_CONFUSABLE_EDITS,
                  similar_char_max_distance: int = config.VRN_SIMILAR_CHAR_MAX_DISTANCE
                  ) -> VrnVerdict:
    """Maps the raw plate read onto the flowchart's three mismatch-aware
    outcomes. `match_vrn` already tolerates up to `max_confusable_edits`
    confusable substitutions (0/O, 1/I, 5/S, ...) as a match. Beyond that, a
    still-small confusable-edit-distance reads as "probably a smudge/misread,
    same plate" (-> manual review); a large one reads as "a different plate
    entirely" (-> reject). Both thresholds are config knobs, not model calls.
    """
    m = match_vrn(read_text, claimed_vrn, max_confusable_edits)
    if m.matched:
        outcome = "match"
    elif m.distance <= similar_char_max_distance:
        outcome = "mismatch_similar"
    else:
        outcome = "mismatch_other"
    return VrnVerdict(outcome, m.distance, m.score, m.read_norm, m.claimed_norm, m.inferred)


@dataclass
class MakerVerdict:
    outcome: str  # "match" | "mismatch"
    score: float
    extracted_brands: list[str]
    claimed_brands: list[str]
    method: str


def classify_maker(read_text: str, claimed_make: str) -> MakerVerdict:
    m = match_make(read_text, claimed_make)
    return MakerVerdict(
        outcome="match" if m.matched else "mismatch",
        score=m.score,
        extracted_brands=sorted(m.extracted_brands),
        claimed_brands=sorted(m.claimed_brands),
        method=m.method,
    )
