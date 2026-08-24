"""Q3: extract the make (painted word + optional logo) and match Parivahan's maker."""

from __future__ import annotations

from typing import Any

from ..core import (FieldVerification, LogoClassifier, Region, RegionDetector,
                    TextReader, VerificationStatus, blend_score, decide_status)
from .aliases import match_make


class MakeVerifier:
    """read grille/front word (+ optional logo) -> canonicalise -> match claimed maker.

    Handles the brand<->legal-entity gap (EICHER <-> VE COMMERCIAL VEHICLES LTD) via the
    alias map in ``aliases.py``. The logo classifier is optional but covers logo-only
    trucks and stylised wordmarks.
    """

    def __init__(self, detector: RegionDetector, reader: TextReader,
                 logo_classifier: LogoClassifier | None = None):
        self.detector = detector
        self.reader = reader
        self.logo_classifier = logo_classifier

    def verify(self, image: Any, claimed_make: str) -> FieldVerification:
        regions = sorted(self.detector.detect(image), key=lambda r: r.score, reverse=True)
        if not regions:
            return FieldVerification("make", VerificationStatus.UNREADABLE, claimed_make,
                                     notes=["no region to read"])

        # gather word candidates + best logo hit across regions
        candidates = []
        logo_brand, logo_prob = None, 0.0
        for region in regions:
            candidates.extend(self.reader.read(image, region))
            if self.logo_classifier is not None:
                hit = self.logo_classifier.classify(image, region)
                if hit and hit[1] > logo_prob:
                    logo_brand, logo_prob = hit

        best: tuple | None = None  # (matched, blended, cand, mm)
        for cand in candidates:
            if not cand.text.strip():
                continue
            mm = match_make(cand.text, claimed_make, logo_brand, logo_prob)
            blended = blend_score(mm.score, cand.confidence)
            key = (mm.matched, blended)
            if best is None or key > (best[0], best[1]):
                best = (mm.matched, blended, cand, mm)

        # logo-only trucks: no readable word but a confident logo
        if best is None and logo_brand is not None:
            mm = match_make("", claimed_make, logo_brand, logo_prob)
            return FieldVerification(
                field="make",
                status=decide_status(mm.matched, logo_brand),
                claimed=claimed_make, extracted_raw=None, extracted_norm=logo_brand,
                match_score=blend_score(mm.score, logo_prob), read_confidence=logo_prob,
                evidence={"logo_brand": logo_brand, "logo_prob": logo_prob,
                          "method": mm.method, "claimed_brands": sorted(mm.claimed_brands)},
                notes=["matched on logo only (no readable wordmark)"],
            )

        if best is None:
            return FieldVerification("make", VerificationStatus.UNREADABLE, claimed_make,
                                     notes=["no readable make word or logo"])

        matched, blended, cand, mm = best
        return FieldVerification(
            field="make",
            status=decide_status(matched, cand.text or None),
            claimed=claimed_make,
            extracted_raw=cand.text,
            extracted_norm=" / ".join(sorted(mm.extracted_brands)) or cand.text,
            match_score=blended,
            read_confidence=cand.confidence,
            evidence={"method": mm.method, "logo_brand": logo_brand, "logo_prob": logo_prob,
                      "extracted_brands": sorted(mm.extracted_brands),
                      "claimed_brands": sorted(mm.claimed_brands), "source": cand.source},
        )
