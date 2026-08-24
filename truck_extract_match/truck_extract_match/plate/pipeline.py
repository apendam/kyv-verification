"""Q2: verify the plate number (VRN) against the claimed DB VRN."""

from __future__ import annotations

from typing import Any

from ..core import (FieldVerification, ReadCandidate, Region, RegionDetector,
                    TextReader, VerificationStatus, blend_score, decide_status)
from .format import match_vrn, normalize_vrn, parse_and_correct


class PlateVerifier:
    """detect plate box -> OCR the crop -> confusion-aware match to claimed VRN.

    Reading only the plate crop is what structurally excludes painted body text
    ("Horn OK Please", phone numbers): that text is outside the box and never read.
    """

    def __init__(self, detector: RegionDetector, reader: TextReader,
                 max_confusable_edits: int = 1, min_read_confidence: float = 0.0):
        self.detector = detector
        self.reader = reader
        self.max_confusable_edits = max_confusable_edits
        self.min_read_confidence = min_read_confidence

    def verify(self, image: Any, claimed_vrn: str) -> FieldVerification:
        regions = sorted(self.detector.detect(image), key=lambda r: r.score, reverse=True)
        if not regions:
            return FieldVerification("vrn", VerificationStatus.UNREADABLE, claimed_vrn,
                                     notes=["no plate detected"])

        best: tuple | None = None  # (matched, blended, cand, region, vmatch)
        for region in regions:
            for cand in self.reader.read(image, region):
                if cand.confidence < self.min_read_confidence:
                    continue
                norm = normalize_vrn(cand.text)
                if not norm:
                    continue
                vm = match_vrn(norm, claimed_vrn, self.max_confusable_edits)
                blended = blend_score(vm.score, cand.confidence)
                key = (vm.matched, blended)
                if best is None or key > (best[0], best[1]):
                    best = (vm.matched, blended, cand, region, vm)

        if best is None:
            return FieldVerification("vrn", VerificationStatus.UNREADABLE, claimed_vrn,
                                     notes=["plate detected but no readable text"])

        matched, blended, cand, region, vm = best
        parse = parse_and_correct(cand.text)
        notes = list(parse.notes)
        if vm.inferred:
            notes.append(f"smudge-inferred: read '{vm.read_norm}' accepted as "
                         f"'{vm.claimed_norm}' (confusable dist {vm.distance})")
        return FieldVerification(
            field="vrn",
            status=decide_status(matched, vm.read_norm),
            claimed=vm.claimed_norm,
            extracted_raw=cand.text,
            extracted_norm=parse.corrected or vm.read_norm,
            match_score=blended,
            read_confidence=cand.confidence,
            evidence={"bbox": region.bbox, "confusable_distance": vm.distance,
                      "grammar_valid": parse.valid, "source": cand.source},
            notes=notes,
        )
