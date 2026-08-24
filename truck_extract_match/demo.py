"""End-to-end demo of the extract-and-match LOGIC using a stub reader.

IMPORTANT: this exercises the deterministic normalise/infer/match layer only. The stub
reader below returns fixed example strings to stand in for OCR/VLM output so you can see
the decision logic without a GPU. In production you MUST wire a real detector + reader
from ``adapters.py`` (fast-alpr / PaddleOCR / Qwen2.5-VL) — do not ship the stub.

Run:  PYTHONPATH=. python demo.py
"""

from truck_extract_match import MakeVerifier, PlateVerifier
from truck_extract_match.adapters import WholeImageDetector
from truck_extract_match.core import ReadCandidate, Region


class _StubReader:
    """Stands in for a real OCR/VLM reader. Returns a preset read for demo purposes."""
    def __init__(self, text: str, confidence: float = 0.8):
        self._cand = ReadCandidate(text=text, confidence=confidence, source="stub")

    def read(self, image, region: Region):
        return [self._cand]


def _show(title: str, r):
    print(f"\n=== {title} ===")
    print(f"  status     : {r.status.value}")
    print(f"  claimed    : {r.claimed}")
    print(f"  read (raw) : {r.extracted_raw}")
    print(f"  normalised : {r.extracted_norm}")
    print(f"  score      : {r.match_score}  (read_conf {r.read_confidence})")
    if r.notes:
        print(f"  notes      : {'; '.join(r.notes)}")


if __name__ == "__main__":
    img = object()  # placeholder; the stub reader ignores it
    det = WholeImageDetector()

    # Q2 — clean read
    _show("VRN clean", PlateVerifier(det, _StubReader("MH12AB1234")).verify(img, "MH12AB1234"))
    # Q2 — smudged: OCR read district 0 as O and series B as 8 -> inferred match
    _show("VRN smudged", PlateVerifier(det, _StubReader("MHO2A81234")).verify(img, "MH02AB1234"))
    # Q2 — genuine mismatch -> auto-reject
    _show("VRN mismatch", PlateVerifier(det, _StubReader("KA05C9999")).verify(img, "MH02AB1234"))

    # Q3 — painted brand vs Parivahan legal entity
    _show("Make Eicher", MakeVerifier(det, _StubReader("EICHER")).verify(img, "VE COMMERCIAL VEHICLES LTD"))
    _show("Make BharatBenz", MakeVerifier(det, _StubReader("BharatBenz")).verify(img, "DAIMLER INDIA COMMERCIAL VEHICLES PVT LTD"))
    _show("Make mismatch", MakeVerifier(det, _StubReader("TATA")).verify(img, "ASHOK LEYLAND LTD"))
