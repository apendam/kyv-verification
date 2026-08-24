"""Model backends for the detect + read stages.

These are the ONLY places that touch a real model. Nothing here fabricates a read:
adapters that need a third-party library import it lazily and raise a clear error if it
is missing. Wire these to the same real models you already run (Qwen2.5-VL, PaddleOCR,
fast-alpr / YOLO plate detector, a logo classifier).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .core import ReadCandidate, Region


class WholeImageDetector:
    """Trivial detector that returns the whole image as one region.

    Use with a VLM reader (which localises internally). For the plate path prefer a real
    plate detector (``FastAlprDetector`` / a YOLO box) so OCR never sees body text.
    """
    def __init__(self, label: str = "image"):
        self.label = label

    def detect(self, image: Any) -> list[Region]:
        return [Region(bbox=None, score=1.0, label=self.label, crop=image)]


class FastAlprDetector:
    """Real plate detector backed by ``fast-alpr`` (MIT). Returns tight plate boxes."""
    def __init__(self, detector_model: str = "yolo-v9-t-384-license-plate-end2end"):
        try:
            from fast_alpr.detector import LicensePlateDetector  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "fast-alpr not installed. `pip install fast-alpr` and wire the real "
                "detector, or pass your own RegionDetector.") from e
        self._det = LicensePlateDetector(detector_model=detector_model)

    def detect(self, image: Any) -> list[Region]:
        out = []
        for d in self._det.predict(image):  # API: adapt to your fast-alpr version
            x1, y1, x2, y2 = d.bounding_box
            out.append(Region(bbox=(int(x1), int(y1), int(x2), int(y2)),
                              score=float(getattr(d, "confidence", 1.0)), label="plate"))
        return out


class PaddleOCRReader:
    """Real OCR backend (PaddleOCR PP-OCRv5) reading a pre-cropped region."""
    def __init__(self, lang: str = "en", **kwargs):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "paddleocr not installed. `pip install paddleocr` or pass your own "
                "TextReader.") from e
        self._ocr = PaddleOCR(lang=lang, **kwargs)

    def read(self, image: Any, region: Region) -> list[ReadCandidate]:
        crop = region.crop if region.crop is not None else image  # crop upstream ideally
        result = self._ocr.ocr(crop)  # adapt parsing to your PaddleOCR version's schema
        cands: list[ReadCandidate] = []
        for line in (result or []):
            for _box, (text, conf) in (line or []):
                cands.append(ReadCandidate(text=text, confidence=float(conf),
                                           source="paddleocr"))
        cands.sort(key=lambda c: c.confidence, reverse=True)
        return cands


_PLATE_PROMPT = (
    "You are reading an Indian vehicle number plate. Look ONLY at the licence plate; "
    "ignore any other painted text on the vehicle body (owner names, phone numbers, "
    "'Horn OK Please', slogans). Return strict JSON: "
    '{"plate": "<registration number, letters/digits only>", "confidence": <0..1>}. '
    'If unreadable return {"plate": "", "confidence": 0}.')

_MAKE_PROMPT = (
    "Identify the vehicle MANUFACTURER (make) from the front of this truck/bus, using "
    "the badge/wordmark on the grille and the logo. Return strict JSON: "
    '{"make": "<brand as written, e.g. TATA/EICHER/BharatBenz>", "confidence": <0..1>}. '
    'If unreadable return {"make": "", "confidence": 0}.')


class VLMReader:
    """VLM-backed reader (e.g. Qwen2.5-VL / Gemini) for plate OR make.

    You inject ``generate(image, prompt) -> str`` — the actual call to YOUR real model
    (self-hosted Qwen via vLLM/HF, or a Gemini client). This adapter only builds the
    prompt and parses the JSON reply; it never invents a value.
    """
    def __init__(self, generate: Callable[[Any, str], str], mode: str,
                 source: str = "vlm"):
        if mode not in ("plate", "make"):
            raise ValueError("mode must be 'plate' or 'make'")
        self.generate = generate
        self.prompt = _PLATE_PROMPT if mode == "plate" else _MAKE_PROMPT
        self.key = "plate" if mode == "plate" else "make"
        self.source = source

    def read(self, image: Any, region: Region) -> list[ReadCandidate]:
        img = region.crop if region.crop is not None else image
        raw = self.generate(img, self.prompt)
        text, conf = self._parse(raw)
        return [ReadCandidate(text=text, confidence=conf, source=self.source)] if text else []

    def _parse(self, raw: str) -> tuple[str, float]:
        if not raw:
            return "", 0.0
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return raw.strip(), 0.0
        try:
            obj = json.loads(m.group(0))
            return str(obj.get(self.key, "")).strip(), float(obj.get("confidence", 0.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return "", 0.0
