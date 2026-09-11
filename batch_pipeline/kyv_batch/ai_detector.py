"""Local no-reference AI-generation classifier (Organika/sdxl-detector via
transformers) -- a second, independent signal alongside the vision model's own
is_altered_or_ai_generated judgment, checked BEFORE that vision call so a clear
local hit skips the model call entirely (same "cheap signal gates the costlier
one" order as the vehicle-type check). Free (no API call) but not free to run:
the model is a few hundred MB and needs torch + transformers -- see
requirements.txt. Lazily loaded on first use, same shape as siglip.py.

Empirically weak on its own: tested during development against a real photo
and a confirmed AI-altered version of the SAME photo (a plate edit, not a full
re-generation) -- both scored above 99.9% "artificial", including a false
positive on the real one. Wired in as one more MANUAL_REVIEW-triggering
signal, never a REJECT gate, and never trusted as the sole tamper signal --
the vision model's own judgment (now checking for more specific tells: visible
generator watermarks, a composited-looking windshield sticker) still runs
regardless of what this returns.
"""
from __future__ import annotations

from typing import Optional

from . import config
from .device import resolve_device


class AIDetector:
    """Shared classifier, lazily warmed on first use."""

    def __init__(self):
        self.model = None
        self.proc = None

    def warmup(self):
        from transformers import AutoModelForImageClassification, AutoProcessor
        mid = config.AI_DETECTOR_MODEL
        # AutoProcessor (not AutoImageProcessor) resolves to a pure-PIL image
        # processor for this model family -- avoids an extra torchvision
        # dependency for something this small.
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForImageClassification.from_pretrained(mid).to(
            resolve_device(config.AI_DETECTOR_DEVICE)).eval()

    def _ensure(self):
        if self.model is None:
            self.warmup()

    def score_artificial(self, image_path: str) -> float:
        """Returns P(artificial) in [0, 1] -- the model's own "artificial" label
        probability, whatever else it also outputs."""
        import torch
        from PIL import Image
        self._ensure()
        img = Image.open(image_path).convert("RGB")
        inputs = self.proc(images=img, return_tensors="pt").to(resolve_device(config.AI_DETECTOR_DEVICE))
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]
        for idx, label in self.model.config.id2label.items():
            if label.lower() == "artificial":
                return float(probs[idx])
        return 0.0


_detector: Optional[AIDetector] = None


def get_ai_detector() -> AIDetector:
    global _detector
    if _detector is None:
        _detector = AIDetector()
    return _detector
