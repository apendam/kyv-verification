"""SigLIP 2 zero-shot classifiers — copied from
truck_front_extractor/src/tfe/backends/real.py's ``_SigLIP``/``_SigLIPPose``/
``_SigLIPMake`` classes. One shared model serves both:
  - pose (Q1) — front/front34/side/rear, feeding the front-gate.
  - make (Q3) — zero-shot brand among a fixed Indian-truck manufacturer list.
    Model/variant (exact designation) is NOT covered here — SigLIP only does
    coarse brand classification; see backends/qwen.py for exact model reading.

Weights (``google/siglip2-base-patch16-512``) are already cached locally from prior
work in this environment — no fresh download needed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vfiv import config
from vfiv.backends.device import resolve_device


class SigLipModel:
    """Shared zero-shot classifier, lazily warmed on first use."""

    def __init__(self):
        self.model = None

    def warmup(self):
        from transformers import AutoModel, AutoProcessor
        mid = config.SIGLIP_MODEL
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid).to(resolve_device(config.DEVICE)).eval()

    def _ensure(self):
        if self.model is None:
            self.warmup()

    def zero_shot(self, crop, labels: dict[str, str]) -> dict[str, float]:
        """labels: {key -> prompt}. Returns {key -> prob} (softmax over keys)."""
        import torch
        from PIL import Image
        self._ensure()
        texts = list(labels.values())
        img = crop if isinstance(crop, Image.Image) else Image.fromarray(crop)
        inp = self.proc(text=texts, images=img, padding="max_length",
                        return_tensors="pt").to(resolve_device(config.DEVICE))
        with torch.no_grad():
            probs = self.model(**inp).logits_per_image.softmax(dim=-1)[0].tolist()
        return dict(zip(labels.keys(), probs))


class PoseClassifier:
    POSES = {
        "front": "the front view of a truck with headlights, grille and windshield",
        "front34": "a three-quarter front view of a truck",
        "side": "the side profile view of a truck",
        "rear": "the rear view of a truck with tail lights",
    }

    def __init__(self, siglip: SigLipModel):
        self.s = siglip

    def predict(self, image, bbox) -> dict[str, float]:
        x1, y1, x2, y2 = bbox
        crop = image[max(y1, 0):y2, max(x1, 0):x2]
        return self.s.zero_shot(crop, self.POSES)


class MakeClassifier:
    """Zero-shot brand among Indian truck makes. Exact model left to Qwen (opt-in)."""
    BRANDS = ["Tata", "Ashok Leyland", "Eicher", "BharatBenz", "Mahindra",
              "Volvo", "Force Motors", "SML Isuzu"]

    def __init__(self, siglip: SigLipModel):
        self.s = siglip

    def predict(self, truck_crop) -> dict:
        labels = {b: f"a photo of the front of a {b} truck" for b in self.BRANDS}
        probs = self.s.zero_shot(truck_crop, labels)
        make = max(probs, key=probs.get)
        return {"make": make, "make_confidence": float(probs[make]) * 100.0}


_siglip: Optional[SigLipModel] = None
_pose: Optional[PoseClassifier] = None
_make: Optional[MakeClassifier] = None


def _shared_siglip() -> SigLipModel:
    global _siglip
    if _siglip is None:
        _siglip = SigLipModel()
    return _siglip


def get_pose_classifier() -> PoseClassifier:
    global _pose
    if _pose is None:
        _pose = PoseClassifier(_shared_siglip())
    return _pose


def get_make_classifier() -> MakeClassifier:
    global _make
    if _make is None:
        _make = MakeClassifier(_shared_siglip())
    return _make
