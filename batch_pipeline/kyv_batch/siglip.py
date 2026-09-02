"""Local SigLIP 2 image embedding -- the second, heavier duplicate-detection
signal (see duplicate.py), run only when pHash doesn't already flag a
duplicate. Same weights/approach as
vehicle_front_image_validator/src/vfiv/backends/siglip.py's ``embed_image``,
copied rather than imported since the two packages don't depend on each
other; unlike that module this file only needs the embedding, not the
zero-shot classifiers.

Free (no API call), but not free to run: the model (``google/siglip2-base-
patch16-512`` by default) is a few hundred MB and needs `torch` +
`transformers` installed -- see requirements.txt. Lazily loaded on first use
so importing this module (or the rest of the package) doesn't pay that cost
until a duplicate check actually falls through to the vector signal.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import config
from .device import resolve_device


class SigLipModel:
    """Shared embedding model, lazily warmed on first use."""

    def __init__(self):
        self.model = None
        self.proc = None

    def warmup(self):
        from transformers import AutoModel, AutoProcessor
        mid = config.SIGLIP_MODEL
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid).to(resolve_device(config.SIGLIP_DEVICE)).eval()

    def _ensure(self):
        if self.model is None:
            self.warmup()

    def embed_image(self, image) -> np.ndarray:
        import torch
        from PIL import Image
        self._ensure()
        img = image if isinstance(image, Image.Image) else Image.open(image)
        inp = self.proc(images=img, return_tensors="pt").to(resolve_device(config.SIGLIP_DEVICE))
        with torch.no_grad():
            feats = self.model.get_image_features(**inp)
        return feats[0].cpu().numpy()


_siglip: Optional[SigLipModel] = None


def get_siglip_model() -> SigLipModel:
    global _siglip
    if _siglip is None:
        _siglip = SigLipModel()
    return _siglip
