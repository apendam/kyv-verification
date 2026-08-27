"""Shared image loading for the CV backends (YOLO/SigLIP need an RGB ndarray;
Rekognition/Claude accept a path or PIL.Image directly and convert internally)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgb_array(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = image if isinstance(image, Image.Image) else Image.open(image)
    return np.array(img.convert("RGB"))
