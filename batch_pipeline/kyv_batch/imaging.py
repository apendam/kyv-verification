"""Two image-prep steps for the duplicate check, both working on a copy —
the original file is never touched:

  mask_normalized_box — blacks out the vehicle registration plate/VRN before
    an image is hashed/embedded, so near-duplicate comparison is based on
    the rest of the truck (body, cabin, decorations) rather than the plate
    area — robust to someone reusing the same photo under a different
    claimed VRN by only editing the plate, and not biased by whatever text
    happens to be on it.

  crop_normalized_box — crops to just the vehicle (with a small margin) so
    the comparison isn't influenced by background/environment (road, sky,
    other vehicles, depot buildings) — two different trucks photographed in
    the same spot shouldn't look alike, and the same truck photographed in
    different surroundings shouldn't look different.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

# A degenerate/all-zero box (the schema's "not visible" sentinel) or one
# that doesn't resolve to at least this many pixels either way isn't worth
# masking -- almost certainly a model that returned the "not visible"
# default rather than a real (if tiny) plate location.
_MIN_BOX_FRACTION = 0.001


def mask_normalized_box(image_path: str, bbox: tuple[float, float, float, float]) -> str:
    """Blacks out `bbox` (x_min, y_min, x_max, y_max, each a 0-1 fraction of
    image width/height) on a copy of the image at `image_path`, and returns
    the path to that copy (a new temp file — the original is untouched).
    Returns `image_path` unchanged if `bbox` is degenerate (too small / not
    a real box), rather than writing a no-op copy.
    """
    x_min, y_min, x_max, y_max = bbox
    if x_max - x_min < _MIN_BOX_FRACTION or y_max - y_min < _MIN_BOX_FRACTION:
        return image_path

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    box = (
        max(0, min(width, round(x_min * width))),
        max(0, min(height, round(y_min * height))),
        max(0, min(width, round(x_max * width))),
        max(0, min(height, round(y_max * height))),
    )
    ImageDraw.Draw(image).rectangle(box, fill=(0, 0, 0))

    suffix = Path(image_path).suffix or ".jpg"
    fd, out_path = tempfile.mkstemp(suffix=suffix, prefix="plate-masked-")
    os.close(fd)
    image.save(out_path)
    return out_path


def crop_normalized_box(image_path: str, bbox: tuple[float, float, float, float],
                         margin: float = 0.08) -> str:
    """Crops to `bbox` (x_min, y_min, x_max, y_max, each a 0-1 fraction of
    image width/height) plus a proportional `margin` on each side (a
    fraction of that box's own width/height, so a tight or loose model
    localization is tolerated the same way), on a copy of the image at
    `image_path`, and returns the path to that copy (a new temp file — the
    original is untouched). Returns `image_path` unchanged if `bbox` is
    degenerate (too small / not a real box), rather than writing a no-op
    copy — same convention as `mask_normalized_box`.
    """
    x_min, y_min, x_max, y_max = bbox
    if x_max - x_min < _MIN_BOX_FRACTION or y_max - y_min < _MIN_BOX_FRACTION:
        return image_path

    box_width, box_height = x_max - x_min, y_max - y_min
    x_min -= box_width * margin
    x_max += box_width * margin
    y_min -= box_height * margin
    y_max += box_height * margin

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    box = (
        max(0, min(width, round(x_min * width))),
        max(0, min(height, round(y_min * height))),
        max(0, min(width, round(x_max * width))),
        max(0, min(height, round(y_max * height))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return image_path
    cropped = image.crop(box)

    suffix = Path(image_path).suffix or ".jpg"
    fd, out_path = tempfile.mkstemp(suffix=suffix, prefix="vehicle-cropped-")
    os.close(fd)
    cropped.save(out_path)
    return out_path
