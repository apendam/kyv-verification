"""Torch device resolution -- same cuda -> mps -> cpu fallback as
vehicle_front_image_validator/src/vfiv/backends/device.py, copied rather than
imported since the two packages don't depend on each other.
"""
from __future__ import annotations

from typing import Optional

_DEV_CACHE: Optional[str] = None


def resolve_device(want: str = "cuda") -> str:
    global _DEV_CACHE
    if _DEV_CACHE is not None:
        return _DEV_CACHE
    import torch
    if want.startswith("cuda") and torch.cuda.is_available():
        dev = want
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    _DEV_CACHE = dev
    return dev
