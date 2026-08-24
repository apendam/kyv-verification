"""YOLOv8 vehicle detector — copied from
truck_front_extractor/src/tfe/backends/real.py's ``_Vehicle`` class (Q1's real
truck/bus detection, feeding the front-gate).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vfiv import config
from vfiv.backends.device import resolve_device

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Detection:
    bbox: BBox
    conf: float
    frame_wh: tuple[int, int]


class VehicleDetector:
    """COCO truck/bus detector; falls back to downloadable yolov8n.pt in dev
    (no custom-trained weights in this environment)."""
    _TRUCK = {"truck", "bus"}

    def __init__(self):
        self.m = None

    def warmup(self):
        from ultralytics import YOLO
        p = config.YOLO_VEHICLE_WEIGHTS
        if not Path(p).exists() and p != "yolov8n.pt":
            print(f"[vfiv] vehicle weights {p!r} missing -> COCO yolov8n (dev)",
                  file=sys.stderr)
            p = "yolov8n.pt"
        self.m = YOLO(p)

    def best_truck(self, image) -> Optional[Detection]:
        if self.m is None:
            self.warmup()
        H, W = image.shape[:2]
        r = self.m(image, verbose=False, device=resolve_device(config.DEVICE))[0]
        best = None
        for b in r.boxes:
            if r.names[int(b.cls)] in self._TRUCK and (best is None or float(b.conf) > best.conf):
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                best = Detection((x1, y1, x2, y2), float(b.conf), (W, H))
        return best


_detector: Optional[VehicleDetector] = None


def get_vehicle_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector
