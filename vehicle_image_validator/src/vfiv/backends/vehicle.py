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
    cls_name: str = "truck"  # the COCO class that actually matched -- "truck" | "bus"


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
            name = r.names[int(b.cls)]
            if name in self._TRUCK and (best is None or float(b.conf) > best.conf):
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                best = Detection((x1, y1, x2, y2), float(b.conf), (W, H), cls_name=name)
        return best


_detector: Optional[VehicleDetector] = None


def get_vehicle_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector


# --- Claimed-vs-detected vehicle category (truck vs bus) ----------------------------
# Both truck and bus VRNs get issued against this platform, so "is this even a
# truck/bus" (the existing REJECT gate in front_image.py) isn't the whole story --
# a bus photo uploaded against a claimed truck (or vice versa) should also be
# flagged. Reuses whichever COCO class ``best_truck`` actually matched (see
# ``Detection.cls_name`` above) rather than a second model call.

VEHICLE_TYPES = ("truck", "bus")


def decide_vehicle_type_match(detected: str | None, claimed_vehicle_type: str | None) -> dict:
    """Pure decision logic -- does the DETECTED vehicle category (from a real YOLO
    detection, ``Detection.cls_name`` -- "truck" | "bus" | None if nothing was
    detected) agree with a CLAIMED category ("truck" | "bus", case-insensitive)?

    Never a solo REJECT, only ever PASS/MANUAL_REVIEW/UNREADABLE-as-MANUAL_REVIEW:
    ``VehicleDetector`` is an off-the-shelf COCO model (``yolov8n.pt`` in this dev
    environment), explicitly documented elsewhere in this codebase as weak on
    Indian trucks -- confusing a covered/box truck for a bus (or a mini-bus for a
    truck) is a real, plausible failure mode for a generic detector, not the kind
    of confident signal worth auto-rejecting a genuine upload over. A mismatch is
    a lead for a human reviewer, same posture as every other soft signal in this
    codebase (colour-histogram, embedding-similarity, framing-completeness)."""
    claimed = (claimed_vehicle_type or "").strip().lower()
    if claimed not in VEHICLE_TYPES:
        return {"decision": "MANUAL_REVIEW", "status": "UNREADABLE",
                "reason": f"unknown claimed vehicle type {claimed_vehicle_type!r} (expected 'truck' or 'bus')"}
    if detected not in VEHICLE_TYPES:
        return {"decision": "MANUAL_REVIEW", "status": "UNREADABLE",
                "reason": "no truck/bus detected to compare against the claimed vehicle type"}
    if detected == claimed:
        return {"decision": "PASS", "status": "MATCH",
                "reason": f"detected vehicle type '{detected}' matches claimed '{claimed}'"}
    return {"decision": "MANUAL_REVIEW", "status": "MISMATCH",
            "reason": (f"detected vehicle type '{detected}' != claimed '{claimed}' -- flagged for "
                       f"human review rather than auto-rejected (see decide_vehicle_type_match's "
                       f"docstring for why)")}
