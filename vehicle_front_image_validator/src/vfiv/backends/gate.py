"""Front-gate fusion — copied from truck_front_extractor/src/tfe/gate/gate.py's
``_completeness_score``/``_fuse`` (Q1's real vehicle-detection + pose + completeness
pipeline, feeding ``front_image.classify_front_image``)."""
from __future__ import annotations

import numpy as np

from vfiv import config
from vfiv.backends.siglip import get_pose_classifier
from vfiv.backends.vehicle import Detection, get_vehicle_detector


def frontal_score(image, bbox) -> float:
    """Pose classifier -> P(front-facing). A three-quarter front view still shows
    the grille, headlights and front plate, so it counts as frontal; only pure
    side / rear views are rejected."""
    p = get_pose_classifier().predict(image, bbox)
    return p.get("front", 0.0) + p.get("front34", 0.0)


def completeness_score(bbox, frame_wh) -> float:
    """Heuristic 'is the whole truck in frame'. The truck bbox touching the left,
    top or right frame edge suggests the truck is cut off there; each clipped edge
    lowers the score. The bottom edge is ignored (trucks normally sit on the ground
    there). Also scaled by coverage so a tiny detection can't score high.

    Heuristic by nature — it can't see beyond the frame. Claude's judgment (already
    in the Q1 prompt) is the authoritative "cut off?" call; this is a real,
    model-backed cross-check."""
    (x1, y1, x2, y2), (W, H) = bbox, frame_wh
    coverage = ((x2 - x1) * (y2 - y1)) / float(W * H)
    margin = 0.01
    clipped = sum([x1 <= W * margin,
                   y1 <= H * margin,
                   x2 >= W * (1 - margin)])
    edge_factor = {0: 1.0, 1: 1.0, 2: 0.7, 3: 0.5}[clipped]
    cov_score = min(coverage / config.GATE_COVERAGE_MIN, 1.0)
    return cov_score * edge_factor


def fuse(truck: float, frontal: float, complete: float) -> float:
    """Weighted geometric mean — any near-zero sub-score tanks the gate."""
    w = np.array([0.4, 0.35, 0.25])
    s = np.array([max(truck, 1e-6), max(frontal, 1e-6), max(complete, 1e-6)])
    return float(np.exp((w * np.log(s)).sum()))


def run_gate(image: np.ndarray) -> dict:
    """Real CV portion of Q1: vehicle_type/view/is_front/front_complete/confidence.

    Returns a dict shaped for ``front_image.decide_front_image`` (merged with the
    narrowed Claude judgment-call fields upstream). ``vehicle_type`` is "truck" if
    a truck/bus was detected above threshold, else "other" (car/nothing detected —
    this backend can't distinguish car-vs-nothing, only truck/bus-vs-not).
    """
    det: Detection | None = get_vehicle_detector().best_truck(image)
    truck_conf = det.conf if det is not None else 0.0

    if det is None or truck_conf < config.GATE_TRUCK_MIN:
        return {
            "vehicle_type": "other", "view": "other", "is_front": False,
            "front_complete": False, "confidence": round(truck_conf * 100.0, 1),
            "gate_reason": "no_truck_detected",
        }

    pose = get_pose_classifier().predict(image, det.bbox)
    frontal = pose.get("front", 0.0) + pose.get("front34", 0.0)
    complete = completeness_score(det.bbox, det.frame_wh)
    conf = fuse(truck_conf, frontal, complete)

    view = max(pose, key=pose.get)  # front | front34 | side | rear
    is_front = frontal >= config.GATE_FRONTAL_MIN
    front_complete = conf >= config.GATE_ACCEPT_MIN and is_front

    return {
        "vehicle_type": "truck", "view": ("front" if view == "front34" else view),
        "is_front": is_front, "front_complete": front_complete,
        "confidence": round(conf * 100.0, 1),
        "gate_reason": None if front_complete else (
            "not_frontal" if not is_front else "low_gate_confidence"),
        "subscores": {"truck": round(truck_conf, 4), "frontal": round(frontal, 4),
                      "complete": round(complete, 4)},
    }
