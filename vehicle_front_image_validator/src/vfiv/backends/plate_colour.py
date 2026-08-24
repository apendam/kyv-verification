"""HSV plate-colour classifier — copied verbatim (pure numpy, no model weights) from
truck_front_extractor/src/tfe/backends/real.py's ``_HSVColour`` class."""
from __future__ import annotations

import numpy as np


class HSVColourClassifier:
    """India plate colours: white(private) yellow(commercial) green(EV)
    black(rental) red(temp/govt)."""

    def predict(self, plate_crop: np.ndarray) -> dict[str, float]:
        import colorsys
        rgb = plate_crop.reshape(-1, 3).astype(np.float32) / 255.0
        hsv = np.array([colorsys.rgb_to_hsv(*p) for p in rgb])
        h, s, v = hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean()
        scores = {
            "white": float(v > 0.6 and s < 0.25),
            "yellow": float(0.10 < h < 0.20 and s > 0.35),
            "green": float(0.25 < h < 0.45 and s > 0.30),
            "black": float(v < 0.30),
            "red": float((h < 0.05 or h > 0.95) and s > 0.40),
        }
        tot = sum(scores.values()) or 1.0
        out = {k: round(val / tot, 4) for k, val in scores.items() if val}
        return out or {"unknown": 0.5}


def classify_plate_colour(plate_crop: np.ndarray) -> tuple[str, float]:
    """(colour, confidence 0-100) — argmax convenience wrapper."""
    probs = HSVColourClassifier().predict(plate_crop)
    colour = max(probs, key=probs.get)
    return colour, float(probs[colour]) * 100.0
