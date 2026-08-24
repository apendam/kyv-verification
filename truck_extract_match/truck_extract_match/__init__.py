"""truck_extract_match: shared detect -> read -> normalise -> match module.

Q2 (VRN) and Q3 (Make) of the truck/bus verification workflow, sharing one skeleton.
Domain logic (Indian plate inference, brand<->maker aliases) is pure Python; model
backends live in ``adapters`` and must be wired to real models.
"""

from .core import (FieldVerification, LogoClassifier, ReadCandidate, Region,
                   RegionDetector, TextReader, VerificationStatus)
from .plate.pipeline import PlateVerifier
from .make.pipeline import MakeVerifier

__all__ = [
    "FieldVerification", "VerificationStatus", "Region", "ReadCandidate",
    "RegionDetector", "TextReader", "LogoClassifier",
    "PlateVerifier", "MakeVerifier",
]
