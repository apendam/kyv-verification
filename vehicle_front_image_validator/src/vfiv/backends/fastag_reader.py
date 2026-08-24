"""FASTag sticker reader — decodes the tag's barcode + QR code directly, and reads
the printed human-readable serial via AWS Rekognition.

Three independent representations of the tag's identity live on the sticker:
  - the 1D barcode (Code128-style) — decoded directly, checksum-backed
  - the QR code (``<fastag_id>@<bank_code>``, the UPI-recharge payload) — decoded
    directly, with built-in error correction that tolerates glare/damage better
    than the 1D barcode
  - the human-readable digits printed below the barcode — read via OCR, the only
    genuinely fuzzy/error-prone source of the three

``validators/fastag_check.py`` cross-checks all three against each other as well as
against the claimed value — this module only does the raw reads, no decisioning.

Needs the system ``libzbar0`` shared library (``apt-get install libzbar0`` on
Debian/Ubuntu) for ``pyzbar`` to import — not just a pip package, a real OS-level
dependency worth calling out in deploy docs alongside AWS credentials.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import boto3
from PIL import Image

from vfiv import config


class FastagReadError(RuntimeError):
    """Raised when AWS credentials are missing/invalid — mirrors
    ``backends.rekognition.RekognitionCredentialError``, kept separate since this
    reads plain text/barcodes, not VRN-specific plate parsing."""


@dataclass
class DecodedCode:
    symbology: str  # e.g. "CODE128", "QRCODE"
    data: str


@dataclass
class FastagRead:
    decoded_codes: list[DecodedCode]
    printed_id_text: Optional[str]  # best-effort OCR'd printed barcode number


_CREDENTIAL_ERROR_TOKENS = (
    "InvalidClientTokenId", "ExpiredToken", "UnrecognizedClientException",
    "security token", "AccessDenied", "credentials",
)

_PRINTED_ID_RE = re.compile(r"^[\d\-]{8,}$")


def decode_codes(image) -> list[DecodedCode]:
    """Direct barcode/QR decode via pyzbar (zbar). Returns an empty list if nothing
    decodes — a normal, expected outcome on a poor-quality photo, not an error."""
    from pyzbar.pyzbar import decode as zbar_decode

    img = image if isinstance(image, Image.Image) else Image.open(image)
    results = zbar_decode(img.convert("RGB"))
    return [DecodedCode(symbology=r.type, data=r.data.decode("utf-8", errors="replace"))
            for r in results]


def parse_qr_payload(data: str) -> tuple[Optional[str], Optional[str]]:
    """``<fastag_id>@<bank_code>`` (the UPI-recharge payload format) -> (fastag_id,
    bank_code). Returns (None, None) if the payload isn't shaped like that — e.g. a
    QR code that turned out to be something unrelated."""
    if data.count("@") != 1:
        return None, None
    fastag_id, bank_code = data.split("@", 1)
    fastag_id, bank_code = fastag_id.strip(), bank_code.strip()
    if not fastag_id or not bank_code:
        return None, None
    return fastag_id, bank_code


class _FastagTextReader:
    """One instance per process — boto3 client is reused, same pattern as
    ``backends.rekognition.RekognitionPlateDetector``."""

    def __init__(self, region: str | None = None):
        self.client = boto3.client("rekognition", region_name=region or config.AWS_REKOGNITION_REGION)

    def _image_bytes(self, image, max_bytes: int = 4_900_000) -> bytes:
        img = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
        scale = (max_bytes / len(data)) ** 0.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def read_lines(self, image, min_conf: float = 45.0) -> list[str]:
        resp = self.client.detect_text(Image={"Bytes": self._image_bytes(image)})
        return [d["DetectedText"].strip() for d in resp.get("TextDetections", [])
                if d["Type"] == "LINE" and d["Confidence"] >= min_conf]


def _pick_printed_id(lines: list[str]) -> Optional[str]:
    """Best-effort: the longest digits-and-hyphens line (the barcode's printed
    human-readable number, e.g. '607469-009-0874936')."""
    candidates = [ln for ln in lines if _PRINTED_ID_RE.match(ln)]
    return max(candidates, key=len) if candidates else None


def read_fastag(image, reader: _FastagTextReader | None = None) -> FastagRead:
    """Raises ``FastagReadError`` only on an AWS credential failure — a photo with
    nothing decodable/legible is still a normal result (all fields None/empty),
    not an error; ``validators/fastag_check.py`` decides what that means."""
    decoded = decode_codes(image)

    reader = reader or _FastagTextReader()
    try:
        lines = reader.read_lines(image)
    except Exception as e:
        msg = str(e)
        if any(tok in msg for tok in _CREDENTIAL_ERROR_TOKENS):
            raise FastagReadError(
                "AWS Rekognition credentials are invalid or expired. "
                "Refresh AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.\n"
                f"(underlying error: {msg})"
            ) from e
        raise

    return FastagRead(
        decoded_codes=decoded,
        printed_id_text=_pick_printed_id(lines),
    )
