"""Deterministic barcode/QR decoding for the FASTag flow -- no model call, no
guessing. A barcode or QR code is built to be machine-decoded, not visually
read the way a plate or maker's badge is, so this uses a real decoder
(`zxingcpp`, the Python binding for the ZXing-C++ library) instead of asking
a vision model to transcribe it.

"Readable" is defined entirely by decode success, per the agreed design:
  - barcode readable = the decoder recovers a value at all (a partially
    obscured/cropped barcode fails to decode outright, rather than
    returning a half-guessed number)
  - QR readable = the decoder recovers a payload AND it matches the expected
    FASTag QR schema (see parse_fastag_qr_payload) -- a garbled or
    wrong-format payload is treated as unreadable, not force-parsed

zxingcpp's `valid` flag on a decoded barcode reflects whether the symbology's
own checksum digit (built into the stripe pattern itself, e.g. for Code128/
EAN-13) reconciles -- this is the self-consistency signal the "can we derive
the same barcode from the stripes" check needs, with no extra logic required.

**UNCONFIRMED**: `parse_fastag_qr_payload`'s expected format below is a
placeholder, not a verified NETC/NPCI FASTag QR encoding -- it hasn't been
checked against a real decoded tag. Replace it with the real format (ideally
from a real decoded QR string or NETC documentation) before relying on this
in production; until then, every QR will read as unreadable unless it
happens to match this guessed shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import zxingcpp
from PIL import Image

# 1D symbologies actually printed on a FASTag barcode strip -- restricting the
# decode to these (plus QRCode separately) avoids false positives from
# unrelated patterns elsewhere in the photo.
_BARCODE_FORMATS = (
    zxingcpp.BarcodeFormat.Code128 | zxingcpp.BarcodeFormat.Code39
    | zxingcpp.BarcodeFormat.EAN13 | zxingcpp.BarcodeFormat.ITF
)


@dataclass
class BarcodeRead:
    readable: bool
    value: str | None
    valid: bool  # the symbology's own checksum reconciled -- see module docstring


@dataclass
class QrRead:
    readable: bool
    tag_id: str | None
    bank_code: str | None
    raw_text: str | None


def read_barcode(image_path: str) -> BarcodeRead:
    results = zxingcpp.read_barcodes(Image.open(image_path), formats=_BARCODE_FORMATS)
    if not results:
        return BarcodeRead(readable=False, value=None, valid=False)
    best = results[0]
    return BarcodeRead(readable=True, value=best.text, valid=bool(best.valid))


# UNCONFIRMED placeholder -- see module docstring. Assumes a delimited
# "TAGID:<id>;BANK:<code>" payload; adjust once the real format is known.
_QR_PAYLOAD_RE = re.compile(r"TAGID:([A-Za-z0-9]+);BANK:([A-Za-z0-9]+)", re.IGNORECASE)


def parse_fastag_qr_payload(text: str) -> tuple[str, str] | None:
    """Returns (tag_id, bank_code) if `text` matches the expected FASTag QR
    schema, else None (treated as unreadable, per the agreed design -- never
    force a partial/best-guess parse).
    """
    m = _QR_PAYLOAD_RE.search(text)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper()


def read_qr(image_path: str) -> QrRead:
    results = zxingcpp.read_barcodes(Image.open(image_path), formats=zxingcpp.BarcodeFormat.QRCode)
    if not results:
        return QrRead(readable=False, tag_id=None, bank_code=None, raw_text=None)
    raw_text = results[0].text
    parsed = parse_fastag_qr_payload(raw_text)
    if parsed is None:
        return QrRead(readable=False, tag_id=None, bank_code=None, raw_text=raw_text)
    tag_id, bank_code = parsed
    return QrRead(readable=True, tag_id=tag_id, bank_code=bank_code, raw_text=raw_text)
