"""JSON schema for the one vision call in the FASTag flow. `additionalProperties:
false` + every property in `required` is needed for OpenRouter's `strict: true`
mode -- without it, some models silently drop fields instead of erroring.

Barcode and QR reading are NOT part of this schema -- they're decoded
deterministically by barcode_qr.py, not read by the model (see prompts.py's
explicit instruction not to attempt it).
"""
from __future__ import annotations

FASTAG_FRONT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "fastag_fully_framed": {
            "type": "boolean",
            "description": (
                "True only if a FASTag sticker is fully visible in frame, affixed to a "
                "windshield -- not cropped out of frame, not a loose tag photographed off "
                "the vehicle, not a screenshot."
            ),
        },
        "is_altered_or_ai_generated": {
            "type": "boolean",
            "description": (
                "True if the image looks tampered, edited, AI-generated/synthetic, "
                "a screenshot/UI capture, or a re-photographed printed photo."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0-1 confidence in BOTH verdicts above, taken together.",
        },
        "reasoning": {"type": "string", "description": "One or two sentences, for a human reviewer."},
    },
    "required": ["fastag_fully_framed", "is_altered_or_ai_generated", "confidence", "reasoning"],
    "additionalProperties": False,
}
