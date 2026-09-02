"""JSON schemas for each check's structured output. `additionalProperties: false`
+ every property in `required` is needed for OpenRouter's `strict: true` mode —
without it, some models silently drop fields instead of erroring.
"""
from __future__ import annotations

FRONT_IMAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "vehicle_type_is_bus_or_truck": {
            "type": "boolean",
            "description": "True only if the main subject is clearly a bus or truck.",
        },
        "is_altered_or_ai_generated": {
            "type": "boolean",
            "description": (
                "True if the image looks tampered, edited, AI-generated/synthetic, "
                "a screenshot, or a re-photographed printed photo."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0-1 confidence in BOTH verdicts above, taken together.",
        },
        "reasoning": {"type": "string", "description": "One or two sentences, for a human reviewer."},
    },
    "required": ["vehicle_type_is_bus_or_truck", "is_altered_or_ai_generated", "confidence", "reasoning"],
    "additionalProperties": False,
}

PLATE_READ_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "plate_readable": {"type": "boolean", "description": "False if no plate is legible at all."},
        "plate_text": {
            "type": "string",
            "description": "Best-effort plate read, uppercase, no spaces/punctuation. Empty string if unreadable.",
        },
        "confidence": {"type": "number", "description": "0-1 confidence in plate_text."},
        "reasoning": {"type": "string"},
        "plate_visible": {
            "type": "boolean",
            "description": (
                "True if any part of a plate is visible in frame, even if plate_readable is "
                "false (e.g. angled, blurry, or partly cropped) -- used to black out the plate "
                "area before duplicate-image comparison, so this can be true independently of "
                "plate_readable."
            ),
        },
        "bbox_x_min": {
            "type": "number",
            "description": "Left edge of the plate's bounding box, as a 0-1 fraction of image width. 0 if plate_visible is false.",
        },
        "bbox_y_min": {
            "type": "number",
            "description": "Top edge of the plate's bounding box, as a 0-1 fraction of image height. 0 if plate_visible is false.",
        },
        "bbox_x_max": {
            "type": "number",
            "description": "Right edge of the plate's bounding box, as a 0-1 fraction of image width. 0 if plate_visible is false.",
        },
        "bbox_y_max": {
            "type": "number",
            "description": "Bottom edge of the plate's bounding box, as a 0-1 fraction of image height. 0 if plate_visible is false.",
        },
    },
    "required": ["plate_readable", "plate_text", "confidence", "reasoning", "plate_visible",
                 "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"],
    "additionalProperties": False,
}

MAKER_READ_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "maker_readable": {"type": "boolean", "description": "False if no brand/manufacturer marking is legible."},
        "maker_text": {
            "type": "string",
            "description": "Best-effort manufacturer/brand as painted or badged on the vehicle. Empty string if unreadable.",
        },
        "confidence": {"type": "number", "description": "0-1 confidence in maker_text."},
        "reasoning": {"type": "string"},
    },
    "required": ["maker_readable", "maker_text", "confidence", "reasoning"],
    "additionalProperties": False,
}
