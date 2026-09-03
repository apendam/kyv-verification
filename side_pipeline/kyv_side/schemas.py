"""JSON schemas for each vision call in the side/axle flow. `additionalProperties:
false` + every property in `required` is needed for OpenRouter's `strict: true`
mode -- without it, some models silently drop fields instead of erroring.
"""
from __future__ import annotations

TYPE_TAMPER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "detected_vehicle_type": {
            "type": "string",
            "enum": ["bus", "truck", "other"],
            "description": (
                "The main subject's actual vehicle type, read independently from the image alone -- "
                "'other' for a car, motorcycle, auto-rickshaw, or no vehicle at all. Never influenced "
                "by what type was claimed for this upload; compared against the claim in code."
            ),
        },
        "is_altered_or_ai_generated": {
            "type": "boolean",
            "description": (
                "True if the image looks tampered, edited, AI-generated/synthetic, "
                "a screenshot/UI capture, or a re-photographed printed photo."
            ),
        },
        "confidence": {"type": "number", "description": "0-1 confidence in BOTH verdicts above, taken together."},
        "reasoning": {"type": "string", "description": "One or two sentences, for a human reviewer."},
    },
    "required": ["detected_vehicle_type", "is_altered_or_ai_generated", "confidence", "reasoning"],
    "additionalProperties": False,
}

FRAMING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "full_side_visible": {
            "type": "boolean",
            "description": (
                "True only if the WHOLE side of the vehicle is visible in one continuous view -- "
                "both the cabin/front end AND the axle/wheel area in frame together. False if "
                "either end is cropped out, obscured, or only one portion is shown."
            ),
        },
        "confidence": {"type": "number", "description": "0-1 confidence."},
        "reasoning": {"type": "string", "description": "One or two sentences, for a human reviewer."},
        "vehicle_bbox_x_min": {
            "type": "number",
            "description": (
                "Left edge of the bounding box tightly enclosing the main vehicle (not the "
                "background/environment around it), as a 0-1 fraction of image width. 0 if no "
                "vehicle is visible at all."
            ),
        },
        "vehicle_bbox_y_min": {
            "type": "number",
            "description": "Top edge of the vehicle's bounding box, as a 0-1 fraction of image height. 0 if no vehicle is visible.",
        },
        "vehicle_bbox_x_max": {
            "type": "number",
            "description": "Right edge of the vehicle's bounding box, as a 0-1 fraction of image width. 0 if no vehicle is visible.",
        },
        "vehicle_bbox_y_max": {
            "type": "number",
            "description": "Bottom edge of the vehicle's bounding box, as a 0-1 fraction of image height. 0 if no vehicle is visible.",
        },
    },
    "required": ["full_side_visible", "confidence", "reasoning",
                 "vehicle_bbox_x_min", "vehicle_bbox_y_min", "vehicle_bbox_x_max", "vehicle_bbox_y_max"],
    "additionalProperties": False,
}

AXLE_COUNT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "detected_axle_count": {
            "type": "integer",
            "description": (
                "The number of axles counted from the image, read independently -- never adjusted "
                "toward what might be expected or claimed for this upload."
            ),
        },
        "confidence": {"type": "number", "description": "0-1 confidence in the count."},
        "reasoning": {"type": "string", "description": "One or two sentences, for a human reviewer."},
    },
    "required": ["detected_axle_count", "confidence", "reasoning"],
    "additionalProperties": False,
}

SIDE_VRN_READ_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "vrn_visible": {
            "type": "boolean",
            "description": (
                "True if the VRN appears anywhere on the vehicle -- a physical plate, or text "
                "painted/stencilled directly on the body (door, cabin side, mudguard)."
            ),
        },
        "vrn_readable": {
            "type": "boolean",
            "description": (
                "False unless EVERY character is individually legible, wherever the VRN appears. "
                "Never true from completing/guessing. Always false if conflicting_renderings is true."
            ),
        },
        "vrn_text": {
            "type": "string",
            "description": (
                "The VRN exactly as read, uppercase, no spaces/punctuation. Empty string whenever "
                "vrn_readable is false."
            ),
        },
        "conflicting_renderings": {
            "type": "boolean",
            "description": (
                "True if the VRN appears more than once (e.g. a plate AND painted text) and the "
                "renderings do not agree. When true, vrn_readable must be false and vrn_text empty "
                "-- never pick one at random."
            ),
        },
        "confidence": {"type": "number", "description": "0-1 confidence in vrn_text."},
        "reasoning": {
            "type": "string",
            "description": "If conflicting_renderings is true, state both values seen, for a human reviewer.",
        },
        "bbox_x_min": {
            "type": "number",
            "description": "Left edge of the VRN's bounding box, as a 0-1 fraction of image width. 0 if vrn_visible is false.",
        },
        "bbox_y_min": {
            "type": "number",
            "description": "Top edge of the VRN's bounding box, as a 0-1 fraction of image height. 0 if vrn_visible is false.",
        },
        "bbox_x_max": {
            "type": "number",
            "description": "Right edge of the VRN's bounding box, as a 0-1 fraction of image width. 0 if vrn_visible is false.",
        },
        "bbox_y_max": {
            "type": "number",
            "description": "Bottom edge of the VRN's bounding box, as a 0-1 fraction of image height. 0 if vrn_visible is false.",
        },
    },
    "required": ["vrn_visible", "vrn_readable", "vrn_text", "conflicting_renderings", "confidence", "reasoning",
                 "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"],
    "additionalProperties": False,
}
