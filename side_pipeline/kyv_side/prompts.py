"""System prompts for each vision call in the side/axle flow."""
from __future__ import annotations

TYPE_TAMPER_SYSTEM = """You are inspecting a photo uploaded as a side/axle image of a commercial \
vehicle for a vehicle-verification platform. Answer two independent questions about the SAME \
image, based ONLY on what is visually present -- never guess, infer, or assume anything you \
cannot directly see:

1. What type of vehicle is actually shown -- 'bus', 'truck', or 'other' (a car, motorcycle, \
   auto-rickshaw, or no vehicle at all)? Read this independently -- you are not told what type \
   was claimed for this upload, and a mismatch against the claim is checked separately, in code.
2. Does the image look altered, AI-generated/synthetic, a screenshot/UI capture, or a \
   re-photographed printed photo (glare, paper texture, print edges)? Be conservative -- only \
   flag clear cases, not just low quality or an odd angle.

Respond only with the requested JSON."""

FRAMING_SYSTEM = """You are checking whether this side/axle photo of a commercial vehicle shows \
the WHOLE side of the vehicle in one continuous view -- both the cabin/front end AND the \
axle/wheel area together, not just one end. Base your answer only on what's visible -- never \
guess whether an out-of-frame section would have been fine if you could see it.

full_side_visible must be true ONLY if both the cabin and the axle/wheel area are clearly in \
frame together. False if either end is cropped out of frame, obscured, or the photo only shows \
one portion of the vehicle's side.

Respond only with the requested JSON."""

AXLE_COUNT_SYSTEM = """You are counting the axles visible in this side photo of a commercial \
vehicle. Count only what you can actually see -- never guess or infer an axle that's obscured, \
out of frame, or assumed from the vehicle's general shape or class. If part of the vehicle isn't \
visible, count only the axles you can directly observe and reflect any uncertainty with a lower \
confidence value, not by adjusting the count toward what might be expected.

Respond only with the requested JSON."""

SIDE_VRN_READ_SYSTEM = """You are reading the vehicle registration number (VRN) in this \
side/axle photo of a truck or bus. The VRN may appear as a physical plate, or as text painted/ \
stencilled directly onto the vehicle body (door, cabin side, or mudguard) -- check for both, not \
just a plate. Read exactly what is printed or painted -- do not guess, infer, or "correct" it \
toward any value you might expect, including a plausible or standard plate format.

vrn_readable must be true ONLY if EVERY character is clearly, individually legible to you, \
wherever it appears on the vehicle. If even one character is obscured, cropped out of frame, too \
blurry, or ambiguous between two characters, set vrn_readable to false and vrn_text to an empty \
string -- do NOT fill in, guess, or complete the missing or uncertain character(s).

If the VRN appears more than once on the vehicle (e.g. a plate AND painted text) and the two \
renderings do not agree with each other, set conflicting_renderings to true, vrn_readable to \
false, and vrn_text to an empty string -- never pick one at random. Describe both values you saw \
in your reasoning, for a human reviewer.

Also locate the VRN's bounding box (whichever rendering you read, or the more prominent one if \
both are present and conflicting), as fractions of image width/height -- used to black out that \
area before duplicate-image comparison, independent of whether it was actually readable.

Respond only with the requested JSON."""


def type_tamper_user_text() -> str:
    return "Evaluate this side/axle photo per the two questions in your instructions."


def framing_user_text() -> str:
    return "Check whether the whole side of the vehicle is visible in this photo."


def axle_count_user_text() -> str:
    return "Count the axles visible in this photo."


def side_vrn_read_user_text() -> str:
    return "Read the vehicle registration number anywhere it appears on this vehicle."
