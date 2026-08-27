"""The ORIGINAL all-VLM prompts for Q1/Q2/Q3, from before each was rewritten to use
real CV/OCR models (see README.md's "Origin" section for the full history). Reconstructed
verbatim from source control history in this conversation — this repo isn't a git repo,
so there was no `git log` to pull them from directly.

These exist ONLY for the test/inference interface (``experiments/``) to let you compare
the original single-VLM-call design against the real-model rewrite and against Gemini,
side by side. Production (``front_image/``, ``side_image/``, ``fastag_image/``) does not use these.
"""

Q1_LEGACY_PROMPT = """You are inspecting an uploaded image for a document/image validation platform.
The upload is supposed to be a photograph of the FRONT of a truck/bus, of the REAL
physical vehicle. It may come straight from a camera OR from the phone's photo gallery/
camera roll (either is fine — do not penalise it for coming from a gallery). What matters
is that it is a direct, original photograph of the real vehicle — not a screenshot and not
a re-photograph of an existing picture.
Identify the MAIN vehicle, the viewpoint, and image-quality red flags.
Reply with STRICT JSON only:
{"vehicle_type":"truck|bus|car|other","view":"front|side|rear|other","is_front":true|false,"front_complete":true|false,"is_screenshot":true|false,"is_photo_of_photo":true|false,"ai_generated":true|false,"ai_confidence":0-100,"confidence":0-100,"reason":"<short>"}

Definitions:
- vehicle_type: the largest/main vehicle in the image.
- view: which face of the vehicle the camera is looking at. "front" = you can see
  the grille / windshield / front number-plate area head-on.
- is_front: true ONLY for a clear, mostly head-on FRONT view (not 3/4 side, not rear).
- front_complete: true ONLY if the WHOLE front of the truck is clearly visible and not
  cut off / cropped / heavily occluded / too far or too partial to read.
- is_screenshot: true if this is a screenshot, a photo OF a digital screen/monitor, or an
  image embedded in a web/app interface (browser chrome, toolbars, buttons, window
  borders, cursor, status bars, watermarks/overlaid UI text).
- is_photo_of_photo: true if this is a photograph of an existing PRINTED photo, poster, or
  picture of the truck — i.e. someone re-photographed a physical print rather than
  photographing the real vehicle. Look for: visible photo-paper texture/gloss/sheen, a
  physical print's border, edge, curl, or corner, glare/reflection off glossy paper, a
  hand/fingers holding the printout, or the truck's image sitting inside a visibly flat
  rectangular photograph within the frame. (Distinct from is_screenshot, which is about
  digital screens/UI — this is about physical prints/photos.)
- ai_generated: true if the image looks AI-GENERATED / synthetic / CGI / heavily AI-edited.
  Look for: implausible/garbled text on the plate or signage, melted or asymmetric
  badges/logos, impossible reflections or lighting, warped geometry, over-smooth or
  "too perfect" surfaces, fused/extra parts. Be conservative — real photos can look odd.
- ai_confidence: 0-100, how sure you are it is AI-generated (0 if it looks like a real photo).
- confidence: 0-100, your confidence in the vehicle_type / view classification.
- reason: one short phrase citing the deciding factor."""


Q2_LEGACY_PROMPT = """You are reading the license/registration plate on an Indian truck/bus, for a
document-validation platform. Look ONLY at the licence plate itself; ignore ALL other
painted text/stickers on the vehicle body — Indian trucks are covered in this kind of
decoration and NONE of it is the registration number. This includes but is not limited to:
  - owner / transport-company name, phone or helpline numbers, route or destination names
  - slogans, blessings and warnings, e.g. "Horn OK Please", "Use Dipper at Night",
    "Wait for Side", "Keep Distance", "Jai Mata Di", "Shubh Yatra", "Buri Nazar Wale Tera
    Muh Kala" (and countless regional-language equivalents/variants of these — treat any
    similar decorative phrase the same way even if it isn't in this list)
  - permit / carrier labels, e.g. "National Permit", "All India Permit", "Public Carrier",
    "Goods Carrier"
  - load/capacity (GVW) markings, chassis- or engine-number plates (separate small metal
    plates, NOT the registration plate)
  - windshield stickers: insurance, fitness, PUC, RTO/tax stickers, FASTag stickers
The ONLY text that counts as the plate reading is what's printed on the actual
registration/number plate.

Two things about the plate itself:
- LAYOUT: the registration number may be printed on ONE line or split across TWO
  stacked lines (very common on trucks/buses). If it's two lines, read them
  top-to-bottom and concatenate into the single registration number — do not treat
  the line break as a character, a space, or a reset.
- IND MARKING: newer High-Security Registration Plates (HSRP) carry a small "IND"
  mark with the Ashoka Chakra emblem (usually a blue strip, top-left of the plate) —
  this is a country/security marking, NOT part of the registration number. Exclude it
  entirely from the plate reading.
Reply with STRICT JSON only:
{"plate":"<registration number, letters/digits only, empty if unreadable>","plate_confidence":0-100,"plate_colour":"white|yellow|green|black|red|unknown","colour_confidence":0-100,"reason":"<short>"}

Definitions:
- plate: the characters on the number plate as best you can read them (no spaces/dashes,
  no "IND", both lines concatenated if the plate is two-line).
  Empty string if the plate isn't visible or is fully illegible.
- plate_confidence: 0-100, your confidence in the plate reading.
- plate_colour: the BACKGROUND colour of the plate — white (private vehicle), yellow
  (commercial/transport), green (electric vehicle), black (self-drive rental
  commercial), red (temporary registration / government), or unknown if not
  determinable.
- colour_confidence: 0-100, your confidence in the colour classification.
- reason: one short phrase citing the deciding factor."""


Q3_LEGACY_PROMPT = """You are identifying the manufacturer (make) and model of an Indian truck/bus
from its front view, for a document-validation platform.

Read the MANUFACTURER BRAND from the grille badge, logo, or wordmark (e.g. TATA, ASHOK
LEYLAND, EICHER, BHARATBENZ, MAHINDRA, VOLVO, SML ISUZU, FORCE, MAN, SCANIA) — this is
NOT the owner/transport-company name, a route/destination name, or any slogan painted on
the body (those are separate from the manufacturer's own badge/logo).

Separately, read the MODEL designation if a badge/sticker showing it is legible (e.g.
"407", "1616", "PRO 3015", "LPT 1613", "1617R") — this is often on a small badge near the
grille or on a side panel and may not always be visible/legible from the front; leave it
empty rather than guessing if you're not confident.

Reply with STRICT JSON only:
{"make":"<manufacturer brand as painted/badged, empty if unreadable>","make_confidence":0-100,"model":"<model designation as read/badged, empty if unreadable>","model_confidence":0-100,"reason":"<short>"}

Definitions:
- make: the manufacturer brand word/logo as shown on the vehicle. Empty string if no
  badge/logo is legible.
- make_confidence: 0-100, your confidence in the make reading.
- model: the specific model/variant designation as badged, if legible. Empty string if
  not shown or not confidently legible — do not guess.
- model_confidence: 0-100, your confidence in the model reading (0 if empty).
- reason: one short phrase citing the deciding factor."""
