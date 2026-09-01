"""One-off diagnostic: shows exactly what the side/axle-image identity check saw --
which bucket (vrn_visible / corner_view / pure_side_profile) the bucket-routing VLM
call picked, and -- for corner_view / pure_side_profile -- the CAB-ONLY crop region
it located in both the side/corner photo and the front reference photo (window/glass
and cargo box excluded), the resulting colour-histogram similarity, and the overall
identity decision. Saves both crops as image files so you can look at them directly
and confirm the VLM actually stayed within the cab.

Usage:
    python diagnose_side_image.py path/to/side_or_corner_photo.jpg [path/to/front_reference.jpg]

The front reference photo is optional -- without it you'll see bucket routing and the
side photo's own cab-crop, but not the colour-histogram comparison (same as the real
checks: no front_reference_image means the identity buckets can't verify anything).
"""
import sys

from PIL import Image

from vfiv.side_image.side_image_check import (
    _cab_crop,
    _color_histogram_similarity,
    check_side_identity,
    classify_front_reference_cab_crop,
    classify_side_image_type,
)

side_path = sys.argv[1]
ref_path = sys.argv[2] if len(sys.argv) > 2 else None


def _print_cab_crop(raw: dict) -> None:
    print(f"  cab_crop_visible: {raw.get('cab_crop_visible')}")
    if raw.get("cab_crop_visible"):
        print(f"  x: {raw['cab_x_start']:.2f} - {raw['cab_x_end']:.2f}   "
              f"y: {raw['cab_y_start']:.2f} - {raw['cab_y_end']:.2f}")


print("=== Bucket routing + cab-crop location (classify_side_image_type) ===")
type_raw = classify_side_image_type(side_path)
if not type_raw.get("checked"):
    print(f"Unavailable: {type_raw.get('error')}")
    sys.exit(1)

bucket = type_raw.get("bucket")
print(f"Bucket: {bucket!r}")
print(f"Reason: {type_raw.get('reason')}")
_print_cab_crop(type_raw)

side_cab = _cab_crop(side_path, type_raw)
if side_cab is not None:
    Image.fromarray(side_cab).save("cab_crop_side.jpg")
    print("Saved side-photo cab crop to: cab_crop_side.jpg")
else:
    print("Side-photo cab crop unavailable (not confidently located).")

if bucket == "vrn_visible":
    print("\nvrn_visible bucket doesn't use the colour histogram at all -- it re-runs "
          "the plate OCR/match instead. Nothing more to show here.")
    sys.exit(0)

if ref_path is None:
    print("\nNo front-reference photo given -- pass one as the 2nd argument to also see "
          "the reference cab-crop + colour-histogram similarity.")
    sys.exit(0)

print("\n=== Reference photo cab-crop (classify_front_reference_cab_crop) ===")
ref_raw = classify_front_reference_cab_crop(ref_path)
if not ref_raw.get("checked"):
    print(f"Unavailable: {ref_raw.get('error')}")
    sys.exit(1)
_print_cab_crop(ref_raw)

ref_cab = _cab_crop(ref_path, ref_raw)
if ref_cab is not None:
    Image.fromarray(ref_cab).save("cab_crop_reference.jpg")
    print("Saved reference-photo cab crop to: cab_crop_reference.jpg")
else:
    print("Reference-photo cab crop unavailable (not confidently located).")

if side_cab is not None and ref_cab is not None:
    similarity = _color_histogram_similarity(side_cab, ref_cab)
    print(f"\nColour-histogram similarity (cab-only): {similarity:.4f}")

print("\n=== Full check_side_identity() decision ===")
result = check_side_identity(
    side_path, claimed_vrn="(not checked in this diagnostic)",
    claimed_make="(not checked in this diagnostic)", front_reference_image=ref_path,
)
print(f"decision={result.decision}  bucket={result.identity_bucket}")
print(f"front_similarity={result.front_similarity}  color_hist_similarity={result.color_hist_similarity}")
print(f"reason: {result.reason}")
