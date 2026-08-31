"""One-off diagnostic: shows exactly what the side/axle-image identity checks saw --
both which bucket (vrn_visible / corner_view / pure_side_profile) the type
classifier routed this photo into, and (if it lands in pure_side_profile) what the
make classifier read from it. Both are SigLIP zero-shot judgment calls and both can
be wrong -- this prints the full probability breakdown for each instead of just the
single winning label, so you can see how close/uncertain the call actually was.

Usage:
    python diagnose_side_image.py path/to/your_photo.jpg
"""
import sys

from PIL import Image

from vfiv.backends.image_io import load_rgb_array
from vfiv.backends.siglip import get_make_classifier, get_side_image_type_classifier
from vfiv.backends.vehicle import get_vehicle_detector

path = sys.argv[1]
arr = load_rgb_array(path)

print("=== Bucket routing (SideImageTypeClassifier) ===")
type_clf = get_side_image_type_classifier()
bucket_probs = type_clf.s.zero_shot(Image.fromarray(arr), type_clf.LABELS)
for bucket, p in sorted(bucket_probs.items(), key=lambda kv: -kv[1]):
    print(f"  {bucket:18s} {p * 100:5.1f}%")
winning_bucket = max(bucket_probs, key=bucket_probs.get)
print(f"\n-> Routed to: {winning_bucket!r}")

if winning_bucket == "vrn_visible":
    print("\nvrn_visible bucket doesn't use the make classifier -- it re-runs the "
          "plate OCR/match instead. Nothing more to show here.")
    sys.exit(0)
if winning_bucket == "corner_view":
    print("\ncorner_view bucket no longer uses the make classifier -- identity now "
          "rests entirely on the front-photo embedding-similarity check, which needs "
          "a front_reference_image to run at all.")
    sys.exit(0)

print("\n=== Make read (MakeClassifier, pure_side_profile's only signal) ===")
det = get_vehicle_detector().best_truck(arr)
if det is None:
    print("No truck detected -- classifier ran on the FULL uncropped image.")
    crop = arr
else:
    print(f"Detected bbox: {det.bbox} (confidence {det.conf:.2f})")
    crop = arr[det.bbox[1]:det.bbox[3], det.bbox[0]:det.bbox[2]]

crop_path = "make_read_crop.jpg"
Image.fromarray(crop).save(crop_path)
print(f"Saved the exact crop the classifier saw to: {crop_path} -- open it and look.")

clf = get_make_classifier()
labels = {b: f"a photo of the front of a {b} truck" for b in clf.BRANDS}
probs = clf.s.zero_shot(Image.fromarray(crop), labels)
print("\nFull probability breakdown:")
for brand, p in sorted(probs.items(), key=lambda kv: -kv[1]):
    print(f"  {brand:15s} {p * 100:5.1f}%")
