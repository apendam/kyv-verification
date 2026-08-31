"""One-off diagnostic: shows exactly what the side-image make classifier saw.

Usage:
    python diagnose_make_read.py path/to/your_photo.jpg
"""
import sys

from PIL import Image

from vfiv.backends.image_io import load_rgb_array
from vfiv.backends.siglip import get_make_classifier
from vfiv.backends.vehicle import get_vehicle_detector

path = sys.argv[1]
arr = load_rgb_array(path)

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
