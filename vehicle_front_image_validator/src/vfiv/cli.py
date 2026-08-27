import argparse
import json
import os
import sys

from vfiv import config
from vfiv.validators.combined import validate_upload
from vfiv.validators.duplicate_check import check_duplicate
from vfiv.validators.fastag_image.fastag_check import check_fastag_upload
from vfiv.validators.front_image.front_image import validate_front_image
from vfiv.validators.front_image.make_model_check import validate_make_model
from vfiv.validators.front_image.vrn_check import validate_vrn
from vfiv.validators.side_image.side_image_check import check_side_image_upload

VALIDATORS = {
    "front": validate_front_image,
    "vrn": validate_vrn,
    "make_model": validate_make_model,
    "combined": validate_upload,
    "duplicate": check_duplicate,      # not wired into "combined" — see validators/duplicate_check.py
    "fastag": check_fastag_upload,     # not wired into "combined" — see validators/fastag_image/fastag_check.py
    "side": check_side_image_upload,   # not wired into "combined" — see validators/side_image/side_image_check.py
}
def main() -> None:
    ap = argparse.ArgumentParser(description="Validate an uploaded vehicle document/image.")
    ap.add_argument("--image", required=True, help="Path to the uploaded image.")
    ap.add_argument("--type", choices=sorted(VALIDATORS), default="front", help="Which validator to run.")
    ap.add_argument("--vrn", help="VRN to check against, sent alongside the image "
                                  "(required for --type vrn|combined|duplicate; manual input for testing).")
    ap.add_argument("--make", help="Claimed make/manufacturer (required for --type make_model|combined).")
    ap.add_argument("--model-name", help="Claimed model (optional for --type make_model|combined — "
                                          "only enforced if read with high confidence).")
    ap.add_argument("--upload-id", help="Stable id for this upload (--type duplicate|side); "
                                         "defaults to the image's filename if omitted.")
    ap.add_argument("--image-type", choices=config.IMAGE_TYPES, default="front",
                     help="Which reference corpus to search/store against (--type duplicate only); "
                          "front/side/fastag are never compared against each other.")
    ap.add_argument("--no-store", action="store_true",
                     help="--type duplicate only: search/decide without storing this upload's "
                          "embedding (a pure lookup, doesn't grow the reference library).")
    ap.add_argument("--fastag-id", help="Claimed FASTag id (required for --type fastag).")
    ap.add_argument("--bank-code", help="Claimed issuing-bank code from the QR payload "
                                         "(optional for --type fastag).")
    ap.add_argument("--axle-count", type=int, help="Claimed axle count (required for --type side).")
    ap.add_argument("--front-reference", help="Path to this truck's on-file front photo "
                                               "(optional for --type side, corner_view bucket only).")
    ap.add_argument("--backend", help="Model backend override — printed-digit OCR backend for "
                                       "--type fastag ('rekognition'|'claude'|'gemini'), or the "
                                       "axle-count model backend for --type side ('claude'|'gemini'). "
                                       "Defaults to config.py's FASTAG_OCR_BACKEND/AXLE_COUNT_BACKEND.")
    args = ap.parse_args()

    if args.type == "vrn":
        if not args.vrn:
            ap.error("--vrn is required for --type vrn")
        result = validate_vrn(args.image, args.vrn)
    elif args.type == "make_model":
        if not args.make:
            ap.error("--make is required for --type make_model")
        result = validate_make_model(args.image, args.make, args.model_name)
    elif args.type == "combined":
        if not args.vrn or not args.make:
            ap.error("--vrn and --make are required for --type combined")
        result = validate_upload(args.image, args.vrn, args.make, args.model_name)
    elif args.type == "duplicate":
        if not args.vrn:
            ap.error("--vrn is required for --type duplicate")
        upload_id = args.upload_id or os.path.basename(args.image)
        result = check_duplicate(args.image, upload_id, args.vrn, image_type=args.image_type,
                                  store=not args.no_store)
    elif args.type == "fastag":
        if not args.fastag_id:
            ap.error("--fastag-id is required for --type fastag")
        fastag_kwargs = {"backend": args.backend} if args.backend else {}
        result = check_fastag_upload(args.image, args.fastag_id, args.bank_code, **fastag_kwargs)
    elif args.type == "side":
        if not args.vrn or not args.make or args.axle_count is None:
            ap.error("--vrn, --make, and --axle-count are required for --type side")
        upload_id = args.upload_id or os.path.basename(args.image)
        side_kwargs = {"axle_backend": args.backend} if args.backend else {}
        result = check_side_image_upload(args.image, args.vrn, args.make, args.axle_count,
                                          upload_id=upload_id, front_reference_image=args.front_reference,
                                          **side_kwargs)
    else:
        result = VALIDATORS[args.type](args.image)

    print(json.dumps(result.model_dump(), indent=2))
    sys.exit(0 if result.decision == "PASS" else 1)


if __name__ == "__main__":
    main()
