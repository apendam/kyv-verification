import argparse
import json
import sys

from vfiv.validators.combined import validate_upload
from vfiv.validators.front_image import validate_front_image
from vfiv.validators.make_model_check import validate_make_model
from vfiv.validators.vrn_check import validate_vrn

VALIDATORS = {
    "front": validate_front_image,
    "vrn": validate_vrn,
    "make_model": validate_make_model,
    "combined": validate_upload,
    # "side": validate_side_image,      # planned
    # "fastag": validate_fastag_image,  # planned
}
def main() -> None:
    ap = argparse.ArgumentParser(description="Validate an uploaded vehicle document/image.")
    ap.add_argument("--image", required=True, help="Path to the uploaded image.")
    ap.add_argument("--type", choices=sorted(VALIDATORS), default="front", help="Which validator to run.")
    ap.add_argument("--vrn", help="VRN to check against, sent alongside the image "
                                  "(required for --type vrn|combined; manual input for testing).")
    ap.add_argument("--make", help="Claimed make/manufacturer (required for --type make_model|combined).")
    ap.add_argument("--model-name", help="Claimed model (optional for --type make_model|combined — "
                                          "only enforced if read with high confidence).")
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
    else:
        result = VALIDATORS[args.type](args.image)

    print(json.dumps(result.model_dump(), indent=2))
    sys.exit(0 if result.decision == "PASS" else 1)


if __name__ == "__main__":
    main()
