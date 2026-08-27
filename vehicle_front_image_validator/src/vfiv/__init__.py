from vfiv.validators.combined import validate_upload
from vfiv.validators.duplicate_check import check_duplicate
from vfiv.validators.fastag_image.fastag_check import check_fastag_upload
from vfiv.validators.front_image.front_image import validate_front_image
from vfiv.validators.front_image.make_model_check import validate_make_model
from vfiv.validators.front_image.vrn_check import validate_vrn
from vfiv.validators.side_image.side_image_check import check_side_image_upload

__all__ = ["validate_front_image", "validate_vrn", "validate_make_model", "validate_upload",
           "check_duplicate", "check_fastag_upload", "check_side_image_upload"]
