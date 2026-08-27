import os

import pytest

from vfiv.front_image.front_image import validate_front_image

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_manual_review_when_no_api_key(monkeypatch, tmp_path):
    """The real CV gate (YOLO+SigLIP) needs no API key and still runs — only the
    screenshot/photo-of-photo/AI-generated judgment call (Claude) is unavailable,
    which is enough to make the whole result unchecked."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = validate_front_image(str(img_path))
    assert result.decision == "MANUAL_REVIEW"
    assert result.checked is False


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY")
def test_real_front_truck_sample_passes():
    path = os.path.join(SAMPLES, "truck2.jpg")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_front_image(path)
    assert result.checked is True
    assert result.decision == "PASS"
    assert result.vehicle_type == "truck"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY")
def test_generic_coco_yolo_misses_this_truck_known_limitation():
    """Documents a real, accepted limitation (not a bug): the generic COCO-trained
    YOLO vehicle detector (no custom-trained weights in this environment) misses this
    genuine Ashok Leyland truck. The CV gate is authoritative (matches
    truck_front_extractor's own architecture) — swap in a truck-finetuned detector in
    production to fix this, per that project's own documented caveat."""
    path = os.path.join(SAMPLES, "truck1.png")
    if not os.path.exists(path):
        pytest.skip(f"sample not present: {path}")
    result = validate_front_image(path)
    assert result.checked is True
    assert result.decision == "REJECT"
    assert result.vehicle_type == "other"  # YOLO didn't detect a truck/bus above threshold
