from vfiv.backends.vehicle import decide_vehicle_type_match


def test_matching_type_passes():
    result = decide_vehicle_type_match("truck", "truck")
    assert result["decision"] == "PASS"
    assert result["status"] == "MATCH"


def test_matching_type_is_case_insensitive():
    result = decide_vehicle_type_match("bus", "BUS")
    assert result["decision"] == "PASS"
    assert result["status"] == "MATCH"


def test_mismatched_type_is_manual_review_not_reject():
    """A generic COCO detector confusing truck/bus is a real, plausible failure
    mode -- never a solo REJECT, same posture as every other soft signal added
    to this codebase (colour-histogram, embedding-similarity, completeness)."""
    result = decide_vehicle_type_match("bus", "truck")
    assert result["decision"] == "MANUAL_REVIEW"
    assert result["status"] == "MISMATCH"
    assert "bus" in result["reason"] and "truck" in result["reason"]


def test_no_detection_is_manual_review():
    result = decide_vehicle_type_match(None, "truck")
    assert result["decision"] == "MANUAL_REVIEW"
    assert result["status"] == "UNREADABLE"
    assert "no truck/bus detected" in result["reason"]


def test_unknown_claimed_type_is_manual_review():
    result = decide_vehicle_type_match("truck", "car")
    assert result["decision"] == "MANUAL_REVIEW"
    assert result["status"] == "UNREADABLE"
    assert "unknown claimed vehicle type" in result["reason"]


def test_blank_claimed_type_is_manual_review():
    result = decide_vehicle_type_match("truck", "")
    assert result["decision"] == "MANUAL_REVIEW"
    assert result["status"] == "UNREADABLE"
