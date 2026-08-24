"""Tests for the Indian plate inference engine (pure logic, no models)."""

from truck_extract_match.plate.format import (confusable_distance, is_confusable,
                                              match_vrn, normalize_vrn,
                                              parse_and_correct)


def test_normalize_strips_noise():
    assert normalize_vrn("MH 12 AB-1234") == "MH12AB1234"
    assert normalize_vrn("IND MH12AB1234") == "MH12AB1234"
    assert normalize_vrn("") == ""


def test_confusable_pairs():
    assert is_confusable("0", "O")
    assert is_confusable("8", "B")
    assert is_confusable("5", "S")
    assert not is_confusable("A", "X")


def test_confusable_distance_zero_for_lookalikes():
    # O<->0 and B<->8 are free substitutions
    assert confusable_distance("MH12AB1234", "MH12AB1234") == 0
    assert confusable_distance("MH12A81234", "MH12AB1234") == 0  # 8 vs B


def test_exact_match():
    m = match_vrn("MH12AB1234", "MH12AB1234")
    assert m.matched and m.distance == 0 and not m.inferred


def test_smudged_digit_inferred():
    # OCR read the district '0' as 'O' and series 'B' as '8' -> still a match, flagged inferred
    m = match_vrn("MHO2A81234", "MH02AB1234")
    assert m.matched
    assert m.inferred
    assert m.distance == 0


def test_one_real_error_within_tolerance():
    # a genuine single-char error (2 vs 7, not confusable) is within default tolerance 1
    m = match_vrn("MH12AB1234", "MH12AB1734")
    assert m.matched and m.distance == 1


def test_true_mismatch_rejected():
    m = match_vrn("MH12AB1234", "KA05C9999")
    assert not m.matched


def test_parse_and_correct_fixes_state_class():
    # leading '1' should be corrected toward a letter for the state field
    p = parse_and_correct("MH12AB1234")
    assert p.state == "MH" and p.district == "12"
    assert p.number == "1234"
    assert p.valid


def test_parse_flags_unknown_state():
    p = parse_and_correct("XX12AB1234")
    assert not p.valid
    assert any("unknown state" in n for n in p.notes)
